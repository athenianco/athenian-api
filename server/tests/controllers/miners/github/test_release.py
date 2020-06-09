from datetime import datetime, timedelta, timezone
import marshal
from typing import Dict

from databases import Database
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import delete, select, sql

from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.controllers.miners.github.precomputed_prs import store_precomputed_done_times
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner, \
    PullRequestTimesMiner
from athenian.api.controllers.miners.github.release import _fetch_commit_history_dag, \
    _fetch_first_parents, extract_matched_bys_from_releases, load_releases, map_prs_to_releases, \
    map_releases_to_prs
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting
from athenian.api.models.metadata.github import PullRequest, Release
from athenian.api.models.precomputed.models import GitHubCommitFirstParents, GitHubCommitHistory


def generate_repo_settings(prs: pd.DataFrame) -> Dict[str, ReleaseMatchSetting]:
    return {
        "github.com/" + r: ReleaseMatchSetting(branches="", tags=".*", match=ReleaseMatch.tag)
        for r in prs[PullRequest.repository_full_name.key]
    }


async def test_map_prs_to_releases_cache(branches, default_branches, mdb, pdb, cache):
    prs = await read_sql_query(select([PullRequest]).where(PullRequest.number == 1126),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    settings = generate_repo_settings(prs)
    releases = await load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to, settings,
        mdb, pdb, None)
    matched_bys = extract_matched_bys_from_releases(releases)
    tag = "https://github.com/src-d/go-git/releases/tag/v4.12.0"
    for i in range(2):
        released_prs = await map_prs_to_releases(
            prs, releases, matched_bys, default_branches, time_to, settings, mdb, pdb, cache)
        assert len(cache.mem) > 0
        assert len(released_prs) == 1, str(i)
        assert released_prs.iloc[0][Release.published_at.key] == \
            pd.Timestamp("2019-06-18 22:57:34+0000", tzinfo=timezone.utc)
        assert released_prs.iloc[0][Release.author.key] == "mcuadros"
        assert released_prs.iloc[0][Release.url.key] == tag
    released_prs = await map_prs_to_releases(
        prs, releases, matched_bys, default_branches, time_to, settings, mdb, pdb, None)
    # the PR was merged and released in the past, we must detect that
    assert len(released_prs) == 1
    assert released_prs.iloc[0][Release.url.key] == tag


async def test_map_prs_to_releases_pdb(branches, default_branches, mdb, pdb, cache):
    prs = await read_sql_query(select([PullRequest]).where(PullRequest.number.in_((1126, 1180))),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    settings = generate_repo_settings(prs)
    releases = await load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to, settings,
        mdb, pdb, None)
    matched_bys = extract_matched_bys_from_releases(releases)
    released_prs = await map_prs_to_releases(
        prs, releases, matched_bys, default_branches, time_to, settings, mdb, pdb, None)
    assert len(released_prs) == 1
    released_prs = await map_prs_to_releases(
        prs, releases, matched_bys, default_branches, time_to, settings,
        Database("sqlite://"), pdb, None)
    assert len(released_prs) == 1


async def test_map_prs_to_releases_empty(branches, default_branches, mdb, pdb, cache):
    prs = await read_sql_query(select([PullRequest]).where(PullRequest.number == 1231),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    settings = generate_repo_settings(prs)
    releases = await load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to, settings,
        mdb, pdb, None)
    matched_bys = extract_matched_bys_from_releases(releases)
    for _ in range(2):
        released_prs = await map_prs_to_releases(
            prs, releases, matched_bys, default_branches, time_to, settings, mdb, pdb, cache)
        assert len(cache.mem) == 2
        assert released_prs.empty
    prs = prs.iloc[:0]
    released_prs = await map_prs_to_releases(
        prs, releases, matched_bys, default_branches, time_to, settings, mdb, pdb, cache)
    assert len(cache.mem) == 2
    assert released_prs.empty


async def test_map_prs_to_releases_precomputed_released(
        branches, default_branches, mdb, pdb, release_match_setting_tag):
    time_to = datetime(year=2019, month=8, day=2, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=2)

    miner = await PullRequestMiner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        branches, default_branches,
        False,
        release_match_setting_tag,
        mdb,
        pdb,
        None,
    )
    times_miner = PullRequestTimesMiner()
    true_prs = [pr for pr in miner if pr.release[Release.published_at.key] is not None]
    times = [times_miner(pr) for pr in true_prs]
    prs = pd.DataFrame([pr.pr for pr in true_prs]).set_index(PullRequest.node_id.key)
    releases = await load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to,
        release_match_setting_tag, mdb, pdb, None)
    matched_bys = extract_matched_bys_from_releases(releases)

    await pdb.execute(delete(GitHubCommitHistory))
    with pytest.raises(Exception):
        await map_prs_to_releases(
            prs, releases, matched_bys, default_branches, time_to, release_match_setting_tag,
            Database("sqlite://"), pdb, None)

    await store_precomputed_done_times(
        true_prs, times, default_branches, release_match_setting_tag, pdb)

    released_prs = await map_prs_to_releases(
        prs, releases, matched_bys, default_branches, time_to, release_match_setting_tag,
        Database("sqlite://"), pdb, None)
    assert len(released_prs) == len(prs)


async def test_map_releases_to_prs_early_merges(
        branches, default_branches, mdb, pdb, release_match_setting_tag):
    prs, releases, matched_bys = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2018, month=1, day=7, tzinfo=timezone.utc),
        datetime(year=2018, month=1, day=9, tzinfo=timezone.utc),
        [], [],
        release_match_setting_tag, mdb, pdb, None)
    assert (prs[PullRequest.merged_at.key] >
            datetime(year=2017, month=9, day=4, tzinfo=timezone.utc)).all()


async def test_map_releases_to_prs_smoke(
        branches, default_branches, mdb, pdb, cache, release_match_setting_tag):
    for _ in range(2):
        prs, releases, matched_bys = await map_releases_to_prs(
            ["src-d/go-git"],
            branches, default_branches,
            datetime(year=2019, month=7, day=31, tzinfo=timezone.utc),
            datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
            [], [],
            release_match_setting_tag, mdb, pdb, cache)
        assert len(prs) == 7
        assert (prs[PullRequest.merged_at.key] < pd.Timestamp(
            "2019-07-31 00:00:00", tzinfo=timezone.utc)).all()
        assert (prs[PullRequest.merged_at.key] > pd.Timestamp(
            "2019-06-19 00:00:00", tzinfo=timezone.utc)).all()
        assert len(releases) == 2
        assert set(releases[Release.sha.key]) == {
            "0d1a009cbb604db18be960db5f1525b99a55d727",
            "6241d0e70427cb0db4ca00182717af88f638268c",
        }
        assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}


async def test_map_releases_to_prs_empty(
        branches, default_branches, mdb, pdb, cache, release_match_setting_tag):
    prs, releases, matched_bys = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2019, month=7, day=1, tzinfo=timezone.utc),
        datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
        [], [], release_match_setting_tag, mdb, pdb, cache)
    assert prs.empty
    assert len(cache.mem) == 1
    assert len(releases) == 2
    assert set(releases[Release.sha.key]) == {
        "0d1a009cbb604db18be960db5f1525b99a55d727",
        "6241d0e70427cb0db4ca00182717af88f638268c",
    }
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
    prs, releases, matched_bys = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2019, month=7, day=1, tzinfo=timezone.utc),
        datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
        [], [], {
            "github.com/src-d/go-git": ReleaseMatchSetting(
                branches="master", tags=".*", match=ReleaseMatch.branch),
        }, mdb, pdb, cache)
    assert prs.empty
    assert len(cache.mem) == 8
    assert len(releases) == 19
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}


async def test_map_releases_to_prs_blacklist(
        branches, default_branches, mdb, pdb, cache, release_match_setting_tag):
    prs, releases, matched_bys = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2019, month=7, day=31, tzinfo=timezone.utc),
        datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
        [], [], release_match_setting_tag, mdb, pdb, cache,
        pr_blacklist=PullRequest.node_id.notin_([
            "MDExOlB1bGxSZXF1ZXN0Mjk3Mzk1Mzcz", "MDExOlB1bGxSZXF1ZXN0Mjk5NjA3MDM2",
            "MDExOlB1bGxSZXF1ZXN0MzAxODQyNDg2", "MDExOlB1bGxSZXF1ZXN0Mjg2ODczMDAw",
            "MDExOlB1bGxSZXF1ZXN0Mjk0NTUyNTM0", "MDExOlB1bGxSZXF1ZXN0MzAyMTMwODA3",
            "MDExOlB1bGxSZXF1ZXN0MzAyMTI2ODgx",
        ]))
    assert prs.empty
    assert len(releases) == 2
    assert set(releases[Release.sha.key]) == {
        "0d1a009cbb604db18be960db5f1525b99a55d727",
        "6241d0e70427cb0db4ca00182717af88f638268c",
    }
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}


@pytest.mark.parametrize("authors, mergers, n", [(["mcuadros"], [], 42),
                                                 ([], ["mcuadros"], 147),
                                                 (["mcuadros"], ["mcuadros"], 147)])
async def test_map_releases_to_prs_authors_mergers(
        branches, default_branches, mdb, pdb, cache,
        release_match_setting_tag, authors, mergers, n):
    prs, releases, matched_bys = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2019, month=7, day=31, tzinfo=timezone.utc),
        datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
        authors, mergers, release_match_setting_tag, mdb, pdb, cache)
    assert prs.size == n
    assert len(releases) == 2
    assert set(releases[Release.sha.key]) == {
        "0d1a009cbb604db18be960db5f1525b99a55d727",
        "6241d0e70427cb0db4ca00182717af88f638268c",
    }
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}


async def test_map_releases_to_prs_hard(
        branches, default_branches, mdb, pdb, cache, release_match_setting_tag):
    prs, releases, matched_bys = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2019, month=6, day=18, tzinfo=timezone.utc),
        datetime(year=2019, month=6, day=30, tzinfo=timezone.utc),
        [], [],
        release_match_setting_tag, mdb, pdb, cache)
    assert len(prs) == 24
    assert len(releases) == 1
    assert set(releases[Release.sha.key]) == {
        "f9a30199e7083bdda8adad3a4fa2ec42d25c1fdb",
    }
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}


async def test_map_prs_to_releases_smoke_metrics(branches, default_branches, mdb, pdb):
    time_from = datetime(year=2015, month=10, day=13, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=1, day=24, tzinfo=timezone.utc)
    filters = [
        sql.and_(PullRequest.merged_at > time_from, PullRequest.created_at < time_to),
        PullRequest.repository_full_name.in_(["src-d/go-git"]),
        PullRequest.user_login.in_(["mcuadros", "vmarkovtsev"]),
    ]
    prs = await read_sql_query(select([PullRequest]).where(sql.and_(*filters)),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    settings = generate_repo_settings(prs)
    releases = await load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to, settings,
        mdb, pdb, None)
    matched_bys = extract_matched_bys_from_releases(releases)
    released_prs = await map_prs_to_releases(
        prs, releases, matched_bys, default_branches, time_to, settings, mdb, pdb, None)
    assert set(released_prs[Release.url.key].unique()) == {
        "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc10",
        "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc11",
        "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc13",
        "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc12",
        "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc14",
        "https://github.com/src-d/go-git/releases/tag/v4.0.0-rc15",
        "https://github.com/src-d/go-git/releases/tag/v4.0.0",
        "https://github.com/src-d/go-git/releases/tag/v4.2.0",
        "https://github.com/src-d/go-git/releases/tag/v4.1.1",
        "https://github.com/src-d/go-git/releases/tag/v4.2.1",
        "https://github.com/src-d/go-git/releases/tag/v4.5.0",
        "https://github.com/src-d/go-git/releases/tag/v4.11.0",
        "https://github.com/src-d/go-git/releases/tag/v4.7.1",
        "https://github.com/src-d/go-git/releases/tag/v4.8.0",
        "https://github.com/src-d/go-git/releases/tag/v4.10.0",
        "https://github.com/src-d/go-git/releases/tag/v4.12.0",
        "https://github.com/src-d/go-git/releases/tag/v4.13.0",
    }


def check_branch_releases(releases: pd.DataFrame, n: int, date_from: datetime, date_to: datetime):
    assert len(releases) == n
    assert "mcuadros" in set(releases[Release.author.key])
    assert len(releases[Release.commit_id.key].unique()) == n
    assert releases[Release.id.key].all()
    assert all(len(n) == 40 for n in releases[Release.name.key])
    assert releases[Release.published_at.key].between(date_from, date_to).all()
    assert (releases[Release.repository_full_name.key] == "src-d/go-git").all()
    assert all(len(n) == 40 for n in releases[Release.sha.key])
    assert len(releases[Release.sha.key].unique()) == n
    assert (~releases[Release.tag.key].values.astype(bool)).all()
    assert releases[Release.url.key].str.startswith("http").all()


@pytest.mark.parametrize("branches_", ["{{default}}", "master", "m.*"])
async def test_load_releases_branches(branches, default_branches, mdb, pdb, cache, branches_):
    date_from = datetime(year=2017, month=10, day=13, tzinfo=timezone.utc)
    date_to = datetime(year=2020, month=1, day=24, tzinfo=timezone.utc)
    releases = await load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        date_from,
        date_to,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches=branches_, tags="", match=ReleaseMatch.branch)},
        mdb,
        pdb,
        cache,
    )
    check_branch_releases(releases, 240, date_from, date_to)


async def test_load_releases_branches_empty(branches, default_branches, mdb, pdb, cache):
    date_from = datetime(year=2017, month=10, day=13, tzinfo=timezone.utc)
    date_to = datetime(year=2020, month=1, day=24, tzinfo=timezone.utc)
    releases = await load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        date_from,
        date_to,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="unknown", tags="", match=ReleaseMatch.branch)},
        mdb,
        pdb,
        cache,
    )
    assert len(releases) == 0


@pytest.mark.parametrize("date_from, n", [
    (datetime(year=2017, month=10, day=4, tzinfo=timezone.utc), 45),
    (datetime(year=2017, month=9, day=4, tzinfo=timezone.utc), 1),
    (datetime(year=2017, month=12, day=8, tzinfo=timezone.utc), 0),
])
async def test_load_releases_tag_or_branch_dates(
        branches, default_branches, mdb, pdb, cache, date_from, n):
    date_to = datetime(year=2017, month=12, day=8, tzinfo=timezone.utc)
    releases = await load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        date_from,
        date_to,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="master", tags=".*", match=ReleaseMatch.tag_or_branch)},
        mdb,
        pdb,
        cache,
    )
    if n > 1:
        check_branch_releases(releases, n, date_from, date_to)
    else:
        assert len(releases) == n


async def test_load_releases_tag_or_branch_initial(branches, default_branches, mdb, pdb):
    date_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2015, month=10, day=22, tzinfo=timezone.utc)
    releases = await load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        date_from,
        date_to,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="master", tags="", match=ReleaseMatch.branch)},
        mdb,
        pdb,
        None,
    )
    check_branch_releases(releases, 17, date_from, date_to)


async def test_map_releases_to_prs_branches(branches, default_branches, mdb, pdb):
    date_from = datetime(year=2015, month=4, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2015, month=5, day=1, tzinfo=timezone.utc)
    prs, releases, matched_bys = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        date_from,
        date_to,
        [], [],
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="master", tags="", match=ReleaseMatch.branch)},
        mdb,
        pdb,
        None)
    assert prs.empty
    assert len(releases) == 1
    assert releases[Release.sha.key][0] == "5d7303c49ac984a9fec60523f2d5297682e16646"
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}


@pytest.mark.parametrize("repos", [["src-d/gitbase"], []])
async def test_load_releases_empty(branches, default_branches, mdb, pdb, repos):
    releases = await load_releases(
        repos,
        branches, default_branches,
        datetime(year=2020, month=6, day=30, tzinfo=timezone.utc),
        datetime(year=2020, month=7, day=30, tzinfo=timezone.utc),
        {"github.com/src-d/gitbase": ReleaseMatchSetting(
            branches=".*", tags=".*", match=ReleaseMatch.branch)},
        mdb,
        pdb,
        None,
        index=Release.id.key)
    assert releases.empty
    date_from = datetime(year=2017, month=3, day=4, tzinfo=timezone.utc)
    date_to = datetime(year=2017, month=12, day=8, tzinfo=timezone.utc)
    releases = await load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        date_from,
        date_to,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="master", tags="", match=ReleaseMatch.tag)},
        mdb,
        pdb,
        None,
    )
    assert releases.empty
    releases = await load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        date_from,
        date_to,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="", tags=".*", match=ReleaseMatch.branch)},
        mdb,
        pdb,
        None,
    )
    assert releases.empty


async def test_fetch_commit_history_smoke(mdb, pdb):
    dag = await _fetch_commit_history_dag(
        None,
        "src-d/go-git",
        ["MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw==",
         "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ=="],
        ["d2a38b4a5965d529566566640519d03d2bd10f6c",
         "31eae7b619d166c366bf5df4991f04ba8cebea0a"],
        [0, 0],
        mdb, pdb, None)
    dag["31eae7b619d166c366bf5df4991f04ba8cebea0a"] = set(
        dag["31eae7b619d166c366bf5df4991f04ba8cebea0a"])
    ground_truth = {
        "31eae7b619d166c366bf5df4991f04ba8cebea0a": {"b977a025ca21e3b5ca123d8093bd7917694f6da7",
                                                     "d2a38b4a5965d529566566640519d03d2bd10f6c"},
        "b977a025ca21e3b5ca123d8093bd7917694f6da7": ["35b585759cbf29f8ec428ef89da20705d59f99ec"],
        "d2a38b4a5965d529566566640519d03d2bd10f6c": ["35b585759cbf29f8ec428ef89da20705d59f99ec"],
        "35b585759cbf29f8ec428ef89da20705d59f99ec": ["c2bbf9fe8009b22d0f390f3c8c3f13937067590f"],
        "c2bbf9fe8009b22d0f390f3c8c3f13937067590f": ["fc9f0643b21cfe571046e27e0c4565f3a1ee96c8"],
        "fc9f0643b21cfe571046e27e0c4565f3a1ee96c8": ["c088fd6a7e1a38e9d5a9815265cb575bb08d08ff"],
        "c088fd6a7e1a38e9d5a9815265cb575bb08d08ff": ["5fddbeb678bd2c36c5e5c891ab8f2b143ced5baf"],
        "5fddbeb678bd2c36c5e5c891ab8f2b143ced5baf": ["5d7303c49ac984a9fec60523f2d5297682e16646"],
        "5d7303c49ac984a9fec60523f2d5297682e16646": [],
    }
    assert dag == ground_truth
    dag = await _fetch_commit_history_dag(
        marshal.dumps(dag),
        "src-d/go-git",
        ["MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw==",
         "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ=="],
        ["31eae7b619d166c366bf5df4991f04ba8cebea0a",
         "d2a38b4a5965d529566566640519d03d2bd10f6c"],
        [0, 0],
        Database("sqlite://"), pdb, None)
    assert dag == ground_truth
    with pytest.raises(Exception):
        await _fetch_commit_history_dag(
            marshal.dumps(dag),
            "src-d/go-git",
            ["MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw==",
             "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ=="],
            ["31eae7b619d166c366bf5df4991f04ba8cebea0a",
             "1353ccd6944ab41082099b79979ded3223db98ec"],
            [0, 0],
            Database("sqlite://"), pdb,
            None,
        )


async def test_fetch_commit_history_initial_commit(mdb, pdb):
    dag = await _fetch_commit_history_dag(
        None,
        "src-d/go-git",
        ["MDY6Q29tbWl0NDQ3MzkwNDQ6NWQ3MzAzYzQ5YWM5ODRhOWZlYzYwNTIzZjJkNTI5NzY4MmUxNjY0Ng=="],
        ["5d7303c49ac984a9fec60523f2d5297682e16646"],
        [0],
        mdb, pdb, None)
    assert dag == {"5d7303c49ac984a9fec60523f2d5297682e16646": []}


async def test_fetch_commit_history_cache(mdb, pdb, cache):
    dag = await _fetch_commit_history_dag(
        None,
        "src-d/go-git",
        ["MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw==",
         "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ=="],
        ["d2a38b4a5965d529566566640519d03d2bd10f6c",
         "31eae7b619d166c366bf5df4991f04ba8cebea0a"],
        [0, 0],
        mdb, pdb, cache)
    dag["31eae7b619d166c366bf5df4991f04ba8cebea0a"] = set(
        dag["31eae7b619d166c366bf5df4991f04ba8cebea0a"])
    ground_truth = {
        "31eae7b619d166c366bf5df4991f04ba8cebea0a": {"b977a025ca21e3b5ca123d8093bd7917694f6da7",
                                                     "d2a38b4a5965d529566566640519d03d2bd10f6c"},
        "b977a025ca21e3b5ca123d8093bd7917694f6da7": ["35b585759cbf29f8ec428ef89da20705d59f99ec"],
        "d2a38b4a5965d529566566640519d03d2bd10f6c": ["35b585759cbf29f8ec428ef89da20705d59f99ec"],
        "35b585759cbf29f8ec428ef89da20705d59f99ec": ["c2bbf9fe8009b22d0f390f3c8c3f13937067590f"],
        "c2bbf9fe8009b22d0f390f3c8c3f13937067590f": ["fc9f0643b21cfe571046e27e0c4565f3a1ee96c8"],
        "fc9f0643b21cfe571046e27e0c4565f3a1ee96c8": ["c088fd6a7e1a38e9d5a9815265cb575bb08d08ff"],
        "c088fd6a7e1a38e9d5a9815265cb575bb08d08ff": ["5fddbeb678bd2c36c5e5c891ab8f2b143ced5baf"],
        "5fddbeb678bd2c36c5e5c891ab8f2b143ced5baf": ["5d7303c49ac984a9fec60523f2d5297682e16646"],
        "5d7303c49ac984a9fec60523f2d5297682e16646": [],
    }
    assert dag == ground_truth
    dag = await _fetch_commit_history_dag(
        None,
        "src-d/go-git",
        ["MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ==",
         "MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw=="],
        ["d2a38b4a5965d529566566640519d03d2bd10f6c",
         "31eae7b619d166c366bf5df4991f04ba8cebea0a"],
        [0, 0],
        None, None, cache)
    dag["31eae7b619d166c366bf5df4991f04ba8cebea0a"] = set(
        dag["31eae7b619d166c366bf5df4991f04ba8cebea0a"])
    assert dag == ground_truth


async def test_fetch_commit_history_many(mdb, pdb):
    commit_ids = [
        "MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw==",
        "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ=="] * 50
    commit_shas = ["d2a38b4a5965d529566566640519d03d2bd10f6c",
                   "31eae7b619d166c366bf5df4991f04ba8cebea0a"] * 50
    dag = await _fetch_commit_history_dag(
        None,
        "src-d/go-git",
        np.asarray(commit_ids),
        np.asarray(commit_shas),
        np.zeros(len(commit_ids)),
        mdb, pdb, None)
    assert dag


async def test_fetch_first_parents_smoke(mdb, pdb):
    fp = await _fetch_first_parents(
        None,
        "src-d/go-git",
        ["MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw==",
         "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ=="],
        datetime(2015, 4, 5),
        datetime(2015, 5, 20),
        mdb, pdb, None)
    ground_truth = {
        "MDY6Q29tbWl0NDQ3MzkwNDQ6NWQ3MzAzYzQ5YWM5ODRhOWZlYzYwNTIzZjJkNTI5NzY4MmUxNjY0Ng==",
        "MDY6Q29tbWl0NDQ3MzkwNDQ6NWZkZGJlYjY3OGJkMmMzNmM1ZTVjODkxYWI4ZjJiMTQzY2VkNWJhZg==",
        "MDY6Q29tbWl0NDQ3MzkwNDQ6YzA4OGZkNmE3ZTFhMzhlOWQ1YTk4MTUyNjVjYjU3NWJiMDhkMDhmZg==",
    }
    assert fp == ground_truth
    obj = await pdb.fetch_val(select([GitHubCommitFirstParents.commits]))
    fp = await _fetch_first_parents(
        obj,
        "src-d/go-git",
        ["MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw==",
         "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ=="],
        datetime(2015, 4, 5),
        datetime(2015, 5, 20),
        Database("sqlite://"), pdb, None)
    assert fp == ground_truth
    with pytest.raises(Exception):
        await _fetch_first_parents(
            obj,
            "src-d/go-git",
            ["MDY6Q29tbWl0NDQ3MzkwNDQ6OTQwNDYwZjU0MjJiMDJmMDEzNTEzOTZhZjcwM2U5YjYzZTg1OTZhZQ=="],
            datetime(2015, 4, 5),
            datetime(2015, 5, 20),
            Database("sqlite://"), pdb, None)


async def test_fetch_first_parents_initial_commit(mdb, pdb):
    fp = await _fetch_first_parents(
        None,
        "src-d/go-git",
        ["MDY6Q29tbWl0NDQ3MzkwNDQ6NWQ3MzAzYzQ5YWM5ODRhOWZlYzYwNTIzZjJkNTI5NzY4MmUxNjY0Ng=="],
        datetime(2015, 4, 5),
        datetime(2015, 5, 20),
        mdb, pdb, None)
    assert fp == {
        "MDY6Q29tbWl0NDQ3MzkwNDQ6NWQ3MzAzYzQ5YWM5ODRhOWZlYzYwNTIzZjJkNTI5NzY4MmUxNjY0Ng==",
    }
    fp = await _fetch_first_parents(
        None,
        "src-d/go-git",
        ["MDY6Q29tbWl0NDQ3MzkwNDQ6NWQ3MzAzYzQ5YWM5ODRhOWZlYzYwNTIzZjJkNTI5NzY4MmUxNjY0Ng=="],
        datetime(2015, 3, 5),
        datetime(2015, 3, 20),
        mdb, pdb, None)
    assert fp == set()


async def test_fetch_first_parents_index_error(mdb, pdb):
    fp1 = await _fetch_first_parents(
        None,
        "src-d/go-git",
        ["MDY6Q29tbWl0NDQ3MzkwNDQ6NWQ3MzAzYzQ5YWM5ODRhOWZlYzYwNTIzZjJkNTI5NzY4MmUxNjY0Ng=="],
        datetime(2015, 4, 5),
        datetime(2015, 5, 20),
        mdb, pdb, None)
    data = await pdb.fetch_val(select([GitHubCommitFirstParents.commits]))
    assert data
    fp2 = await _fetch_first_parents(
        data,
        "src-d/go-git",
        ["MDY6Q29tbWl0NDQ3MzkwNDQ6NWZkZGJlYjY3OGJkMmMzNmM1ZTVjODkxYWI4ZjJiMTQzY2VkNWJhZg=="],
        datetime(2015, 4, 5),
        datetime(2015, 5, 20),
        mdb, pdb, None)
    assert fp1 != fp2


async def test_fetch_first_parents_cache(mdb, pdb, cache):
    await _fetch_first_parents(
        None,
        "src-d/go-git",
        ["MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw==",
         "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ=="],
        datetime(2015, 4, 5),
        datetime(2015, 5, 20),
        mdb, pdb, cache)
    ground_truth = {
        "MDY6Q29tbWl0NDQ3MzkwNDQ6NWQ3MzAzYzQ5YWM5ODRhOWZlYzYwNTIzZjJkNTI5NzY4MmUxNjY0Ng==",
        "MDY6Q29tbWl0NDQ3MzkwNDQ6NWZkZGJlYjY3OGJkMmMzNmM1ZTVjODkxYWI4ZjJiMTQzY2VkNWJhZg==",
        "MDY6Q29tbWl0NDQ3MzkwNDQ6YzA4OGZkNmE3ZTFhMzhlOWQ1YTk4MTUyNjVjYjU3NWJiMDhkMDhmZg==",
    }
    fp = await _fetch_first_parents(
        None,
        "src-d/go-git",
        ["MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ==",
         "MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw=="],
        datetime(2015, 4, 5),
        datetime(2015, 5, 20),
        None, None, cache)
    assert fp == ground_truth
    with pytest.raises(Exception):
        await _fetch_first_parents(
            None,
            "src-d/go-git",
            ["MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ==",
             "MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw=="],
            datetime(2015, 4, 6),
            datetime(2015, 5, 20),
            None, None, cache)
