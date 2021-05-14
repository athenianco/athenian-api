from collections import defaultdict
from datetime import datetime, timedelta, timezone
import pickle

from databases import Database
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from sqlalchemy import delete, insert, select, sql
from sqlalchemy.schema import CreateTable

from athenian.api.async_utils import read_sql_query
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.bots import bots
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.commit import _empty_dag, _fetch_commit_history_dag, \
    fetch_repository_commits
from athenian.api.controllers.miners.github.dag_accelerated import extract_subdag, join_dags, \
    mark_dag_access, mark_dag_parents, partition_dag
from athenian.api.controllers.miners.github.precomputed_prs import store_precomputed_done_facts
from athenian.api.controllers.miners.github.pull_request import PullRequestFactsMiner, \
    PullRequestMiner
from athenian.api.controllers.miners.github.release_load import \
    group_repos_by_release_match, ReleaseLoader
from athenian.api.controllers.miners.github.release_match import \
    _fetch_repository_first_commit_dates, _find_dead_merged_prs, map_prs_to_releases, \
    map_releases_to_prs
from athenian.api.controllers.miners.github.release_mine import mine_releases, \
    mine_releases_by_name
from athenian.api.controllers.miners.github.released_pr import matched_by_column
from athenian.api.controllers.miners.types import released_prs_columns
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting, ReleaseSettings
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.experiments.preloading.entries import PreloadedBranchMiner, \
    PreloadedReleaseLoader
from athenian.api.models.metadata.github import Branch, NodeCommit, PullRequest, \
    PullRequestLabel, Release
from athenian.api.models.persistentdata.models import ReleaseNotification
from athenian.api.models.precomputed.models import GitHubCommitHistory
from tests.controllers.test_filter_controller import force_push_dropped_go_git_pr_numbers


def generate_repo_settings(prs: pd.DataFrame) -> ReleaseSettings:
    return ReleaseSettings({
        "github.com/" + r: ReleaseMatchSetting(branches="", tags=".*", match=ReleaseMatch.tag)
        for r in prs[PullRequest.repository_full_name.key]
    })


@with_defer
async def test_map_prs_to_releases_cache(branches, default_branches, dag, mdb, pdb, rdb, cache,
                                         with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    prs = await read_sql_query(select([PullRequest]).where(PullRequest.number == 1126),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    prs["dead"] = False
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    settings = generate_repo_settings(prs)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to, settings,
        1, (6366825,), mdb, pdb, rdb, None)
    tag = "https://github.com/src-d/go-git/releases/tag/v4.12.0"
    for i in range(2):
        released_prs, facts, _ = await map_prs_to_releases(
            prs, releases, matched_bys, branches, default_branches, time_to, dag, settings,
            1, (6366825,), mdb, pdb, cache)
        await wait_deferred()
        assert isinstance(facts, dict)
        assert len(facts) == 0
        assert len(cache.mem) > 0
        assert len(released_prs) == 1, str(i)
        assert released_prs.iloc[0][Release.url.key] == tag
        assert released_prs.iloc[0][Release.published_at.key] == \
            pd.Timestamp("2019-06-18 22:57:34+0000", tzinfo=timezone.utc)
        assert released_prs.iloc[0][Release.author.key] == "mcuadros"
    released_prs, _, _ = await map_prs_to_releases(
        prs, releases, matched_bys, branches, default_branches, time_to, dag, settings,
        1, (6366825,), mdb, pdb, None)
    # the PR was merged and released in the past, we must detect that
    assert len(released_prs) == 1
    assert released_prs.iloc[0][Release.url.key] == tag


@with_defer
async def test_map_prs_to_releases_pdb(branches, default_branches, dag, mdb, pdb, rdb,
                                       with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    prs = await read_sql_query(select([PullRequest]).where(PullRequest.number.in_((1126, 1180))),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    prs["dead"] = False
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    settings = generate_repo_settings(prs)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to, settings,
        1, (6366825,), mdb, pdb, rdb, None)
    released_prs, _, _ = await map_prs_to_releases(
        prs, releases, matched_bys, branches, default_branches, time_to, dag, settings,
        1, (6366825,), mdb, pdb, None)
    await wait_deferred()
    assert len(released_prs) == 1
    dummy_mdb = Database("sqlite://", force_rollback=True)
    await dummy_mdb.connect()
    try:
        prlt = PullRequestLabel.__table__
        if prlt.schema:
            prlt.name = "%s.%s" % (prlt.schema, prlt.name)
            prlt.schema = None
        for table in (PullRequestLabel, NodeCommit):
            await dummy_mdb.execute(CreateTable(table.__table__))
        released_prs, _, _ = await map_prs_to_releases(
            prs, releases, matched_bys, branches, default_branches, time_to, dag, settings,
            1, (6366825,), dummy_mdb, pdb, None)
        assert len(released_prs) == 1
    finally:
        if "." in prlt.name:
            prlt.schema, prlt.name = prlt.name.split(".")
        await dummy_mdb.disconnect()


@with_defer
async def test_map_prs_to_releases_empty(branches, default_branches, dag, mdb, pdb, rdb, cache,
                                         with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    prs = await read_sql_query(select([PullRequest]).where(PullRequest.number == 1231),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    prs["dead"] = False
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    settings = generate_repo_settings(prs)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to, settings,
        1, (6366825,), mdb, pdb, rdb, None)
    for i in range(2):
        released_prs, _, _ = await map_prs_to_releases(
            prs, releases, matched_bys, branches, default_branches, time_to, dag, settings,
            1, (6366825,), mdb, pdb, cache)
        assert len(cache.mem) == 1, i
        assert released_prs.empty
    prs = prs.iloc[:0]
    released_prs, _, _ = await map_prs_to_releases(
        prs, releases, matched_bys, branches, default_branches, time_to, dag, settings,
        1, (6366825,), mdb, pdb, cache)
    assert len(cache.mem) == 1
    assert released_prs.empty


@with_defer
async def test_map_prs_to_releases_precomputed_released(
        branches, default_branches, dag, mdb, pdb, rdb, release_match_setting_tag,
        with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    time_to = datetime(year=2019, month=8, day=2, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=2)

    miner, _, _, _ = await PullRequestMiner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    true_prs = [pr for pr in miner if pr.release[Release.published_at.key] is not None]
    facts = [facts_miner(pr) for pr in true_prs]
    prs = pd.DataFrame([pr.pr for pr in true_prs]).set_index(PullRequest.node_id.key)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to,
        release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, None)

    await pdb.execute(delete(GitHubCommitHistory))
    dummy_mdb = Database("sqlite://", force_rollback=True)
    await dummy_mdb.connect()
    try:
        prlt = PullRequestLabel.__table__
        if prlt.schema:
            prlt.name = "%s.%s" % (prlt.schema, prlt.name)
            prlt.schema = None
        for table in (PullRequestLabel, NodeCommit):
            await dummy_mdb.execute(CreateTable(table.__table__))

        await store_precomputed_done_facts(
            true_prs, facts, default_branches, release_match_setting_tag, 1, pdb)

        released_prs, _, _ = await map_prs_to_releases(
            prs, releases, matched_bys, branches, default_branches, time_to, dag,
            release_match_setting_tag, 1, (6366825,), dummy_mdb, pdb, None)
        assert len(released_prs) == len(prs)
    finally:
        if "." in prlt.name:
            prlt.schema, prlt.name = prlt.name.split(".")
        await dummy_mdb.disconnect()


@with_defer
async def test_map_releases_to_prs_early_merges(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag):
    prs, releases, matched_bys, dag = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2018, month=1, day=7, tzinfo=timezone.utc),
        datetime(year=2018, month=1, day=9, tzinfo=timezone.utc),
        [], [], JIRAFilter.empty(),
        release_match_setting_tag, None, None, None, 1, (6366825,), mdb, pdb, rdb, None)
    assert len(prs) == 60
    assert (prs[PullRequest.merged_at.key] >
            datetime(year=2017, month=9, day=4, tzinfo=timezone.utc)).all()
    assert isinstance(dag, dict)
    dag = dag["src-d/go-git"]
    assert len(dag) == 3
    assert len(dag[0]) == 1015
    assert dag[0].dtype == np.dtype("S40")
    assert len(dag[1]) == 1016
    assert dag[1].dtype == np.uint32
    assert len(dag[2]) == dag[1][-1]
    assert dag[2].dtype == np.uint32


@with_defer
async def test_map_releases_to_prs_smoke(
        branches, default_branches, mdb, pdb, rdb, cache, release_match_setting_tag):
    for _ in range(2):
        prs, releases, matched_bys, dag = await map_releases_to_prs(
            ["src-d/go-git"],
            branches, default_branches,
            datetime(year=2019, month=7, day=31, tzinfo=timezone.utc),
            datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
            [], [], JIRAFilter.empty(),
            release_match_setting_tag, None, None, None, 1, (6366825,), mdb, pdb, rdb, cache)
        await wait_deferred()
        assert len(prs) == 7
        assert len(dag["src-d/go-git"][0]) == 1508
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


@with_defer
async def test_map_releases_to_prs_no_truncate(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag):
    prs, releases, matched_bys, _ = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2018, month=7, day=31, tzinfo=timezone.utc),
        datetime(year=2018, month=12, day=2, tzinfo=timezone.utc),
        [], [], JIRAFilter.empty(),
        release_match_setting_tag, None, None, None,
        1, (6366825,), mdb, pdb, rdb, None, truncate=False)
    assert len(prs) == 8
    assert len(releases) == 5 + 7
    assert releases[Release.published_at.key].is_monotonic_decreasing
    assert releases.index.is_monotonic
    assert "v4.13.1" in releases[Release.tag.key].values


@with_defer
async def test_map_releases_to_prs_empty(
        branches, default_branches, mdb, pdb, rdb, cache, release_match_setting_tag):
    prs, releases, matched_bys, _ = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2019, month=7, day=1, tzinfo=timezone.utc),
        datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
        [], [], JIRAFilter.empty(), release_match_setting_tag, None, None, None,
        1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    assert prs.empty
    assert len(cache.mem) == 3
    assert len(releases) == 2
    assert set(releases[Release.sha.key]) == {
        "0d1a009cbb604db18be960db5f1525b99a55d727",
        "6241d0e70427cb0db4ca00182717af88f638268c",
    }
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
    prs, releases, matched_bys, _ = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2019, month=7, day=1, tzinfo=timezone.utc),
        datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
        [], [], JIRAFilter.empty(), ReleaseSettings({
            "github.com/src-d/go-git": ReleaseMatchSetting(
                branches="master", tags=".*", match=ReleaseMatch.branch),
        }), None, None, None, 1, (6366825,), mdb, pdb, rdb, cache)
    assert prs.empty
    assert len(cache.mem) == 8
    assert len(releases) == 19
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}


@with_defer
async def test_map_releases_to_prs_blacklist(
        branches, default_branches, mdb, pdb, rdb, cache, release_match_setting_tag):
    prs, releases, matched_bys, _ = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2019, month=7, day=31, tzinfo=timezone.utc),
        datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
        [], [], JIRAFilter.empty(), release_match_setting_tag, None, None, None,
        1, (6366825,), mdb, pdb, rdb, cache,
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


@pytest.mark.parametrize("authors, mergers, n", [(["mcuadros"], [], 2),
                                                 ([], ["mcuadros"], 7),
                                                 (["mcuadros"], ["mcuadros"], 7)])
@with_defer
async def test_map_releases_to_prs_authors_mergers(
        branches, default_branches, mdb, pdb, rdb, cache,
        release_match_setting_tag, authors, mergers, n):
    prs, releases, matched_bys, _ = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2019, month=7, day=31, tzinfo=timezone.utc),
        datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
        authors, mergers, JIRAFilter.empty(), release_match_setting_tag,
        None, None, None, 1, (6366825,), mdb, pdb, rdb, cache)
    assert len(prs) == n
    assert len(releases) == 2
    assert set(releases[Release.sha.key]) == {
        "0d1a009cbb604db18be960db5f1525b99a55d727",
        "6241d0e70427cb0db4ca00182717af88f638268c",
    }
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}


@with_defer
async def test_map_releases_to_prs_hard(
        branches, default_branches, mdb, pdb, rdb, cache, release_match_setting_tag):
    prs, releases, matched_bys, _ = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2019, month=6, day=18, tzinfo=timezone.utc),
        datetime(year=2019, month=6, day=30, tzinfo=timezone.utc),
        [], [], JIRAFilter.empty(),
        release_match_setting_tag, None, None, None, 1, (6366825,), mdb, pdb, rdb, cache)
    assert len(prs) == 24
    assert len(releases) == 1
    assert set(releases[Release.sha.key]) == {
        "f9a30199e7083bdda8adad3a4fa2ec42d25c1fdb",
    }
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}


@with_defer
async def test_map_releases_to_prs_future(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag):
    prs, releases, matched_bys, _ = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2018, month=7, day=31, tzinfo=timezone.utc),
        datetime(year=2030, month=12, day=2, tzinfo=timezone.utc),
        [], [], JIRAFilter.empty(),
        release_match_setting_tag, None, None, None,
        1, (6366825,), mdb, pdb, rdb, None, truncate=False)
    assert len(prs) == 8
    assert releases is not None
    assert len(releases) == 12


@with_defer
async def test_map_prs_to_releases_smoke_metrics(branches, default_branches, dag, mdb, pdb, rdb,
                                                 with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    time_from = datetime(year=2015, month=10, day=13, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=1, day=24, tzinfo=timezone.utc)
    filters = [
        sql.and_(PullRequest.merged_at > time_from, PullRequest.created_at < time_to),
        PullRequest.repository_full_name.in_(["src-d/go-git"]),
        PullRequest.user_login.in_(["mcuadros", "vmarkovtsev"]),
    ]
    prs = await read_sql_query(select([PullRequest]).where(sql.and_(*filters)),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    prs["dead"] = False
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    settings = generate_repo_settings(prs)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to, settings,
        1, (6366825,), mdb, pdb, rdb, None)
    released_prs, _, _ = await map_prs_to_releases(
        prs, releases, matched_bys, branches, default_branches, time_to, dag, settings,
        1, (6366825,), mdb, pdb, None)
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


def check_branch_releases(releases: pd.DataFrame, n: int, time_from: datetime, time_to: datetime):
    assert len(releases) == n
    assert "mcuadros" in set(releases[Release.author.key])
    assert len(releases[Release.commit_id.key].unique()) == n
    assert releases[Release.id.key].all()
    assert all(len(n) == 40 for n in releases[Release.name.key])
    assert releases[Release.published_at.key].between(time_from, time_to).all()
    assert (releases[Release.repository_full_name.key] == "src-d/go-git").all()
    assert all(len(n) == 40 for n in releases[Release.sha.key])
    assert len(releases[Release.sha.key].unique()) == n
    assert (~releases[Release.tag.key].values.astype(bool)).all()
    assert releases[Release.url.key].str.startswith("http").all()


@pytest.mark.parametrize("branches_", ["{{default}}", "master", "m.*"])
@with_defer
async def test_load_releases_branches(branches, default_branches, mdb, pdb, rdb, cache, branches_,
                                      with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    time_from = datetime(year=2017, month=10, day=13, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=1, day=24, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        time_from,
        time_to,
        ReleaseSettings({"github.com/src-d/go-git": ReleaseMatchSetting(
            branches=branches_, tags="", match=ReleaseMatch.branch)}),
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}
    check_branch_releases(releases, 240, time_from, time_to)


@with_defer
async def test_load_releases_branches_empty(branches, default_branches, mdb, pdb, rdb, cache,
                                            with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    time_from = datetime(year=2017, month=10, day=13, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=1, day=24, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        time_from,
        time_to,
        ReleaseSettings({"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="unknown", tags="", match=ReleaseMatch.branch)}),
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    assert len(releases) == 0
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}


@pytest.mark.parametrize("time_from, n, pretag", [
    (datetime(year=2017, month=10, day=4, tzinfo=timezone.utc), 45, False),
    (datetime(year=2017, month=9, day=4, tzinfo=timezone.utc), 1, False),
    (datetime(year=2017, month=12, day=8, tzinfo=timezone.utc), 0, False),
    (datetime(year=2017, month=9, day=4, tzinfo=timezone.utc), 1, True),
])
@with_defer
async def test_load_releases_tag_or_branch_dates(
        branches, default_branches, release_match_setting_tag, mdb, pdb, rdb, cache,
        time_from, n, pretag, with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    time_to = datetime(year=2017, month=12, day=8, tzinfo=timezone.utc)

    if pretag:
        await release_loader.load_releases(
            ["src-d/go-git"],
            branches, default_branches,
            time_from,
            time_to,
            release_match_setting_tag,
            1,
            (6366825,),
            mdb,
            pdb,
            rdb,
            cache,
        )
        await wait_deferred()

    settings = ReleaseSettings({"github.com/src-d/go-git": ReleaseMatchSetting(
        branches="master", tags=".*", match=ReleaseMatch.tag_or_branch)})
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        time_from,
        time_to,
        settings,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    await wait_deferred()
    match_groups, _, repos_count = group_repos_by_release_match(
        ["src-d/go-git"], default_branches, settings)
    spans = (await release_loader.fetch_precomputed_release_match_spans(
        match_groups, 1, pdb))["src-d/go-git"]
    assert ReleaseMatch.tag in spans
    if n > 1:
        assert ReleaseMatch.branch in spans
        check_branch_releases(releases, n, time_from, time_to)
        assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}
    else:
        assert len(releases) == n
        if n > 0:
            if pretag:
                assert ReleaseMatch.branch not in spans
            assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
        else:
            assert ReleaseMatch.branch in spans
            assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}


@with_defer
async def test_load_releases_tag_or_branch_initial(branches, default_branches, mdb, pdb, rdb,
                                                   with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2015, month=10, day=22, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        time_from,
        time_to,
        ReleaseSettings({"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="master", tags="", match=ReleaseMatch.branch)}),
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}
    check_branch_releases(releases, 17, time_from, time_to)


@with_defer
async def test_map_releases_to_prs_branches(branches, default_branches, mdb, pdb, rdb):
    time_from = datetime(year=2015, month=4, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2015, month=5, day=1, tzinfo=timezone.utc)
    prs, releases, matched_bys, _ = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        time_from, time_to,
        [], [], JIRAFilter.empty(),
        ReleaseSettings({"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="master", tags="", match=ReleaseMatch.branch)}),
        None, None, None,
        1, (6366825,), mdb, pdb, rdb, None)
    assert prs.empty
    assert len(releases) == 1
    assert releases[Release.sha.key][0] == "5d7303c49ac984a9fec60523f2d5297682e16646"
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}


@with_defer
async def test_map_releases_to_prs_updated_min_max(
        branches, default_branches, release_match_setting_tag, mdb, pdb, rdb):
    prs, releases, matched_bys, _ = await map_releases_to_prs(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2018, month=7, day=31, tzinfo=timezone.utc),
        datetime(year=2030, month=12, day=2, tzinfo=timezone.utc),
        [], [], JIRAFilter.empty(),
        release_match_setting_tag,
        datetime(2018, 7, 20, tzinfo=timezone.utc), datetime(2019, 1, 1, tzinfo=timezone.utc),
        None, 1, (6366825,), mdb, pdb, rdb, None, truncate=False)
    assert len(prs) == 5
    assert releases is not None
    assert len(releases) == 12


@pytest.mark.parametrize("repos", [["src-d/gitbase"], []])
@with_defer
async def test_load_releases_empty(branches, default_branches, mdb, pdb, rdb, repos,
                                   with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    releases, matched_bys = await release_loader.load_releases(
        repos,
        branches, default_branches,
        datetime(year=2020, month=6, day=30, tzinfo=timezone.utc),
        datetime(year=2020, month=7, day=30, tzinfo=timezone.utc),
        ReleaseSettings({"github.com/src-d/gitbase": ReleaseMatchSetting(
            branches=".*", tags=".*", match=ReleaseMatch.branch)}),
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        index=Release.id.key)
    assert releases.empty
    if repos:
        assert matched_bys == {"src-d/gitbase": ReleaseMatch.branch}
    time_from = datetime(year=2017, month=3, day=4, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=12, day=8, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        time_from,
        time_to,
        ReleaseSettings({"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="master", tags="", match=ReleaseMatch.tag)}),
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert releases.empty
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        time_from,
        time_to,
        ReleaseSettings({"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="", tags=".*", match=ReleaseMatch.branch)}),
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert releases.empty
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}


@with_defer
async def test_load_releases_events_settings(branches, default_branches, mdb, pdb, rdb,
                                             with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    await rdb.execute(insert(ReleaseNotification).values(ReleaseNotification(
        account_id=1,
        repository_node_id="MDEwOlJlcG9zaXRvcnk0NDczOTA0NA==",
        commit_hash_prefix="8d20cc5",
        name="Pushed!",
        author_node_id="MDQ6VXNlcjI3OTM1NTE=",
        url="www",
        published_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    ).create_defaults().explode(with_primary_keys=True)))
    releases, _ = await release_loader.load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2019, month=1, day=30, tzinfo=timezone.utc),
        datetime(year=2020, month=7, day=30, tzinfo=timezone.utc),
        ReleaseSettings({"github.com/src-d/go-git": ReleaseMatchSetting(
            branches=".*", tags=".*", match=ReleaseMatch.tag)}),
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        index=Release.id.key)
    await wait_deferred()
    assert len(releases) == 7
    assert (releases[matched_by_column] == ReleaseMatch.tag).all()
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2019, month=1, day=30, tzinfo=timezone.utc),
        datetime(year=2020, month=7, day=30, tzinfo=timezone.utc),
        ReleaseSettings({"github.com/src-d/go-git": ReleaseMatchSetting(
            branches=".*", tags=".*", match=ReleaseMatch.event)}),
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        index=Release.id.key)
    await wait_deferred()
    assert matched_bys == {"src-d/go-git": ReleaseMatch.event}
    assert (releases[matched_by_column] == ReleaseMatch.event).all()
    assert len(releases) == 1
    assert releases.iloc[0].to_dict() == {
        "repository_full_name": "src-d/go-git",
        "repository_node_id": "MDEwOlJlcG9zaXRvcnk0NDczOTA0NA==",
        "author": "vmarkovtsev",
        "name": "Pushed!",
        "published_at": pd.Timestamp("2020-01-01 00:00:00", tzinfo=timezone.utc),
        "tag": None,
        "url": "www",
        "sha": "8d20cc5916edf7cfa6a9c5ed069f0640dc823c12",
        "commit_id": "MDY6Q29tbWl0NDQ3MzkwNDQ6OGQyMGNjNTkxNmVkZjdjZmE2YTljNWVkMDY5ZjA2NDBkYzgyM2MxMg==",  # noqa
        "matched_by": ReleaseMatch.event,
        "id": "MDY6Q29tbWl0NDQ3MzkwNDQ6OGQyMGNjNTkxNmVkZjdjZmE2YTljNWVkMDY5ZjA2NDBkYzgyM2MxMg==",
    }
    rows = await rdb.fetch_all(select([ReleaseNotification]))
    assert len(rows) == 1
    values = dict(rows[0])
    assert values[ReleaseNotification.updated_at.key] > values[ReleaseNotification.created_at.key]
    del values[ReleaseNotification.updated_at.key]
    del values[ReleaseNotification.created_at.key]
    if rdb.url.dialect == "sqlite":
        tzinfo = None
    else:
        tzinfo = timezone.utc
    assert values == {
        "account_id": 1,
        "repository_node_id": "MDEwOlJlcG9zaXRvcnk0NDczOTA0NA==",
        "commit_hash_prefix": "8d20cc5",
        "resolved_commit_hash": "8d20cc5916edf7cfa6a9c5ed069f0640dc823c12",
        "resolved_commit_node_id": "MDY6Q29tbWl0NDQ3MzkwNDQ6OGQyMGNjNTkxNmVkZjdjZmE2YTljNWVkMDY5ZjA2NDBkYzgyM2MxMg==",  # noqa
        "name": "Pushed!",
        "author_node_id": "MDQ6VXNlcjI3OTM1NTE=",
        "url": "www",
        "published_at": datetime(2020, 1, 1, tzinfo=tzinfo),
        "cloned": False,
    }


@with_defer
async def test_load_releases_events_unresolved(branches, default_branches, mdb, pdb, rdb,
                                               with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    await rdb.execute(insert(ReleaseNotification).values(ReleaseNotification(
        account_id=1,
        repository_node_id="MDEwOlJlcG9zaXRvcnk0NDczOTA0NA==",
        commit_hash_prefix="whatever",
        name="Pushed!",
        author_node_id="MDQ6VXNlcjI3OTM1NTE=",
        url="www",
        published_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    ).create_defaults().explode(with_primary_keys=True)))
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        datetime(year=2019, month=1, day=30, tzinfo=timezone.utc),
        datetime(year=2020, month=7, day=30, tzinfo=timezone.utc),
        ReleaseSettings({"github.com/src-d/go-git": ReleaseMatchSetting(
            branches=".*", tags=".*", match=ReleaseMatch.event)}),
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        index=Release.id.key)
    assert releases.empty
    assert matched_bys == {"src-d/go-git": ReleaseMatch.event}


@pytest.mark.parametrize("prune", [False, True])
@with_defer
async def test__fetch_repository_commits_smoke(mdb, pdb, prune):
    dags = await fetch_repository_commits(
        {"src-d/go-git": _empty_dag()},
        pd.DataFrame([
            ("d2a38b4a5965d529566566640519d03d2bd10f6c",
             "MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw==",
             525,
             "src-d/go-git"),
            ("31eae7b619d166c366bf5df4991f04ba8cebea0a",
             "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ==",
             611,
             "src-d/go-git")],
            columns=["1", "2", "3", "4"],
        ),
        ("1", "2", "3", "4"),
        prune, 1, (6366825,), mdb, pdb, None)
    assert isinstance(dags, dict)
    assert len(dags) == 1
    hashes, vertexes, edges = dags["src-d/go-git"]
    ground_truth = {
        "31eae7b619d166c366bf5df4991f04ba8cebea0a": ["b977a025ca21e3b5ca123d8093bd7917694f6da7",
                                                     "d2a38b4a5965d529566566640519d03d2bd10f6c"],
        "b977a025ca21e3b5ca123d8093bd7917694f6da7": ["35b585759cbf29f8ec428ef89da20705d59f99ec"],
        "d2a38b4a5965d529566566640519d03d2bd10f6c": ["35b585759cbf29f8ec428ef89da20705d59f99ec"],
        "35b585759cbf29f8ec428ef89da20705d59f99ec": ["c2bbf9fe8009b22d0f390f3c8c3f13937067590f"],
        "c2bbf9fe8009b22d0f390f3c8c3f13937067590f": ["fc9f0643b21cfe571046e27e0c4565f3a1ee96c8"],
        "fc9f0643b21cfe571046e27e0c4565f3a1ee96c8": ["c088fd6a7e1a38e9d5a9815265cb575bb08d08ff"],
        "c088fd6a7e1a38e9d5a9815265cb575bb08d08ff": ["5fddbeb678bd2c36c5e5c891ab8f2b143ced5baf"],
        "5fddbeb678bd2c36c5e5c891ab8f2b143ced5baf": ["5d7303c49ac984a9fec60523f2d5297682e16646"],
        "5d7303c49ac984a9fec60523f2d5297682e16646": [],
    }
    for k, v in ground_truth.items():
        vertex = np.where(hashes == k.encode())[0][0]
        assert hashes[edges[vertexes[vertex]:vertexes[vertex + 1]]].astype("U40").tolist() == v
    assert len(hashes) == 9
    await wait_deferred()
    dags2 = await fetch_repository_commits(
        dags,
        pd.DataFrame([
            ("d2a38b4a5965d529566566640519d03d2bd10f6c",
             "MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw==",
             525,
             "src-d/go-git"),
            ("31eae7b619d166c366bf5df4991f04ba8cebea0a",
             "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ==",
             611,
             "src-d/go-git")],
            columns=["1", "2", "3", "4"],
        ),
        ("1", "2", "3", "4"),
        prune, 1, (6366825,), Database("sqlite://"), pdb, None)
    assert pickle.dumps(dags2) == pickle.dumps(dags)
    with pytest.raises(Exception):
        await fetch_repository_commits(
            dags,
            pd.DataFrame([
                ("1353ccd6944ab41082099b79979ded3223db98ec",
                 "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ==",  # noqa
                 525,
                 "src-d/go-git"),
                ("31eae7b619d166c366bf5df4991f04ba8cebea0a",
                 "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ==",  # noqa
                 611,
                 "src-d/go-git")],
                columns=["1", "2", "3", "4"],
            ),
            ("1", "2", "3", "4"),
            prune, 1, (6366825,), Database("sqlite://"), pdb, None)


@pytest.mark.parametrize("prune", [False, True])
@with_defer
async def test__fetch_repository_commits_initial_commit(mdb, pdb, prune):
    dags = await fetch_repository_commits(
        {"src-d/go-git": _empty_dag()},
        pd.DataFrame([
            ("5d7303c49ac984a9fec60523f2d5297682e16646",
             "MDY6Q29tbWl0NDQ3MzkwNDQ6NWQ3MzAzYzQ5YWM5ODRhOWZlYzYwNTIzZjJkNTI5NzY4MmUxNjY0Ng==",
             525,
             "src-d/go-git")],
            columns=["1", "2", "3", "4"],
        ),
        ("1", "2", "3", "4"),
        prune, 1, (6366825,), mdb, pdb, None)
    hashes, vertexes, edges = dags["src-d/go-git"]
    assert hashes == np.array(["5d7303c49ac984a9fec60523f2d5297682e16646"], dtype="S40")
    assert (vertexes == np.array([0, 0], dtype=np.uint32)).all()
    assert (edges == np.array([], dtype=np.uint32)).all()


@with_defer
async def test__fetch_repository_commits_cache(mdb, pdb, cache):
    dags1 = await fetch_repository_commits(
        {"src-d/go-git": _empty_dag()},
        pd.DataFrame([
            ("d2a38b4a5965d529566566640519d03d2bd10f6c",
             "MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw==",
             525,
             "src-d/go-git"),
            ("31eae7b619d166c366bf5df4991f04ba8cebea0a",
             "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ==",
             611,
             "src-d/go-git")],
            columns=["1", "2", "3", "4"],
        ),
        ("1", "2", "3", "4"),
        False, 1, (6366825,), mdb, pdb, cache)
    await wait_deferred()
    dags2 = await fetch_repository_commits(
        {"src-d/go-git": _empty_dag()},
        pd.DataFrame([
            ("d2a38b4a5965d529566566640519d03d2bd10f6c",
             "MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw==",
             525,
             "src-d/go-git"),
            ("31eae7b619d166c366bf5df4991f04ba8cebea0a",
             "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ==",
             611,
             "src-d/go-git")],
            columns=["1", "2", "3", "4"],
        ),
        ("1", "2", "3", "4"),
        False, 1, (6366825,), None, None, cache)
    assert pickle.dumps(dags1) == pickle.dumps(dags2)
    fake_pdb = Database("sqlite://")

    class FakeMetrics:
        def get(self):
            return defaultdict(int)

    fake_pdb.metrics = {"hits": FakeMetrics(), "misses": FakeMetrics()}
    with pytest.raises(Exception):
        await fetch_repository_commits(
            {"src-d/go-git": _empty_dag()},
            pd.DataFrame([
                ("d2a38b4a5965d529566566640519d03d2bd10f6c",
                 "MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw==",  # noqa
                 525,
                 "src-d/go-git"),
                ("31eae7b619d166c366bf5df4991f04ba8cebea0a",
                 "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ==",  # noqa
                 611,
                 "src-d/go-git")],
                columns=["1", "2", "3", "4"],
            ),
            ("1", "2", "3", "4"),
            True, 1, (6366825,), None, fake_pdb, cache)


@with_defer
async def test__fetch_repository_commits_many(mdb, pdb):
    dags = await fetch_repository_commits(
        {"src-d/go-git": _empty_dag()},
        pd.DataFrame([
            ("d2a38b4a5965d529566566640519d03d2bd10f6c",
             "MDY6Q29tbWl0NDQ3MzkwNDQ6ZDJhMzhiNGE1OTY1ZDUyOTU2NjU2NjY0MDUxOWQwM2QyYmQxMGY2Yw==",
             525,
             "src-d/go-git"),
            ("31eae7b619d166c366bf5df4991f04ba8cebea0a",
             "MDY6Q29tbWl0NDQ3MzkwNDQ6MzFlYWU3YjYxOWQxNjZjMzY2YmY1ZGY0OTkxZjA0YmE4Y2ViZWEwYQ==",
             611,
             "src-d/go-git")] * 50,
            columns=["1", "2", "3", "4"],
        ),
        ("1", "2", "3", "4"),
        False, 1, (6366825,), mdb, pdb, None)
    assert len(dags["src-d/go-git"][0]) == 9


@with_defer
async def test__fetch_repository_commits_full(mdb, pdb, dag, cache, with_preloading):
    branch_miner = PreloadedBranchMiner if with_preloading else BranchMiner
    branches, _ = await branch_miner.extract_branches(dag, (6366825,), mdb, None)
    commit_ids = branches[Branch.commit_id.key].values
    commit_dates = await mdb.fetch_all(select([NodeCommit.id, NodeCommit.committed_date])
                                       .where(NodeCommit.id.in_(commit_ids)))
    commit_dates = {r[0]: r[1] for r in commit_dates}
    if mdb.url.dialect == "sqlite":
        commit_dates = {k: v.replace(tzinfo=timezone.utc) for k, v in commit_dates.items()}
    now = datetime.now(timezone.utc)
    branches[Branch.commit_date] = [commit_dates.get(commit_id, now) for commit_id in commit_ids]
    cols = (Branch.commit_sha.key, Branch.commit_id.key, Branch.commit_date,
            Branch.repository_full_name.key)
    commits = await fetch_repository_commits(
        dag, branches, cols, False, 1, (6366825,), mdb, pdb, cache)
    await wait_deferred()
    assert len(commits) == 1
    assert len(commits["src-d/go-git"][0]) == 1919
    branches = branches[branches[Branch.branch_name.key] == "master"]
    commits = await fetch_repository_commits(
        commits, branches, cols, False, 1, (6366825,), mdb, pdb, cache)
    await wait_deferred()
    assert len(commits) == 1
    assert len(commits["src-d/go-git"][0]) == 1919  # with force-pushed commits
    commits = await fetch_repository_commits(
        commits, branches, cols, True, 1, (6366825,), mdb, pdb, cache)
    await wait_deferred()
    assert len(commits) == 1
    assert len(commits["src-d/go-git"][0]) == 1538  # without force-pushed commits


@with_defer
async def test__find_dead_merged_prs_smoke(mdb):
    prs = await read_sql_query(
        select([PullRequest]).where(PullRequest.merged_at.isnot(None)),
        mdb, PullRequest, index=PullRequest.node_id.key)
    prs["dead"] = False
    prs.loc[prs[PullRequest.number.key].isin(force_push_dropped_go_git_pr_numbers), "dead"] = True
    dead_prs = await _find_dead_merged_prs(prs)
    assert len(dead_prs) == len(force_push_dropped_go_git_pr_numbers)
    assert dead_prs[Release.published_at.key].isnull().all()
    assert (dead_prs[matched_by_column] == ReleaseMatch.force_push_drop).all()
    dead_prs = await mdb.fetch_all(
        select([PullRequest.number]).where(PullRequest.node_id.in_(dead_prs.index)))
    assert {pr[0] for pr in dead_prs} == set(force_push_dropped_go_git_pr_numbers)


@with_defer
async def test__fetch_repository_first_commit_dates_pdb_cache(mdb, pdb, cache):
    fcd1 = await _fetch_repository_first_commit_dates(
        ["src-d/go-git"], 1, (6366825,), mdb, pdb, cache)
    await wait_deferred()
    fcd2 = await _fetch_repository_first_commit_dates(
        ["src-d/go-git"], 1, (6366825,), Database("sqlite://"), pdb, None)
    fcd3 = await _fetch_repository_first_commit_dates(
        ["src-d/go-git"], 1, (6366825,), Database("sqlite://"), Database("sqlite://"), cache)
    assert len(fcd1) == len(fcd2) == len(fcd3) == 1
    assert fcd1["src-d/go-git"] == fcd2["src-d/go-git"] == fcd3["src-d/go-git"]
    assert fcd1["src-d/go-git"].tzinfo == timezone.utc


def test_extract_subdag_smoke():
    hashes = np.array(["308a9f90707fb9d12cbcd28da1bc33da436386fe",
                       "33cafc14532228edca160e46af10341a8a632e3e",
                       "61a719e0ff7522cc0d129acb3b922c94a8a5dbca",
                       "a444ccadf5fddad6ad432c13a239c74636c7f94f"],
                      dtype="S40")
    vertexes = np.array([0, 1, 2, 3, 3], dtype=np.uint32)
    edges = np.array([3, 0, 0], dtype=np.uint32)
    heads = np.array(["61a719e0ff7522cc0d129acb3b922c94a8a5dbca"], dtype="S40")
    new_hashes, new_vertexes, new_edges = extract_subdag(hashes, vertexes, edges, heads)
    assert (new_hashes == np.array(["308a9f90707fb9d12cbcd28da1bc33da436386fe",
                                    "61a719e0ff7522cc0d129acb3b922c94a8a5dbca",
                                    "a444ccadf5fddad6ad432c13a239c74636c7f94f"],
                                   dtype="S40")).all()
    assert (new_vertexes == np.array([0, 1, 2, 2], dtype=np.uint32)).all()
    assert (new_edges == np.array([2, 0], dtype=np.uint32)).all()


def test_extract_subdag_empty():
    hashes = np.array([], dtype="S40")
    vertexes = np.array([0], dtype=np.uint32)
    edges = np.array([], dtype=np.uint32)
    heads = np.array(["61a719e0ff7522cc0d129acb3b922c94a8a5dbca"], dtype="S40")
    new_hashes, new_vertexes, new_edges = extract_subdag(hashes, vertexes, edges, heads)
    assert len(new_hashes) == 0
    assert (new_vertexes == vertexes).all()
    assert len(new_edges) == 0


def test_join_dags_smoke():
    hashes = np.array(["308a9f90707fb9d12cbcd28da1bc33da436386fe",
                       "33cafc14532228edca160e46af10341a8a632e3e",
                       "a444ccadf5fddad6ad432c13a239c74636c7f94f"],
                      dtype="S40")
    vertexes = np.array([0, 1, 2, 2], dtype=np.uint32)
    edges = np.array([2, 0], dtype=np.uint32)
    new_hashes, new_vertexes, new_edges = join_dags(
        hashes, vertexes, edges, [("61a719e0ff7522cc0d129acb3b922c94a8a5dbca",
                                   "308a9f90707fb9d12cbcd28da1bc33da436386fe",
                                   0),
                                  ("308a9f90707fb9d12cbcd28da1bc33da436386fe",
                                   "a444ccadf5fddad6ad432c13a239c74636c7f94f",
                                   0),
                                  ("8d27ef15cc9b334667d8adc9ce538222c5ac3607",
                                   "33cafc14532228edca160e46af10341a8a632e3e",
                                   1),
                                  ("8d27ef15cc9b334667d8adc9ce538222c5ac3607",
                                   "308a9f90707fb9d12cbcd28da1bc33da436386fe",
                                   0)])
    assert (new_hashes == np.array(["308a9f90707fb9d12cbcd28da1bc33da436386fe",
                                    "33cafc14532228edca160e46af10341a8a632e3e",
                                    "61a719e0ff7522cc0d129acb3b922c94a8a5dbca",
                                    "8d27ef15cc9b334667d8adc9ce538222c5ac3607",
                                    "a444ccadf5fddad6ad432c13a239c74636c7f94f"],
                                   dtype="S40")).all()
    assert (new_vertexes == np.array([0, 1, 2, 3, 5, 5], dtype=np.uint32)).all()
    assert (new_edges == np.array([4, 0, 0, 0, 1], dtype=np.uint32)).all()


def test_mark_dag_access_smoke():
    hashes = np.array(["308a9f90707fb9d12cbcd28da1bc33da436386fe",
                       "33cafc14532228edca160e46af10341a8a632e3e",
                       "61a719e0ff7522cc0d129acb3b922c94a8a5dbca",
                       "a444ccadf5fddad6ad432c13a239c74636c7f94f"],
                      dtype="S40")
    vertexes = np.array([0, 1, 2, 3, 3], dtype=np.uint32)
    edges = np.array([3, 0, 0], dtype=np.uint32)
    heads = np.array(["33cafc14532228edca160e46af10341a8a632e3e",
                      "61a719e0ff7522cc0d129acb3b922c94a8a5dbca"], dtype="S40")
    marks = mark_dag_access(hashes, vertexes, edges, heads)
    assert (marks == np.array([1, 0, 1, 1], dtype=np.int64)).all()


def test_mark_dag_access_empty():
    hashes = np.array([], dtype="S40")
    vertexes = np.array([0], dtype=np.uint32)
    edges = np.array([], dtype=np.uint32)
    heads = np.array(["33cafc14532228edca160e46af10341a8a632e3e",
                      "61a719e0ff7522cc0d129acb3b922c94a8a5dbca"], dtype="S40")
    marks = mark_dag_access(hashes, vertexes, edges, heads)
    assert len(marks) == 0


async def test_partition_dag(dag):
    hashes, vertexes, edges = dag["src-d/go-git"]
    p = partition_dag(hashes, vertexes, edges, [b"ad9456267524e08efcf4486cadfb6cef8d182677"])
    assert p.tolist() == [b"ad9456267524e08efcf4486cadfb6cef8d182677"]
    p = partition_dag(hashes, vertexes, edges, [b"7cd021554eb318165dd28988fe1675a5e5c32601"])
    assert p.tolist() == [b"7cd021554eb318165dd28988fe1675a5e5c32601",
                          b"ced875aec7bef9113e1c37b1b811a59e17dbd138"]


def test_partition_dag_empty():
    hashes = np.array([], dtype="S40")
    vertexes = np.array([0], dtype=np.uint32)
    edges = np.array([], dtype=np.uint32)
    p = partition_dag(hashes, vertexes, edges, ["ad9456267524e08efcf4486cadfb6cef8d182677"])
    assert len(p) == 0


async def test__fetch_commit_history_dag_stops(mdb, dag):
    hashes, vertexes, edges = dag["src-d/go-git"]
    subhashes, subvertexes, subedges = extract_subdag(
        hashes, vertexes, edges, [b"364866fc77fac656e103c1048dd7da4764c6d9d9"])
    assert len(subhashes) < len(hashes)
    _, newhashes, newvertexes, newedges = await _fetch_commit_history_dag(
        subhashes, subvertexes, subedges,
        ["f6305131a06bd94ef39e444b60f773db75b054f6"],
        ["MDY6Q29tbWl0NDQ3MzkwNDQ6MTdkYmQ4ODY2MTZmODJiZTJhNTljMGQwMmZkOTNkM2Q2OWYyMzkyYw=="],
        "src-d/go-git", (6366825,), mdb)
    assert (newhashes == hashes).all()
    assert (newvertexes == vertexes).all()
    assert (newedges == edges).all()


@with_defer
async def test_mark_dag_parents_smoke(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag, dag,
        with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    hashes, vertexes, edges = dag["src-d/go-git"]
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        time_from,
        time_to,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    release_hashes = releases[Release.sha.key].values.astype("S40")
    release_dates = releases[Release.published_at.key].values
    ownership = mark_dag_access(hashes, vertexes, edges, release_hashes)
    parents = mark_dag_parents(hashes, vertexes, edges, release_hashes, release_dates, ownership)
    assert (parents == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 53,
                        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 53, 47, 48, 49, 50, 51,
                        52, 53]).all()


@with_defer
async def test_mark_dag_parents_empty(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag, with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches, default_branches,
        time_from,
        time_to,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    release_hashes = releases[Release.sha.key].values
    release_dates = releases[Release.published_at.key].values
    hashes = np.array([], dtype="S40")
    vertexes = np.array([0], dtype=np.uint32)
    edges = np.array([], dtype=np.uint32)
    ownership = mark_dag_access(hashes, vertexes, edges, release_hashes)
    parents = mark_dag_parents(hashes, vertexes, edges, release_hashes, release_dates, ownership)
    assert len(parents) == 0


@with_defer
async def test_mine_releases_full_span(mdb, pdb, rdb, release_match_setting_tag, prefixer_promise):
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases, avatars, matched_bys = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to, JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    assert len(releases) == 53
    assert len(avatars) == 124
    assert matched_bys == {"github.com/src-d/go-git": ReleaseMatch.tag}
    for details, facts in releases:
        assert details[Release.name.key]
        assert details[Release.url.key]
        assert details[Release.repository_full_name.key] == "github.com/src-d/go-git"
        assert len(facts.commit_authors) > 0
        assert facts.age
        assert facts.publisher and facts.publisher.startswith("github.com/")
        assert facts.additions > 0
        assert facts.deletions > 0
        assert facts.commits_count > 0
        assert len(facts["prs_" + PullRequest.number.key]) or \
            facts.published <= pd.Timestamp("2017-02-01 09:51:10")
        assert time_from < facts.published.item().replace(tzinfo=timezone.utc) < time_to
        assert facts.matched_by == ReleaseMatch.tag


@with_defer
async def test_mine_releases_precomputed_smoke(
        mdb, pdb, rdb, release_match_setting_tag, release_match_setting_branch,
        prefixer_promise, branches, default_branches):
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to, JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    releases, avatars, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to, JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    assert len(releases) == 53
    assert len(avatars) == 124
    releases, avatars, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_to, time_to, JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    assert len(releases) == 0
    releases, avatars, _ = await mine_releases(
        ["src-d/go-git"], {}, branches, default_branches, time_from, time_to,
        JIRAFilter.empty(), release_match_setting_branch, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, None)
    assert len(releases) == 772
    assert len(avatars) == 131
    await wait_deferred()
    releases, avatars, _ = await mine_releases(
        ["src-d/go-git"], {}, branches, default_branches, time_from, time_to,
        JIRAFilter.empty(), release_match_setting_branch, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, None)
    assert len(releases) == 772
    assert len(avatars) == 131


@with_defer
async def test_mine_releases_precomputed_time_range(
        mdb, pdb, rdb, release_match_setting_tag, prefixer_promise):
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases, avatars, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to, JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()

    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases, avatars, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to, JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    for _, f in releases:
        assert time_from <= f.published.item().replace(tzinfo=timezone.utc) < time_to
        for col in released_prs_columns:
            assert len(f["prs_" + col.key]) > 0
        assert f.commits_count > 0
    assert len(releases) == 22
    assert len(avatars) == 93


@with_defer
async def test_mine_releases_precomputed_update(
        mdb, pdb, rdb, release_match_setting_tag, prefixer_promise):
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2018, month=11, day=1, tzinfo=timezone.utc)
    releases, avatars, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to, JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()

    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases, avatars, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to, JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    for _, f in releases:
        assert time_from <= f.published.item().replace(tzinfo=timezone.utc) < time_to
        assert len(getattr(f, "prs_" + PullRequest.number.key)) > 0
        assert f.commits_count > 0
    assert len(releases) == 22
    assert len(avatars) == 93


@with_defer
async def test_mine_releases_jira(
        mdb, pdb, rdb, release_match_setting_tag, prefixer_promise, cache):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=11, day=1, tzinfo=timezone.utc)
    releases, avatars, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to,
        JIRAFilter(1, ["10003", "10009"], LabelFilter({"bug", "onboarding", "performance"}, set()),
                   set(), set(), False),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    assert len(releases) == 8
    releases, avatars, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to,
        JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    assert len(releases) == 22
    releases, avatars, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to,
        JIRAFilter(1, ["10003", "10009"], LabelFilter({"bug", "onboarding", "performance"}, set()),
                   set(), set(), False),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    assert len(releases) == 8
    await wait_deferred()
    releases, avatars, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to,
        JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    assert len(releases) == 22
    releases, avatars, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to,
        JIRAFilter(1, ["10003", "10009"], LabelFilter.empty(), set(), set(), True),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    assert len(releases) == 15


@with_defer
async def test_mine_releases_cache(
        mdb, pdb, rdb, release_match_setting_tag, prefixer_promise, cache):
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2018, month=11, day=1, tzinfo=timezone.utc)
    releases1, _, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to, JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    releases2, _, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to, JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), None, None, None, cache)
    assert releases1 == releases2
    with pytest.raises(AssertionError):
        await mine_releases(
            ["src-d/go-git"], {}, None, {}, time_from, time_to, JIRAFilter.empty(),
            release_match_setting_tag, prefixer_promise, 1, (6366825,), None, None, None, cache,
            with_pr_titles=True)
    releases3, _, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to, JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache,
        with_pr_titles=True)
    await wait_deferred()
    releases4, _, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to, JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), None, None, None, cache,
        with_pr_titles=True)
    assert releases3 == releases4
    releases2, _, _ = await mine_releases(
        ["src-d/go-git"], {}, None, {}, time_from, time_to, JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, 1, (6366825,), None, None, None, cache)
    assert releases3 == releases2


@pytest.mark.parametrize("settings_index", [0, 1])
@with_defer
async def test_precomputed_releases_low_level(
        mdb, pdb, rdb, branches, default_branches,
        release_match_setting_tag, release_match_setting_branch, settings_index, with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    settings = [release_match_setting_branch, release_match_setting_tag][settings_index]
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases, _ = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to,
        settings, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    prels = await release_loader._fetch_precomputed_releases(
        {ReleaseMatch(settings_index): {["master", ".*"][settings_index]: ["src-d/go-git"]}},
        time_from, time_to, 1, pdb)
    prels = prels[releases.columns]
    assert_frame_equal(releases, prels)


@with_defer
async def test_precomputed_releases_ambiguous(
        mdb, pdb, rdb, branches, default_branches,
        release_match_setting_tag, release_match_setting_branch, with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases_tag, _ = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to,
        release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, None)
    releases_branch, _ = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to,
        release_match_setting_branch, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    prels = await release_loader._fetch_precomputed_releases(
        {ReleaseMatch.tag: {".*": ["src-d/go-git"]},
         ReleaseMatch.branch: {"master": ["src-d/go-git"]}},
        time_from, time_to, 1, pdb)
    prels = prels[releases_tag.columns]
    assert_frame_equal(releases_tag, prels)


async def test_precomputed_release_timespans(pdb, with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime.now(timezone.utc) - timedelta(days=100)
    mg1 = {ReleaseMatch.tag: {".*": ["src-d/go-git"]}}
    async with pdb.connection() as pdb_conn:
        async with pdb_conn.transaction():
            await release_loader._store_precomputed_release_match_spans(
                mg1, {"src-d/go-git": ReleaseMatch.tag}, time_from, time_to, 1, pdb_conn)
            mg2 = {ReleaseMatch.branch: {"master": ["src-d/go-git"]}}
            await release_loader._store_precomputed_release_match_spans(
                mg2, {"src-d/go-git": ReleaseMatch.branch}, time_from, time_to, 1, pdb_conn)
            await release_loader._store_precomputed_release_match_spans(
                mg1, {"src-d/go-git": ReleaseMatch.tag},
                time_from - timedelta(days=300), time_to + timedelta(days=200), 1, pdb_conn)
    spans = await release_loader.fetch_precomputed_release_match_spans(
        {**mg1, **mg2}, 1, pdb)
    assert len(spans) == 1
    assert len(spans["src-d/go-git"]) == 2
    assert spans["src-d/go-git"][ReleaseMatch.tag][0] == time_from - timedelta(days=300)
    assert spans["src-d/go-git"][ReleaseMatch.tag][1] <= datetime.now(timezone.utc)
    assert spans["src-d/go-git"][ReleaseMatch.branch] == (time_from, time_to)


@with_defer
async def test_precomputed_releases_append(
        mdb, pdb, rdb, branches, default_branches, release_match_setting_tag, with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases_tag1, _ = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches,
        time_from + timedelta(days=300), time_to - timedelta(days=900),
        release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    assert len(releases_tag1) == 39
    releases_tag2, _ = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to,
        release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, None)
    assert len(releases_tag2) == 53


@with_defer
async def test_precomputed_releases_tags_after_branches(
        mdb, pdb, rdb, branches, default_branches, release_match_setting_branch,
        release_match_setting_tag_or_branch, with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    # we don't have tags within our reach for this time interval
    time_from = datetime(year=2017, month=3, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=4, day=1, tzinfo=timezone.utc)
    releases_branch, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to,
        release_match_setting_branch, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    assert len(releases_branch) == 15
    assert matched_bys == {"src-d/go-git": ReleaseMatch.branch}

    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases_branch, _ = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to,
        release_match_setting_branch, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    assert len(releases_branch) == 772

    releases_tag, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to,
        release_match_setting_tag_or_branch, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    assert len(releases_tag) == 53
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}

    releases_tag, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to,
        release_match_setting_tag_or_branch, 1, (6366825,), mdb, pdb, rdb, None)
    assert len(releases_tag) == 53
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}


@pytest.mark.flaky(reruns=3)
@with_defer
async def test_mine_releases_by_name(
        mdb, pdb, rdb, branches, default_branches, release_match_setting_branch,
        release_match_setting_tag_or_branch, prefixer_promise, cache):
    # we don't have tags within our reach for this time interval
    time_from = datetime(year=2017, month=3, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=4, day=1, tzinfo=timezone.utc)
    releases, _, _ = await mine_releases(
        ["src-d/go-git"], {}, branches, default_branches, time_from, time_to, JIRAFilter.empty(),
        release_match_setting_branch, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    assert len(releases) == 15
    names = {"36c78b9d1b1eea682703fb1cbb0f4f3144354389", "v4.0.0"}
    releases, _ = await mine_releases_by_name(
        {"src-d/go-git": names},
        release_match_setting_tag_or_branch, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    assert len(releases) == 2
    releases_dict = {r[0][Release.name.key]: r for r in releases}
    assert releases_dict.keys() == names
    assert len(releases_dict["36c78b9d1b1eea682703fb1cbb0f4f3144354389"][1]
               ["prs_" + PullRequest.number.key]) == 1
    assert len(releases_dict["v4.0.0"][1]["prs_" + PullRequest.number.key]) == 62
    releases2, _ = await mine_releases_by_name(
        {"src-d/go-git": names},
        release_match_setting_tag_or_branch, prefixer_promise,
        1, (6366825,), None, None, None, cache)
    assert str(releases) == str(releases2)


"""
https://athenianco.atlassian.net/browse/DEV-250

async def test_map_prs_to_releases_miguel(mdb, pdb, rdb, release_match_setting_tag, cache,
                                          with_preloading):
    release_loader = PreloadedReleaseLoader if with_preloading else ReleaseLoader
    miguel_pr = await read_sql_query(select([PullRequest]).where(PullRequest.number == 907),
                                     mdb, PullRequest, index=PullRequest.node_id.key)
    # https://github.com/src-d/go-git/pull/907
    assert len(miguel_pr) == 1
    time_from = datetime(2018, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 5, 1, tzinfo=timezone.utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], None, None, time_from, time_to,
        release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, cache)
    released_prs, _ = await map_prs_to_releases(
        miguel_pr, releases, matched_bys, pd.DataFrame(), {}, time_to,
        release_match_setting_tag, 1, (6366825,), mdb, pdb, cache)
    assert len(released_prs) == 1
"""
