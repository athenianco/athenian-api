from datetime import datetime, timedelta, timezone
from typing import Dict

import pandas as pd
import pytest
from sqlalchemy import select, sql

from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.controllers.miners.github.release import load_releases, map_prs_to_releases, \
    map_releases_to_prs
from athenian.api.controllers.settings import Match, ReleaseMatchSetting
from athenian.api.models.metadata.github import PullRequest, Release


def generate_repo_settings(prs: pd.DataFrame) -> Dict[str, ReleaseMatchSetting]:
    return {
        "github.com/" + r: ReleaseMatchSetting(branches="", tags=".*", match=Match.tag)
        for r in prs[PullRequest.repository_full_name.key]
    }


async def test_map_prs_to_releases_cache(mdb, cache):
    prs = await read_sql_query(select([PullRequest]).where(PullRequest.number == 1126),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    for i in range(2):
        releases = await map_prs_to_releases(prs, time_from, time_to,
                                             generate_repo_settings(prs), mdb, cache)
        assert len(cache.mem) > 0
        assert len(releases) == 1, str(i)
        assert releases.iloc[0][Release.published_at.key] == \
            pd.Timestamp("2019-06-18 22:57:34+0000", tzinfo=timezone.utc)
        assert releases.iloc[0][Release.author.key] == "mcuadros"
        assert releases.iloc[0][Release.url.key] == "https://github.com/src-d/go-git/releases/tag/v4.12.0"  # noqa
    releases = await map_prs_to_releases(prs, time_to, time_to,
                                         generate_repo_settings(prs), mdb, None)
    # the PR was merged and released in the past, we must detect that
    assert len(releases) == 1
    assert releases.iloc[0][Release.url.key] == "https://github.com/src-d/go-git/releases/tag/v4.12.0"  # noqa


async def test_map_prs_to_releases_empty(mdb, cache):
    prs = await read_sql_query(select([PullRequest]).where(PullRequest.number == 1231),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    for _ in range(2):
        releases = await map_prs_to_releases(prs, time_from, time_to,
                                             generate_repo_settings(prs), mdb, cache)
        assert len(cache.mem) == 0
        assert releases.empty
    prs = prs.iloc[:0]
    releases = await map_prs_to_releases(prs, time_from, time_to,
                                         generate_repo_settings(prs), mdb, cache)
    assert len(cache.mem) == 0
    assert releases.empty


async def test_map_releases_to_prs(mdb, cache, release_match_setting_tag):
    for _ in range(2):
        prs = await map_releases_to_prs(
            ["src-d/go-git"],
            datetime(year=2019, month=7, day=31, tzinfo=timezone.utc),
            datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
            release_match_setting_tag, mdb, cache)
        assert len(prs) == 7
        assert (prs[PullRequest.merged_at.key] < pd.Timestamp(
            "2019-07-31 00:00:00", tzinfo=timezone.utc)).all()
        assert (prs[PullRequest.merged_at.key] > pd.Timestamp(
            "2019-06-19 00:00:00", tzinfo=timezone.utc)).all()


async def test_map_releases_to_prs_empty(mdb, cache, release_match_setting_tag):
    prs = await map_releases_to_prs(
        ["src-d/go-git"],
        datetime(year=2019, month=11, day=1, tzinfo=timezone.utc),
        datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
        release_match_setting_tag, mdb, cache)
    assert prs.empty
    assert len(cache.mem) == 0
    prs = await map_releases_to_prs(
        ["src-d/go-git"],
        datetime(year=2019, month=7, day=1, tzinfo=timezone.utc),
        datetime(year=2019, month=12, day=2, tzinfo=timezone.utc),
        release_match_setting_tag, mdb, cache)
    assert prs.empty
    assert len(cache.mem) > 0


async def test_map_prs_to_releases_smoke_metrics(mdb):
    time_from = datetime(year=2015, month=10, day=13, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=1, day=24, tzinfo=timezone.utc)
    filters = [
        sql.or_(sql.and_(PullRequest.updated_at >= time_from,
                         PullRequest.updated_at < time_to),
                sql.and_(sql.or_(PullRequest.closed_at.is_(None),
                                 PullRequest.closed_at > time_from),
                         PullRequest.created_at < time_to)),
        PullRequest.repository_full_name.in_(["src-d/go-git"]),
        PullRequest.user_login.in_(["mcuadros", "vmarkovtsev"]),
    ]
    prs = await read_sql_query(select([PullRequest]).where(sql.and_(*filters)),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=5 * 365)
    releases = await map_prs_to_releases(prs, time_from, time_to,
                                         generate_repo_settings(prs), mdb, None)
    assert len(releases[Release.url.key].unique()) > 1


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


@pytest.mark.parametrize("branches", ["{{default}}", "master", "m.*"])
async def test_load_releases_branches(mdb, cache, branches):
    date_from = datetime(year=2017, month=10, day=13, tzinfo=timezone.utc)
    date_to = datetime(year=2020, month=1, day=24, tzinfo=timezone.utc)
    releases = await load_releases(
        ["src-d/go-git"],
        date_from,
        date_to,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches=branches, tags="", match=Match.branch)},
        mdb,
        cache,
    )
    check_branch_releases(releases, 240, date_from, date_to)


async def test_load_releases_branches_empty(mdb, cache):
    date_from = datetime(year=2017, month=10, day=13, tzinfo=timezone.utc)
    date_to = datetime(year=2020, month=1, day=24, tzinfo=timezone.utc)
    releases = await load_releases(
        ["src-d/go-git"],
        date_from,
        date_to,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="unknown", tags="", match=Match.branch)},
        mdb,
        cache,
    )
    assert len(releases) == 0


@pytest.mark.parametrize("date_from, n", [
    (datetime(year=2017, month=10, day=4, tzinfo=timezone.utc), 45),
    (datetime(year=2017, month=9, day=4, tzinfo=timezone.utc), 1),
    (datetime(year=2017, month=12, day=8, tzinfo=timezone.utc), 0),
])
async def test_load_releases_tag_or_branch_dates(mdb, cache, date_from, n):
    date_to = datetime(year=2017, month=12, day=8, tzinfo=timezone.utc)
    releases = await load_releases(
        ["src-d/go-git"],
        date_from,
        date_to,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="master", tags=".*", match=Match.tag_or_branch)},
        mdb,
        cache,
    )
    if n > 1:
        check_branch_releases(releases, n, date_from, date_to)
    else:
        assert len(releases) == n


async def test_load_releases_tag_or_branch_initial(mdb):
    date_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2015, month=10, day=22, tzinfo=timezone.utc)
    releases = await load_releases(
        ["src-d/go-git"],
        date_from,
        date_to,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="master", tags="", match=Match.branch)},
        mdb,
        None,
    )
    check_branch_releases(releases, 17, date_from, date_to)


async def test_map_releases_to_prs_branches(mdb):
    date_from = datetime(year=2015, month=4, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2015, month=5, day=1, tzinfo=timezone.utc)
    prs = await map_releases_to_prs(
        ["src-d/go-git"],
        date_from,
        date_to,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="master", tags="", match=Match.branch)},
        mdb,
        None)
    assert prs.empty


@pytest.mark.parametrize("repos", [["src-d/gitbase"], []])
async def test_load_releases_empty(mdb, repos):
    releases = await load_releases(
        repos,
        datetime(year=2020, month=6, day=30, tzinfo=timezone.utc),
        datetime(year=2020, month=7, day=30, tzinfo=timezone.utc),
        {"github.com/src-d/gitbase": ReleaseMatchSetting(
            branches=".*", tags=".*", match=Match.branch)},
        mdb,
        None,
        index=Release.id.key)
    assert releases.empty
    date_from = datetime(year=2017, month=3, day=4, tzinfo=timezone.utc)
    date_to = datetime(year=2017, month=12, day=8, tzinfo=timezone.utc)
    releases = await load_releases(
        ["src-d/go-git"],
        date_from,
        date_to,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="master", tags="", match=Match.tag)},
        mdb,
        None,
    )
    assert releases.empty
    releases = await load_releases(
        ["src-d/go-git"],
        date_from,
        date_to,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="", tags=".*", match=Match.branch)},
        mdb,
        None,
    )
    assert releases.empty
