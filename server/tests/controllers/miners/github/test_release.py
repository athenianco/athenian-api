from datetime import datetime, timezone
from typing import Dict

import pandas as pd
from sqlalchemy import select, sql

from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.controllers.miners.github.release import map_prs_to_releases, \
    map_releases_to_prs
from athenian.api.controllers.settings import Match, ReleaseMatchSetting
from athenian.api.models.metadata.github import PullRequest, Release


def generate_repo_settings(prs: pd.DataFrame) -> Dict[str, ReleaseMatchSetting]:
    return {
        "github.com/" + r: ReleaseMatchSetting(branches="", tags=".*", match=Match.tag)
        for r in prs[PullRequest.repository_full_name.key]
    }


async def test_map_prs_to_releases(mdb, cache):
    prs = await read_sql_query(select([PullRequest]).where(PullRequest.number == 1126),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    for _ in range(2):
        releases = await map_prs_to_releases(prs, datetime.now(tz=timezone.utc),
                                             generate_repo_settings(prs), mdb, cache)
        assert len(cache.mem) > 0
        assert len(releases) == 1
        assert releases.iloc[0][Release.published_at.key] == \
            pd.Timestamp("2019-06-18 22:57:34+0000", tzinfo=timezone.utc)
        assert releases.iloc[0][Release.author.key] == "mcuadros"
        assert releases.iloc[0][Release.url.key] == "https://github.com/src-d/go-git/releases/tag/v4.12.0"  # noqa


async def test_map_prs_to_releases_empty(mdb, cache):
    prs = await read_sql_query(select([PullRequest]).where(PullRequest.number == 1231),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    for _ in range(2):
        releases = await map_prs_to_releases(prs, datetime.now(tz=timezone.utc),
                                             generate_repo_settings(prs), mdb, cache)
        assert len(cache.mem) == 0
        assert releases.empty
    prs = prs.iloc[:0]
    releases = await map_prs_to_releases(prs, datetime.now(tz=timezone.utc),
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
    releases = await map_prs_to_releases(prs, datetime.now(tz=timezone.utc),
                                         generate_repo_settings(prs), mdb, None)
    assert len(releases[Release.url.key].unique()) > 1
