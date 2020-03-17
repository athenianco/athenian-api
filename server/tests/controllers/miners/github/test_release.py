from datetime import date, timezone

import pandas as pd
from sqlalchemy import select

from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.controllers.miners.github.release import column_released_at, \
    column_released_by, map_prs_to_releases, map_releases_to_prs
from athenian.api.models.metadata.github import PullRequest


async def test_map_prs_to_releases(mdb, cache):
    prs = await read_sql_query(select([PullRequest]).where(PullRequest.number == 1126),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    for _ in range(2):
        releases = await map_prs_to_releases(prs, date.today(), mdb, cache)
        assert len(cache.mem) > 0
        assert len(releases) == 1
        assert releases.iloc[0][column_released_at] == pd.Timestamp("2019-07-31 13:41:28")
        assert releases.iloc[0][column_released_by] == "mcuadros"


async def test_map_prs_to_releases_empty(mdb, cache):
    prs = await read_sql_query(select([PullRequest]).where(PullRequest.number == 1231),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    for _ in range(2):
        releases = await map_prs_to_releases(prs, date.today(), mdb, cache)
        assert len(cache.mem) == 0
        assert releases.empty


async def test_map_releases_to_prs(mdb, cache):
    for _ in range(2):
        prs, rels = await map_releases_to_prs(
            ["src-d/go-git"],
            pd.Timestamp("2019-07-31 00:00:00"), pd.Timestamp("2019-12-01 00:00:00"),
            mdb, cache)
        assert len(prs) == len(rels) == 6
        assert list(rels[column_released_at].unique()) == [pd.Timestamp("2019-07-31 13:41:28")]
        assert list(rels[column_released_by].unique()) == ["mcuadros"]
        assert len(cache.mem) > 0
        for pid in rels.index:
            assert not prs.loc[pid].empty
        assert (prs[PullRequest.merged_at.key] < pd.Timestamp(
            "2019-07-31 00:00:00", tzinfo=timezone.utc)).all()


async def test_map_releases_to_prs_empty(mdb, cache):
    prs, rels = await map_releases_to_prs(
        ["src-d/go-git"],
        pd.Timestamp("2019-11-01 00:00:00"), pd.Timestamp("2019-12-01 00:00:00"),
        mdb, cache)
    assert prs is None
    assert rels is None
    assert len(cache.mem) == 0
    prs, rels = await map_releases_to_prs(
        ["src-d/go-git"],
        pd.Timestamp("2019-07-01 00:00:00"), pd.Timestamp("2019-12-01 00:00:00"),
        mdb, cache)
    assert prs.empty
    assert rels.empty
    assert len(cache.mem) > 0
