from datetime import date, timezone

import pandas as pd
from sqlalchemy import select

from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.controllers.miners.github.release import map_prs_to_releases, map_releases_to_prs
from athenian.api.models.metadata.github import PullRequest, Release


async def test_map_prs_to_releases(mdb, cache):
    prs = await read_sql_query(select([PullRequest]).where(PullRequest.number == 1126),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    for _ in range(2):
        releases = await map_prs_to_releases(prs, date.today(), mdb, cache)
        assert len(cache.mem) > 0
        assert len(releases) == 1
        assert releases.iloc[0][Release.published_at.key] == \
            pd.Timestamp("2019-07-31 13:41:28", tzinfo=timezone.utc)
        assert releases.iloc[0][Release.author.key] == "mcuadros"
        assert releases.iloc[0][Release.url.key] == "https://github.com/src-d/go-git/releases/tag/v4.13.0"  # noqa


async def test_map_prs_to_releases_empty(mdb, cache):
    prs = await read_sql_query(select([PullRequest]).where(PullRequest.number == 1231),
                               mdb, PullRequest, index=PullRequest.node_id.key)
    for _ in range(2):
        releases = await map_prs_to_releases(prs, date.today(), mdb, cache)
        assert len(cache.mem) == 0
        assert releases.empty
    prs = prs.iloc[:0]
    releases = await map_prs_to_releases(prs, date.today(), mdb, cache)
    assert len(cache.mem) == 0
    assert releases.empty


async def test_map_releases_to_prs(mdb, cache):
    for _ in range(2):
        prs, rels = await map_releases_to_prs(
            ["src-d/go-git"],
            date(year=2019, month=7, day=31), date(year=2019, month=12, day=1),
            mdb, cache)
        assert len(prs) == len(rels) == 6
        assert list(rels[Release.published_at.key].unique()) == \
            [pd.Timestamp("2019-07-31 13:41:28", tzinfo=timezone.utc)]
        assert list(rels[Release.author.key].unique()) == ["mcuadros"]
        assert list(rels[Release.url.key].unique()) == ["https://github.com/src-d/go-git/releases/tag/v4.13.0"]  # noqa
        assert len(cache.mem) > 0
        for pid in rels.index:
            assert not prs.loc[pid].empty
        assert (prs[PullRequest.merged_at.key] < pd.Timestamp(
            "2019-07-31 00:00:00", tzinfo=timezone.utc)).all()


async def test_map_releases_to_prs_empty(mdb, cache):
    prs, rels = await map_releases_to_prs(
        ["src-d/go-git"],
        date(year=2019, month=11, day=1), date(year=2019, month=12, day=1),
        mdb, cache)
    assert prs is None
    assert rels is None
    assert len(cache.mem) == 0
    prs, rels = await map_releases_to_prs(
        ["src-d/go-git"],
        date(year=2019, month=7, day=1), date(year=2019, month=12, day=1),
        mdb, cache)
    assert prs.empty
    assert rels.empty
    assert len(cache.mem) > 0
