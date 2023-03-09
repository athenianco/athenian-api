from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest

from athenian.api.db import Database
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.features.github.check_run_filter import filter_check_runs
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.types import CodeCheckRunListItem, CodeCheckRunListStats
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_MD_ACCOUNT_ID
from tests.testutils.factory.wizards import insert_repo
from tests.testutils.time import dt


def td_list(items: list[int | None]) -> list[timedelta | None]:
    return [(x if x is None else timedelta(seconds=x)) for x in items]


async def test_filter_check_runs_monthly_quantiles(mdb, logical_settings):
    timeline, items = await _filter(
        time_from=dt(2015, 1, 1),
        time_to=dt(2020, 1, 1),
        quantiles=[0, 0.95],
        logical_settings=logical_settings,
        mdb=mdb,
    )
    assert len(timeline) == len(set(timeline)) == 61
    assert len(items) == 9
    assert items[0] == CodeCheckRunListItem(
        title="DCO",
        repository="src-d/go-git",
        last_execution_time=pd.Timestamp("2019-12-30T09:41:10+00").to_pydatetime(),
        last_execution_url="https://github.com/src-d/go-git/runs/367607194",
        size_groups=[1, 2, 3, 4, 5, 6],
        total_stats=CodeCheckRunListStats(
            count=383,
            successes=361,
            skips=0,
            critical=False,
            flaky_count=0,
            mean_execution_time=timedelta(seconds=0),
            stddev_execution_time=timedelta(seconds=1),
            median_execution_time=timedelta(seconds=1),
            count_timeline=[
                *([0] * 39),
                18,
                18,
                17,
                26,
                49,
                38,
                42,
                11,
                5,
                5,
                15,
                11,
                59,
                17,
                9,
                18,
                12,
                0,
                3,
                6,
                4,
            ],
            successes_timeline=[
                *([0] * 39),
                17,
                17,
                16,
                24,
                48,
                37,
                34,
                10,
                5,
                5,
                15,
                11,
                57,
                17,
                9,
                17,
                10,
                0,
                3,
                5,
                4,
            ],
            mean_execution_time_timeline=td_list(
                [
                    *([None] * 44),
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    None,
                    1,
                    1,
                    1,
                ],
            ),
            median_execution_time_timeline=td_list(
                [
                    *([None] * 44),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    1.0,
                    1.0,
                    None,
                    2.0,
                    1.0,
                    2.0,
                ],
            ),
        ),
        prs_stats=CodeCheckRunListStats(
            count=319,
            successes=301,
            skips=0,
            critical=False,
            flaky_count=0,
            mean_execution_time=timedelta(seconds=0),
            stddev_execution_time=timedelta(seconds=1),
            median_execution_time=timedelta(seconds=1),
            count_timeline=[
                *([0] * 39),
                10,
                18,
                13,
                15,
                39,
                26,
                37,
                10,
                5,
                5,
                14,
                11,
                50,
                15,
                9,
                17,
                12,
                0,
                3,
                6,
                4,
            ],
            successes_timeline=[
                *([0] * 39),
                9,
                17,
                12,
                15,
                38,
                25,
                30,
                9,
                5,
                5,
                14,
                11,
                48,
                15,
                9,
                17,
                10,
                0,
                3,
                5,
                4,
            ],
            mean_execution_time_timeline=td_list(
                [*([None] * 44), 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, None, 1, 1, 1],
            ),
            median_execution_time_timeline=td_list(
                [
                    *([None] * 44),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    1.0,
                    1.0,
                    None,
                    2.0,
                    1.0,
                    2.0,
                ],
            ),
        ),
    )


async def test_filter_check_runs_daily(mdb, logical_settings):
    timeline, items = await _filter(
        time_from=dt(2018, 2, 1),
        time_to=dt(2018, 2, 12),
        logical_settings=logical_settings,
        mdb=mdb,
    )
    assert len(timeline) == len(set(timeline)) == 12
    assert len(items) == 7


async def test_filter_check_runs_empty(mdb, logical_settings):
    timeline, items = await _filter(
        time_from=dt(2018, 2, 1),
        time_to=dt(2018, 2, 12),
        pushers=["xxx"],
        logical_settings=logical_settings,
        mdb=mdb,
    )
    assert len(timeline) == len(set(timeline)) == 12
    assert len(items) == 0


@with_defer
async def test_filter_check_runs_cache(mdb, cache, logical_settings):
    timeline1, items1 = await _filter(
        time_from=dt(2015, 1, 1),
        time_to=dt(2020, 1, 1),
        quantiles=[0, 0.95],
        logical_settings=logical_settings,
        mdb=mdb,
        cache=cache,
    )
    await wait_deferred()
    timeline2, items2 = await _filter(
        time_from=dt(2015, 1, 1),
        time_to=dt(2020, 1, 1),
        quantiles=[0, 0.95],
        logical_settings=logical_settings,
        mdb=None,
        cache=cache,
    )
    assert timeline1 == timeline2
    assert items1 == items2
    timeline2, items2 = await _filter(
        time_from=dt(2015, 1, 1),
        time_to=dt(2020, 1, 1),
        quantiles=[0, 0.05],
        logical_settings=logical_settings,
        mdb=None,
        cache=cache,
    )
    assert items1 != items2
    with pytest.raises(AttributeError):
        await _filter(
            time_from=dt(2015, 1, 1),
            time_to=dt(2020, 1, 2),
            quantiles=[0, 0.95],
            logical_settings=logical_settings,
            mdb=None,
            cache=cache,
        )


async def test_filter_check_runs_logical_repos(mdb, logical_settings):
    timeline, items = await _filter(
        time_from=dt(2016, 2, 1),
        time_to=dt(2018, 2, 12),
        repositories=["src-d/go-git/alpha"],
        logical_settings=logical_settings,
        mdb=mdb,
    )
    assert len(timeline) == len(set(timeline)) == 26
    assert len(items) == 8


async def test_only_queued_runs(mdb_rw: Database, sdb: Database, logical_settings) -> None:
    models = [
        md_factory.CheckRunFactory(
            repository_full_name="o/r",
            repository_node_id=99,
            started_at=dt(2022, 1, 2),
            committed_date=dt(2022, 1, 2),
            check_suite_status="QUEUED",
        ),
    ]
    async with DBCleaner(mdb_rw) as mdb_cleaner:
        repo = md_factory.RepositoryFactory(node_id=99, full_name="o/r")
        await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
        mdb_cleaner.add_models(*models)
        await models_insert(mdb_rw, *models)

        _, items = await _filter(
            time_from=dt(2022, 1, 1),
            time_to=dt(2022, 1, 3),
            repositories=["o/r"],
            logical_settings=logical_settings,
            mdb=mdb_rw,
        )
    assert items == []


async def test_unicode_run_name(mdb_rw: Database, sdb: Database, logical_settings) -> None:
    models = [
        md_factory.CheckRunFactory(
            repository_full_name="o/r",
            repository_node_id=99,
            started_at=dt(2022, 1, 2),
            committed_date=dt(2022, 1, 2),
            check_suite_status="COMPLETED",
            name="check run 🧪",
        ),
    ]
    async with DBCleaner(mdb_rw) as mdb_cleaner:
        repo = md_factory.RepositoryFactory(node_id=99, full_name="o/r")
        await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
        mdb_cleaner.add_models(*models)
        await models_insert(mdb_rw, *models)

        timeline, items = await _filter(
            time_from=dt(2022, 1, 1),
            time_to=dt(2022, 1, 3),
            repositories=["o/r"],
            logical_settings=logical_settings,
            mdb=mdb_rw,
        )
    assert len(items) == 1
    assert items[0].title == "check run 🧪"


async def _filter(**kwargs: Any) -> tuple[list[date], list[CodeCheckRunListItem]]:
    kwargs.setdefault("repositories", ["src-d/go-git"])
    kwargs.setdefault("pushers", [])
    kwargs.setdefault("labels", LabelFilter.empty())
    kwargs.setdefault("jira", JIRAFilter.empty())
    kwargs.setdefault("quantiles", [0, 1])
    kwargs.setdefault("meta_ids", (DEFAULT_MD_ACCOUNT_ID,))
    kwargs.setdefault("cache", None)
    return await filter_check_runs(**kwargs)
