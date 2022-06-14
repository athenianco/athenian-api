import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import logging
from typing import List

import numpy as np
import pandas as pd

from athenian.api import ParallelDatabase
from athenian.api.async_utils import gather
from athenian.api.experiments.aggregates.models import PullRequestEvent
from athenian.api.experiments.aggregates.utils import get_accounts_and_repos

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aggregates.benchmark")


async def benchmark(sdb_conn_uri: str, adb_conn_uri: str, account: int, n: int):
    """Benchmark metrics aggregation query for the provided account."""
    sdb_conn = ParallelDatabase(sdb_conn_uri)
    adb_conn = ParallelDatabase(adb_conn_uri)
    await gather(sdb_conn.connect(), adb_conn.connect())

    to = datetime(2020, 7, 1).replace(tzinfo=timezone.utc)

    froms = [
        (datetime(2020, 6, 1).replace(tzinfo=timezone.utc), "1 month"),
        (datetime(2020, 5, 1).replace(tzinfo=timezone.utc), "2 months"),
        (datetime(2020, 4, 1).replace(tzinfo=timezone.utc), "3 months"),
        (datetime(2020, 1, 1).replace(tzinfo=timezone.utc), "6 months"),
        (datetime(2019, 10, 1).replace(tzinfo=timezone.utc), "9 months"),
        (datetime(2019, 7, 1).replace(tzinfo=timezone.utc), "12 months"),
        (datetime(2019, 1, 1).replace(tzinfo=timezone.utc), "18 months"),
        (datetime(2018, 7, 1).replace(tzinfo=timezone.utc), "24 months"),
        (datetime(2017, 7, 1).replace(tzinfo=timezone.utc), "36 months"),
        (datetime(2015, 7, 1).replace(tzinfo=timezone.utc), "60 months"),
    ]

    async with sdb_conn.connection() as sdb:
        account_repos = await get_accounts_and_repos(sdb, [account])

    repositories = account_repos[account]

    records = defaultdict(lambda: defaultdict(list))
    stats_records = []
    async with adb_conn.connection() as adb:
        for from_, label in froms:
            for granularity in ["all", "day"]:
                for i in range(n):
                    log.info(
                        "[%s][%d/%d] Running query with granularity %s for interval [%s, %s]",
                        label,
                        i,
                        n,
                        granularity,
                        from_,
                        to,
                    )
                    start = datetime.now()
                    try:
                        data = await _run_query(adb, repositories, from_, to, granularity)
                    except Exception as err:
                        log.error(err)
                        continue

                    elapsed = (datetime.now() - start).total_seconds()
                    log.info("Time elapsed: %fs", elapsed)

                    records[label][granularity].append(data)
                    stats_records.append(
                        {
                            "from": from_,
                            "to": to,
                            "label": label,
                            "granularity": granularity,
                            "elapsed": elapsed,
                        }
                    )

    stats = (
        pd.DataFrame(stats_records)
        .groupby(["from", "to", "label", "granularity"])["elapsed"]
        .agg([np.mean, np.std])
        .reset_index()
    )
    pr_numbers_per_label = pd.DataFrame(
        [
            {
                "label": label,
                "number_of_prs": dict(records[label]["all"][0][0])["cycle_count"],
            }
            for label in [f[1] for f in froms]
        ]
    )

    full_stats = stats.join(pr_numbers_per_label.set_index("label"), on="label").sort_values(
        by=["granularity", "number_of_prs"]
    )
    print(full_stats)


async def _run_query(
    adb: ParallelDatabase, repositories: List[str], from_: datetime, to: datetime, granularity: str
) -> pd.DataFrame:
    # TODO: these queries assume that the release setting do not change.
    # This has to be fixed by including in the filtering:
    #
    # WHERE
    #     (<repostory> = 'xxx' AND release_setting = 'abc') OR
    #     (<repostory> = 'yyy' AND release_setting = 'def') OR ...
    query_all = f"""
SELECT
    *,
    wip_time + review_time + merging_time + release_time AS cycle_time
FROM (
    SELECT
        sum_wip_time / GREATEST(sum_wip_count, 1) AS wip_time,
        sum_wip_count AS wip_count,
        sum_review_time / GREATEST(sum_review_count, 1) AS review_time,
        sum_review_count AS review_count,
        sum_merging_time / GREATEST(sum_merging_count, 1) AS merging_time,
        sum_merging_count AS merging_count,
        sum_release_time / GREATEST(sum_release_count, 1) AS release_time,
        sum_release_count AS release_count,
        sum_lead_time / GREATEST(sum_lead_count, 1) AS lead_time,
        sum_lead_count AS lead_count,
        sum_wait_first_review_time / GREATEST(sum_wait_first_review_count, 1) AS wait_first_review_time,
        sum_wait_first_review_count AS wait_first_review_count,
        sum_opened / GREATEST(sum_closed, 1) AS flow_ratio,
        sum_opened AS opened,
        sum_merged AS merged,
        sum_rejected AS rejected,
        sum_closed AS closed,
        sum_released AS released,
        sum_size_added + sum_size_removed AS size,
        sum_cycle_count AS cycle_count
    FROM (
        SELECT
            SUM("wip_time") AS sum_wip_time,
            SUM("wip_count") AS sum_wip_count,
            SUM("review_time") AS sum_review_time,
            SUM("review_count") AS sum_review_count,
            SUM("merging_time") AS sum_merging_time,
            SUM("merging_count") AS sum_merging_count,
            SUM("release_time") AS sum_release_time,
            SUM("release_count") AS sum_release_count,
            SUM("lead_time") AS sum_lead_time,
            SUM("lead_count") AS sum_lead_count,
            SUM("wait_first_review_time") AS sum_wait_first_review_time,
            SUM("wait_first_review_count") AS sum_wait_first_review_count,
            SUM("opened") AS sum_opened,
            SUM("merged") AS sum_merged,
            SUM("rejected") AS sum_rejected,
            SUM("closed") AS sum_closed,
            SUM("released") AS sum_released,
            SUM("size_added") AS sum_size_added,
            SUM("size_removed") AS sum_size_removed,
            COUNT(DISTINCT("repository_full_name", "number")) AS sum_cycle_count
        FROM {PullRequestEvent.__tablename__}
        WHERE
            timestamp >= '{from_.isoformat()}' AND timestamp <= '{to.isoformat()}' AND
            repository_full_name IN ('{"', '".join(repositories)}')
    ) AS q1
) AS q2
"""  # noqa

    query_day = f"""
SELECT
    *,
    wip_time + review_time + merging_time + release_time AS cycle_time
FROM (
    SELECT
        timestamp,
        sum_wip_time / GREATEST(sum_wip_count, 1) AS wip_time,
        sum_wip_count AS wip_count,
        sum_review_time / GREATEST(sum_review_count, 1) AS review_time,
        sum_review_count AS review_count,
        sum_merging_time / GREATEST(sum_merging_count, 1) AS merging_time,
        sum_merging_count AS merging_count,
        sum_release_time / GREATEST(sum_release_count, 1) AS release_time,
        sum_release_count AS release_count,
        sum_lead_time / GREATEST(sum_lead_count, 1) AS lead_time,
        sum_lead_count AS lead_count,
        sum_wait_first_review_time / GREATEST(sum_wait_first_review_count, 1) AS wait_first_review_time,
        sum_wait_first_review_count AS wait_first_review_count,
        sum_opened / GREATEST(sum_closed, 1) AS flow_ratio,
        sum_opened AS opened,
        sum_merged AS merged,
        sum_rejected AS rejected,
        sum_closed AS closed,
        sum_released AS released,
        sum_size_added + sum_size_removed AS size,
        sum_cycle_count AS cycle_count
    FROM (
        SELECT
            DATE_TRUNC('day', "timestamp") AS timestamp,
            SUM("wip_time") AS sum_wip_time,
            SUM("wip_count") AS sum_wip_count,
            SUM("review_time") AS sum_review_time,
            SUM("review_count") AS sum_review_count,
            SUM("merging_time") AS sum_merging_time,
            SUM("merging_count") AS sum_merging_count,
            SUM("release_time") AS sum_release_time,
            SUM("release_count") AS sum_release_count,
            SUM("lead_time") AS sum_lead_time,
            SUM("lead_count") AS sum_lead_count,
            SUM("wait_first_review_time") AS sum_wait_first_review_time,
            SUM("wait_first_review_count") AS sum_wait_first_review_count,
            SUM("opened") AS sum_opened,
            SUM("merged") AS sum_merged,
            SUM("rejected") AS sum_rejected,
            SUM("closed") AS sum_closed,
            SUM("released") AS sum_released,
            SUM("size_added") AS sum_size_added,
            SUM("size_removed") AS sum_size_removed,
            COUNT(DISTINCT(repository_full_name, number)) AS sum_cycle_count
        FROM {PullRequestEvent.__tablename__}
        WHERE
            timestamp >= '{from_.isoformat()}' AND timestamp <= '{to.isoformat()}' AND
            repository_full_name IN ('{"', '".join(repositories)}')
        GROUP BY
            DATE_TRUNC('day', "timestamp")
    ) AS q1
) AS q2
ORDER BY timestamp
    """  # noqa

    query = query_all if granularity == "all" else query_day
    return await adb.fetch_all(query)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-db",
        required=True,
        dest="sdb_conn",
        help="State DB endpoint, e.g. postgresql://0.0.0.0:5432/state",
    )
    parser.add_argument(
        "--aggregates-db",
        required=True,
        dest="adb_conn",
        help="Aggregates DB endpoint, e.g. postgresql://0.0.0.0:5432/aggregates",
    )
    parser.add_argument(
        "--account",
        required=True,
        type=int,
        dest="account",
        help="Account to benchmark the metrics aggregation for",
    )
    parser.add_argument(
        "--debug",
        required=False,
        action="store_true",
        dest="debug",
        help="Whether to run in debug mode",
    )
    parser.add_argument(
        "--n",
        required=False,
        type=int,
        dest="n",
        default=10,
        help="Number of runs for each interval (default: 10)",
    )

    return parser.parse_args()


def main():
    """Run benchmark for metrics aggregation query."""
    args = _parse_args()
    asyncio.run(benchmark(args.sdb_conn, args.adb_conn, args.account, args.n), debug=args.debug)


if __name__ == "__main__":
    exit(main())
