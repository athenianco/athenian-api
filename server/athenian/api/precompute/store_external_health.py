import argparse
import asyncio
from datetime import datetime, timezone
import logging
from typing import Any

import aiohttp
from sqlalchemy import insert, select

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.db import Database
from athenian.api.models.persistentdata.models import HealthMetric
from athenian.api.models.state.models import AccountGitHubAccount
from athenian.api.precompute.context import PrecomputeContext


async def _record_performance_metrics(
    prometheus_endpoint: str,
    rdb: Database,
) -> None:
    log = logging.getLogger(f"{metadata.__package__}._record_performance_metrics")
    for pct in ("50", "95"):
        inserted = []
        for attempt in range(3):
            async with aiohttp.ClientSession() as session:
                now = datetime.now(timezone.utc)
                async with session.get(
                    f"{prometheus_endpoint}/api/v1/query",
                    params={
                        "query": (
                            f"histogram_quantile(0.{pct}, "
                            "sum(rate(request_latency_seconds_bucket[24h])) by "
                            "(endpoint, account, le))"
                        ),
                    },
                ) as response:
                    if not response.ok:
                        log.error(
                            "[%d] failed to query Prometheus: %d: %s",
                            attempt + 1,
                            response.status,
                            await response.text(),
                        )
                        continue
                    for obj in (await response.json())["data"]["result"]:
                        if (value := obj["value"][1]) != "NaN" and (
                            account := obj["metric"]["account"]
                        ) != "N/A":
                            inserted.append(
                                HealthMetric(
                                    account_id=int(account),
                                    name=f'p{pct}/{obj["metric"]["endpoint"]}',
                                    created_at=now,
                                    value=float(value),
                                ).explode(with_primary_keys=True),
                            )
                    break
        if inserted:
            log.info("inserting %d p%s records", len(inserted), pct)
            await rdb.execute_many(insert(HealthMetric), inserted)


async def _record_inconsistency_metrics(
    prometheus_endpoint: str,
    acc_id_map_task: asyncio.Task,
    rdb: Database,
) -> None:
    log = logging.getLogger(f"{metadata.__package__}._record_inconsistency_metrics")
    inserted = []
    for attempt in range(3):
        async with aiohttp.ClientSession() as session:
            now = datetime.now(timezone.utc)
            async with session.get(
                f"{prometheus_endpoint}/api/v1/query",
                params={
                    "query": "metadata_github_consistency_nodes_issues",
                },
            ) as response:
                if not response.ok:
                    log.error(
                        "[%d] failed to query Prometheus: %d: %s",
                        attempt + 1,
                        response.status,
                        await response.text(),
                    )
                    continue
                objs = (await response.json())["data"]["result"]
                await acc_id_map_task
                acc_id_map = acc_id_map_task.result()
                for obj in objs:
                    try:
                        account = acc_id_map[int(obj["metric"]["acc_id"])]
                    except KeyError:
                        continue
                    inserted.append(
                        HealthMetric(
                            account_id=account,
                            name=f'inconsistency/{obj["metric"]["node_type"]}',
                            created_at=now,
                            value=int(obj["value"][1]),
                        ).explode(with_primary_keys=True),
                    )
    if inserted:
        log.info("inserting %d data inconsistency records", len(inserted))
        await rdb.execute_many(insert(HealthMetric), inserted)


async def _record_pending_fetch_metrics(
    acc_id_map_task: asyncio.Task,
    mdb: Database,
    rdb: Database,
) -> None:
    log = logging.getLogger(f"{metadata.__package__}._record_pending_fetch_metrics")
    inserted = []
    # df_pending_prs, df_pending_commits = await gather()
    if inserted:
        log.info("inserting %d data inconsistency records", len(inserted))
        await rdb.execute_many(insert(HealthMetric), inserted)


async def _fetch_acc_id_map(sdb: Database) -> dict[int, int]:
    return dict(
        await sdb.fetch_all(select(AccountGitHubAccount.id, AccountGitHubAccount.account_id)),
    )


async def main(context: PrecomputeContext, args: argparse.Namespace) -> Any:
    """Fill missing commit references in the deployed components."""
    acc_id_map_task = asyncio.create_task(_fetch_acc_id_map(context.sdb), name="_fetch_acc_id_map")
    await gather(
        _record_performance_metrics(args.prometheus, context.rdb),
        _record_inconsistency_metrics(args.prometheus, acc_id_map_task, context.rdb),
        _record_pending_fetch_metrics(acc_id_map_task, context.mdb, context.rdb),
    )
