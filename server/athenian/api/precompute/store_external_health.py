import argparse
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
):
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
    sdb: Database,
    rdb: Database,
):
    log = logging.getLogger(f"{metadata.__package__}._record_inconsistency_metrics")
    acc_id_map = dict(
        await sdb.fetch_all(select(AccountGitHubAccount.id, AccountGitHubAccount.account_id)),
    )
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
                for obj in (await response.json())["data"]["result"]:
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


async def main(context: PrecomputeContext, args: argparse.Namespace) -> Any:
    """Fill missing commit references in the deployed components."""
    await gather(
        _record_performance_metrics(args.prometheus, context.rdb),
        _record_inconsistency_metrics(args.prometheus, context.sdb, context.rdb),
    )
