import argparse
import asyncio
from datetime import datetime, timezone
import logging
from typing import Any

import aiohttp
from sqlalchemy import insert, join, select

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.db import Database, dialect_specific_insert
from athenian.api.models.persistentdata.models import HealthMetric
from athenian.api.models.state.models import Account, AccountGitHubAccount
from athenian.api.precompute.context import PrecomputeContext

endpoints = [
    "/v1/(metrics|histograms)/.*",
    "/v1/(get|filter|paginate)/.*",
    "/v1/(events|invite|settings)/.*",
    "/private/.*",
]


async def _record_performance_metrics(
    prometheus_endpoint: str,
    rdb: Database,
) -> None:
    log = logging.getLogger(f"{metadata.__package__}._record_performance_metrics")
    for pct in ("50", "95"):
        inserted = []
        for endpoint in endpoints:
            for attempt in range(3):
                async with aiohttp.ClientSession() as session:
                    now = datetime.now(timezone.utc)
                    async with session.get(
                        f"{prometheus_endpoint}/api/v1/query",
                        params={
                            "query": (
                                f"histogram_quantile(0.{pct}, "
                                "sum(rate(request_latency_seconds_bucket"
                                f'{{endpoint=~"{endpoint}"}}'
                                "[24h])) by (endpoint, account, le))"
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
                                        name=f'p{pct}{obj["metric"]["endpoint"]}',
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
    now = datetime.now(timezone.utc)
    pending_prs_rows, pending_branches_rows, _ = await gather(
        mdb.fetch_all(
            """
        SELECT edges.acc_id, (edges.value - nodes.value) AS diff
        FROM (
            SELECT
                re.acc_id, count(*) AS value
            FROM github.node_repository_edge_pullrequests re
                JOIN github.account_repos pa
                    ON re.parent_id = pa.repo_graph_id AND re.acc_id = pa.acc_id
            GROUP BY re.acc_id
        ) AS edges JOIN (
            SELECT pr.acc_id, count(DISTINCT pr.database_id) AS value
            FROM github.node_pullrequest pr
            WHERE
            EXISTS (
                SELECT 1 FROM github.graph_nodes prn
                WHERE prn.acc_id = pr.acc_id AND prn.node_id = pr.graph_id AND NOT prn.deleted)
            AND
            EXISTS (
                SELECT 1 FROM github.account_repos pa
                WHERE pr.repository_id = pa.repo_graph_id AND pr.acc_id = pa.acc_id)
            GROUP BY pr.acc_id
        ) AS nodes
        ON edges.acc_id = nodes.acc_id;
        """,
        ),
        mdb.fetch_all(
            """
        SELECT edges.acc_id, (edges.value - nodes.value) AS diff
        FROM (
            SELECT
                re.acc_id, count(*) AS value
            FROM github.node_repository_edge_refs re
                JOIN github.account_repos pa
                    ON re.parent_id = pa.repo_graph_id AND re.acc_id = pa.acc_id
            GROUP BY re.acc_id
        ) AS edges JOIN (
            SELECT ref.acc_id, count(*) AS value
            FROM github.node_ref ref
            WHERE EXISTS (
                SELECT 1 FROM github.graph_nodes prn
                WHERE prn.acc_id = ref.acc_id AND prn.node_id = ref.graph_id AND not prn.deleted)
            AND EXISTS (
                SELECT 1 FROM github.account_repos pa
                WHERE pa.acc_id = ref.acc_id AND pa.repo_graph_id = ref.repository_id)
            GROUP BY ref.acc_id
        ) AS nodes
        ON edges.acc_id = nodes.acc_id;
        """,
        ),
        acc_id_map_task,
    )
    acc_id_map = acc_id_map_task.result()
    for name, rows in (
        ("pending_prs", pending_prs_rows),
        ("pending_branches", pending_branches_rows),
    ):
        for row in rows:
            acc, diff = row
            try:
                acc = acc_id_map[acc]
            except KeyError:
                continue
            inserted.append(
                HealthMetric(
                    account_id=acc,
                    created_at=now,
                    name=name,
                    value=diff,
                ).explode(with_primary_keys=True),
            )
    if inserted:
        log.info("inserting %d pending fetch records", len(inserted))
        await rdb.execute_many(
            (await dialect_specific_insert(rdb))(HealthMetric).on_conflict_do_nothing(), inserted,
        )


async def _fetch_acc_id_map(sdb: Database) -> dict[int, int]:
    return dict(
        await sdb.fetch_all(
            select(AccountGitHubAccount.id, AccountGitHubAccount.account_id)
            .select_from(
                join(AccountGitHubAccount, Account, AccountGitHubAccount.account_id == Account.id),
            )
            .where(Account.expires_at > datetime.now(timezone.utc)),
        ),
    )


async def main(context: PrecomputeContext, args: argparse.Namespace) -> Any:
    """Fill missing commit references in the deployed components."""
    acc_id_map_task = asyncio.create_task(_fetch_acc_id_map(context.sdb), name="_fetch_acc_id_map")
    await gather(
        _record_performance_metrics(args.prometheus, context.rdb),
        _record_inconsistency_metrics(args.prometheus, acc_id_map_task, context.rdb),
        _record_pending_fetch_metrics(acc_id_map_task, context.mdb, context.rdb),
    )
