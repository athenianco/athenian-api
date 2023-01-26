import argparse
import logging
import os

import aiohttp

from athenian.api import metadata
from athenian.api.db import dialect_specific_insert
from athenian.api.models.persistentdata.models import VitallyAccount
from athenian.api.precompute.context import PrecomputeContext


async def main(context: PrecomputeContext, args: argparse.Namespace) -> None:
    """Store the most recent information about accounts from Vitally."""
    log = logging.getLogger(f"{metadata.__package__}.record_vitally")
    cursor = {"limit": 100}
    accounts = []
    if not (token := os.getenv("VITALLY_TOKEN")):
        log.warning("skipped - must define VITALLY_TOKEN")
        return
    while cursor is not None:
        for attempt in range(3):
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://rest.vitally.io/resources/accounts",
                    params=cursor,
                    headers={"Authorization": "Basic " + token},
                ) as response:
                    if not response.ok:
                        log.error(
                            "[%d] failed to query Vitally: %d: %s",
                            attempt + 1,
                            response.status,
                            await response.text(),
                        )
                        continue
                    body = await response.json()
                    if next_page := body.get("next"):
                        cursor["from"] = next_page
                    else:
                        cursor = None
                    for acc in body["results"]:
                        if arr := acc.get("arr", (acc.get("mrr") or 0) * 12):
                            mrr = arr // 12
                        else:
                            mrr = None
                        accounts.append(
                            VitallyAccount(
                                account_id=int(
                                    acc.get("externalId", acc.get("traits", {}).get("id")),
                                ),
                                name=acc.get("name"),
                                mrr=mrr,
                                health_score=acc.get("healthScore"),
                            )
                            .create_defaults()
                            .explode(with_primary_keys=True),
                        )
    if accounts:
        log.info("updating %d accounts", len(accounts))
        sql = (await dialect_specific_insert(context.rdb))(VitallyAccount)
        sql = sql.on_conflict_do_update(
            index_elements=VitallyAccount.__table__.primary_key.columns,
            set_={
                col.name: getattr(sql.excluded, col.name)
                for col in (
                    VitallyAccount.name,
                    VitallyAccount.mrr,
                    VitallyAccount.health_score,
                    VitallyAccount.updated_at,
                )
            },
        )
        await context.rdb.execute_many(sql, accounts)
