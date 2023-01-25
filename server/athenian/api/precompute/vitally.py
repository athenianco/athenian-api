import argparse
import logging
import os

import aiohttp
from sqlalchemy import insert

from athenian.api import metadata
from athenian.api.models.persistentdata.models import VitallyAccount
from athenian.api.precompute.context import PrecomputeContext


async def main(context: PrecomputeContext, args: argparse.Namespace) -> None:
    """Store the most recent information about accounts from Vitally."""
    log = logging.getLogger(f"{metadata.__package__}.record_vitally")
    cursor = {"limit": 100}
    accounts = []
    token = os.environ["VITALLY_TOKEN"]
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
                        accounts.append(
                            VitallyAccount(
                                account_id=int(
                                    acc.get("externalId", acc.get("traits", {}).get("id")),
                                ),
                                name=acc.get("name"),
                                mrr=acc.get("mrr"),
                                health_score=acc.get("healthScore"),
                            )
                            .create_defaults()
                            .explode(with_primary_keys=True),
                        )
    if accounts:
        await context.rdb.execute_many(insert(VitallyAccount), accounts)
