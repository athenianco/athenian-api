import argparse
from datetime import datetime, timezone
from typing import List

from sqlalchemy import desc, select
from tqdm import tqdm

from athenian.api.controllers.reposet import load_account_state
from athenian.api.models.state.models import Account
from athenian.api.precompute.context import PrecomputeContext


async def main(context: PrecomputeContext, args: argparse.Namespace) -> List[int]:
    """Load all accounts, find which must be precomputed, and return their IDs."""
    log, sdb = context.log, context.sdb
    accounts = [r[0] for r in await sdb.fetch_all(
        select([Account.id])
        .where(Account.expires_at > datetime.now(timezone.utc))
        .order_by(desc(Account.created_at)))]
    log.info("Checking progress of %d accounts", len(accounts))
    result = []
    for account in tqdm(accounts):
        state = await load_account_state(
            account, sdb, context.mdb, context.cache, context.slack, log=log)
        if state is None:
            log.info("Skipped account %d because it is not installed", account)
            continue
        if state.finished_date is None:
            log.warning("Skipped account %d because the progress is not 100%%", account)
            continue
        result.append(account)
    return result
