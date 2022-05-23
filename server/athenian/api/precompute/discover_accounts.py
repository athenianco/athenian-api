import argparse
from datetime import datetime, timezone
from itertools import chain
from typing import Dict, List, Tuple, Union

import sqlalchemy as sa
from tqdm import tqdm

from athenian.api.db import Database
from athenian.api.internal.reposet import load_account_state
from athenian.api.models.state.models import Account, RepositorySet
from athenian.api.precompute.context import PrecomputeContext


async def main(
    context: PrecomputeContext, args: argparse.Namespace,
) -> Union[List[int], Dict[str, List[int]]]:
    """Load all accounts, find which must be precomputed, and return their IDs.

    With partition argument accounts are returned in two groups:
    - never precomputed accounts ("fresh" group)
    - already precomputed accounts ("precomputed" group)

    """
    log, sdb = context.log, context.sdb

    accounts = await _get_accounts(sdb)
    log.info("Checking progress of %d accounts", len(accounts))

    discovered: Dict[str, List[int]] = {"precomputed": [], "fresh": []}
    for account, precomputed in tqdm(accounts):
        state = await load_account_state(
            account, sdb, context.mdb, context.cache, context.slack, log=log)
        if state is None:
            log.info("Skipped account %d because it is not installed", account)
            continue
        if state.finished_date is None:
            log.warning("Skipped account %d because the progress is not 100%%", account)
            continue

        discovered["precomputed" if precomputed else "fresh"].append(account)

    if args.partition:
        return discovered
    else:
        return sorted(chain(*discovered.values()))


async def _get_accounts(sdb: Database) -> List[Tuple[int, bool]]:
    """Return the existing account IDs, each with a precomputed flag."""
    # already precomputed accounts have an ALL repository set that is precomputed
    # outer join will return one row per account, since (owner_id, name) is unique in RepositorySet
    join_cond = sa.and_(
        Account.id == RepositorySet.owner_id,
        RepositorySet.name == RepositorySet.ALL,
    )
    stmt = sa.select(
        [Account.id, sa.func.coalesce(RepositorySet.precomputed, False).label("precomputed")],
    ).join(
        RepositorySet, join_cond, isouter=True,
    ).where(
        Account.expires_at > datetime.now(timezone.utc),
    )
    return await sdb.fetch_all(stmt)
