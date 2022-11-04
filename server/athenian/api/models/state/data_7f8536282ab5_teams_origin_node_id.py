#!/usr/bin/env python3

import argparse
import asyncio
from datetime import datetime, timezone
import logging
from typing import Sequence

from flogging import flogging
import sqlalchemy as sa

from athenian.api.async_utils import gather
from athenian.api.db import Database, DatabaseLike, Row, measure_db_overhead_and_retry
from athenian.api.internal.account import get_multiple_metadata_account_ids
from athenian.api.models.metadata.github import (
    Organization as MetadataOrganization,
    Team as MetadataTeam,
)
from athenian.api.models.state.models import Account, Team

log = logging.getLogger("backfill_teams_meta_id")


async def _main() -> None:
    args = _get_args()
    sdb = measure_db_overhead_and_retry(Database(args.state_db))
    mdb = measure_db_overhead_and_retry(Database(args.metadata_db))

    await gather(sdb.connect(), mdb.connect())
    await _backfill_multiple_accounts(args.overwrite, args.account, sdb, mdb)


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    flogging.add_logging_args(parser)
    parser.add_argument("state_db")
    parser.add_argument("metadata_db")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing meta id for teams already having one",
    )
    parser.add_argument("account", nargs="*", type=int, help="backfill only the given accounts")
    return parser.parse_args()


async def _backfill_multiple_accounts(
    overwrite: bool,
    accounts: list[int],
    sdb: Database,
    mdb: Database,
) -> None:
    if not accounts:
        accounts = [r[0] for r in await sdb.fetch_all(sa.select(Account.id))]
    accounts_meta_ids = await get_multiple_metadata_account_ids(accounts, sdb, None)

    for account, meta_ids in sorted(accounts_meta_ids.items()):
        await _backfill_teams_origin_node_id(account, meta_ids, sdb, mdb, overwrite)


async def _backfill_teams_origin_node_id(
    account: int,
    meta_ids: Sequence[int],
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    overwrite: bool = False,
) -> None:
    """Fill the Team.origin_node_id values in sdb by matching mdb teams by name."""
    log.info("backfilling teams meta id for account %d", account)

    state_teams, meta_team_name_to_node_id = await gather(
        _get_state_teams(account, not overwrite, sdb),
        _get_meta_team_name_to_node_id(meta_ids, mdb),
    )

    updates = []
    not_mapped = []
    for state_team in state_teams:
        try:
            team_node_id = meta_team_name_to_node_id[state_team[Team.name.name]]
        except KeyError:
            log.debug("team %s not found in metadata DB", state_team[Team.name.name])
            not_mapped.append(Team.name.name)
            continue

        update = (
            sa.update(Team)
            .where(Team.id == state_team[Team.id.name])
            .values(
                {Team.origin_node_id: team_node_id, Team.updated_at: datetime.now(timezone.utc)},
            )
        )
        updates.append(update)
        log.debug("setting meta id %d for team %s", team_node_id, state_team[Team.name.name])

    batch_size = 5
    for batch in [
        updates[batch_start : batch_start + batch_size]
        for batch_start in range(0, len(updates), batch_size)
    ]:
        await gather(*map(sdb.execute, batch))
    log.info("%d teams mapped, %d team not mapped", len(updates), len(not_mapped))


async def _get_state_teams(
    account: int,
    exclude_mapped: bool,
    sdb: DatabaseLike,
) -> Sequence[Row]:
    where = [Team.owner_id == account, Team.name != Team.BOTS, Team.parent_id.isnot(None)]
    if exclude_mapped:
        where.append(Team.origin_node_id.is_(None))
    query = sa.select(Team.id, Team.name).where(*where)
    return await sdb.fetch_all(query)


async def _get_meta_team_name_to_node_id(
    meta_ids: Sequence[int],
    mdb: DatabaseLike,
) -> dict[str, int]:
    meta_orgs_query = sa.select(MetadataOrganization.id).where(
        MetadataOrganization.acc_id.in_(meta_ids),
    )
    query = sa.select(MetadataTeam.name, MetadataTeam.id).where(
        MetadataTeam.acc_id.in_(meta_ids), MetadataTeam.organization_id.in_(meta_orgs_query),
    )
    rows = await mdb.fetch_all(query)
    return dict(rows)


if __name__ == "__main__":
    asyncio.run(_main())
