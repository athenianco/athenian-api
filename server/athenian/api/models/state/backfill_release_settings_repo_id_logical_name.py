from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import logging
from typing import Optional

from flogging import flogging
import sqlalchemy as sa

from athenian.api.async_utils import gather
from athenian.api.db import Database, Row, measure_db_overhead_and_retry
from athenian.api.internal.account import get_multiple_metadata_account_ids
from athenian.api.internal.prefixer import RepositoryName
from athenian.api.models.metadata.github import NodeRepository
from athenian.api.models.state.models import ReleaseSetting, RepositorySet

log = logging.getLogger("backfill_release_settings_repo_id_logical_name")


async def _main() -> None:
    args = _get_args()
    sdb = measure_db_overhead_and_retry(Database(args.state_db))
    mdb = measure_db_overhead_and_retry(Database(args.metadata_db))

    await gather(sdb.connect(), mdb.connect())
    await _backfill(args.overwrite, sdb, mdb)


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    flogging.add_logging_args(parser)
    parser.add_argument("state_db")
    parser.add_argument("metadata_db")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite already backfilled repo_id and logical_name",
    )
    return parser.parse_args()


async def _backfill(overwrite: bool, sdb: Database, mdb: Database) -> None:
    stmt = sa.select(ReleaseSetting)
    if not overwrite:
        stmt = stmt.where(ReleaseSetting.repo_id.is_(None)).order_by(ReleaseSetting.account_id)

    reposets_access = _ReposetsAccess(sdb)
    rel_settings = await sdb.fetch_all(stmt)

    log.info("updating repo_id in %d release_setting rows", len(rel_settings))

    accounts = {r[ReleaseSetting.account_id.name] for r in rel_settings}
    accounts_meta_ids = await get_multiple_metadata_account_ids(accounts, sdb, None)
    PARALLEL_TASKS = 6

    for batch in [
        rel_settings[i : i + PARALLEL_TASKS] for i in range(0, len(rel_settings), PARALLEL_TASKS)
    ]:
        coros = [_backfill_row(row, accounts_meta_ids, reposets_access, sdb, mdb) for row in batch]
        await gather(*coros)


async def _backfill_row(
    rel_setting: ReleaseSetting,
    accounts_meta_ids: dict[int, list[int]],
    reposets_access: _ReposetsAccess,
    sdb: Database,
    mdb: Database,
) -> None:
    account = rel_setting[ReleaseSetting.account_id.name]
    meta_ids = accounts_meta_ids[account]

    name = RepositoryName.from_prefixed(orig_name := rel_setting[ReleaseSetting.repository.name])

    md_repo_id_stmt = sa.select(NodeRepository.id).where(
        NodeRepository.acc_id.in_(meta_ids),
        NodeRepository.name_with_owner == name.unprefixed_physical,
    )
    md_repo_id = await mdb.fetch_val(md_repo_id_stmt)
    if md_repo_id is None:
        log.error("Repository <%s;account=%d;meta_ids=%s> not found", name, account, meta_ids)
        reposet = await reposets_access.get(account)
        if reposet is None:
            log.error("  reposet missing for account %d", account)
        elif orig_name not in [item[0] for item in reposet[RepositorySet.items.name]]:
            log.error("  reposet for account misses this repo")
        return

    update_values = {
        ReleaseSetting.logical_name: name.logical or "",
        ReleaseSetting.repo_id: md_repo_id,
        ReleaseSetting.updated_at: datetime.now(timezone.utc),
    }
    update_stmt = (
        sa.update(ReleaseSetting)
        .where(ReleaseSetting.account_id == account, ReleaseSetting.repository == orig_name)
        .values(update_values)
    )
    await sdb.execute(update_stmt)


class _ReposetsAccess:
    def __init__(self, sdb: Database):
        self._sdb = sdb
        self._reposets = {}

    async def get(self, account_id: int) -> Optional[Row]:
        if account_id in self._reposets:
            return self._reposets[account_id]

        row = await self._sdb.fetch_one(
            sa.select(RepositorySet).where(RepositorySet.owner_id == account_id),
        )
        self._reposets[account_id] = row
        return row


if __name__ == "__main__":
    asyncio.run(_main())
