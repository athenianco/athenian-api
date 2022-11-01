#!/usr/bin/env python

import argparse
import asyncio

from flogging import flogging
import sqlalchemy as sa

from athenian.api.align.goals.dbaccess import create_default_goal_templates
from athenian.api.db import Database, measure_db_overhead_and_retry
from athenian.api.models.state.models import GoalTemplate


async def _main() -> None:
    args = _get_args()
    sdb = measure_db_overhead_and_retry(Database(args.state_db))
    await sdb.connect()
    await _insert(sdb, args)


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    flogging.add_logging_args(parser)
    parser.add_argument("state_db")
    parser.add_argument("--account", action="append", type=int)
    return parser.parse_args()


async def _insert(sdb: Database, args: argparse.Namespace) -> None:
    if args.account:
        accounts = args.account
    else:
        account_rows = await sdb.fetch_all(sa.select(GoalTemplate.account_id).distinct())
        accounts = [r[0] for r in account_rows]

    async with sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            for account in accounts:
                await create_default_goal_templates(account, sdb_conn)


if __name__ == "__main__":
    asyncio.run(_main())
