from datetime import datetime, timezone
from typing import Dict, Optional

import aiomcache
import aiosqlite
from sqlalchemy import join, select, union_all

from athenian.api.async_utils import gather
from athenian.api.controllers.account import get_metadata_account_ids_or_empty
from athenian.api.controllers.jira import get_jira_installation_or_none
from athenian.api.db import DatabaseLike
from athenian.api.models.metadata.github import NodeCheckRun, NodeStatusContext
from athenian.api.models.state.models import Account, UserAccount
from athenian.api.models.web import AccountStatus


# There are three GitHub user types:
# 1. Regular.
# 2. Bots.
# 3. Mannequins.
# GitHub creates (3) during the import.
MANNEQUIN_PREFIX = "mannequin"


async def load_user_accounts(uid: str,
                             sdb: DatabaseLike,
                             mdb: DatabaseLike,
                             cache: Optional[aiomcache.Client],
                             ) -> Dict[int, AccountStatus]:
    """Fetch the user accounts membership and flags."""
    accounts = await sdb.fetch_all(
        select([UserAccount, Account.expires_at])
        .select_from(join(UserAccount, Account, UserAccount.account_id == Account.id))
        .where(UserAccount.user_id == uid))
    try:
        is_sqlite = sdb.url.dialect == "sqlite"
    except AttributeError:
        is_sqlite = isinstance(sdb.raw_connection, aiosqlite.Connection)
    now = datetime.now(None if is_sqlite else timezone.utc)
    tasks = [
        get_jira_installation_or_none(x[UserAccount.account_id.key], sdb, mdb, cache)
        for x in accounts
    ] + [
        get_metadata_account_ids_or_empty(x[UserAccount.account_id.key], sdb, cache)
        for x in accounts
    ]
    results = await gather(*tasks, op="account_ids")
    jira_ids = results[:len(accounts)]
    if is_sqlite:
        def build_query(meta_ids):
            return union_all(
                select([NodeCheckRun.id])
                .where(NodeCheckRun.acc_id.in_(meta_ids)),
                select([NodeStatusContext.id])
                .where(NodeStatusContext.acc_id.in_(meta_ids)),
            ).limit(1)
    else:
        def build_query(meta_ids):
            return union_all(
                select([NodeCheckRun.id])
                .where(NodeCheckRun.acc_id.in_(meta_ids))
                .limit(1),
                select([NodeStatusContext.id])
                .where(NodeStatusContext.acc_id.in_(meta_ids))
                .limit(1),
            )
    tasks = [
        mdb.fetch_val(build_query(meta_ids)) if meta_ids else _return_none()
        for meta_ids in results[len(accounts):]
    ]
    check_runs = await gather(*tasks, op="check_runs")
    return {
        x[UserAccount.account_id.key]: AccountStatus(
            is_admin=x[UserAccount.is_admin.key],
            expired=x[Account.expires_at.key] < now,
            has_jira=jira_id is not None,
            has_ci=check_run is not None,
        )
        for x, jira_id, check_run in zip(accounts, jira_ids, check_runs)
    }


async def _return_none() -> None:
    return None
