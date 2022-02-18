from datetime import datetime, timezone
from typing import Callable, Coroutine, Dict, Optional

import aiomcache
import aiosqlite
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
from sqlalchemy import func, join, select, union_all

from athenian.api.async_utils import gather
from athenian.api.cache import cached, middle_term_exptime
from athenian.api.controllers.account import get_account_name, get_metadata_account_ids_or_empty
from athenian.api.controllers.jira import get_jira_installation_or_none
from athenian.api.db import Database, DatabaseLike
from athenian.api.defer import defer
from athenian.api.models.metadata.github import NodeCheckRun, NodeStatusContext
from athenian.api.models.persistentdata.models import DeploymentNotification
from athenian.api.models.state.models import Account, UserAccount
from athenian.api.models.web import AccountStatus, User

# There are three GitHub user types:
# 1. Regular.
# 2. Bots.
# 3. Mannequins.
# GitHub creates (3) during the import.
MANNEQUIN_PREFIX = "mannequin"


async def load_user_accounts(uid: str,
                             sdb: DatabaseLike,
                             mdb: DatabaseLike,
                             rdb: DatabaseLike,
                             slack: Optional[SlackWebClient],
                             user_info: Callable[..., Coroutine],
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
        async with sdb.raw_connection() as raw_connection:
            is_sqlite = isinstance(raw_connection, aiosqlite.Connection)
    now = datetime.now(None if is_sqlite else timezone.utc)
    tasks = [
        get_jira_installation_or_none(x[UserAccount.account_id.name], sdb, mdb, cache)
        for x in accounts
    ] + [
        get_metadata_account_ids_or_empty(x[UserAccount.account_id.name], sdb, cache)
        for x in accounts
    ] + [
        rdb.fetch_val(select([func.count(DeploymentNotification.name)])
                      .where(DeploymentNotification.account_id == x[UserAccount.account_id.name]))
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
        for meta_ids in results[len(accounts):2 * len(accounts)]
    ]
    check_runs = await gather(*tasks, op="check_runs")
    result = {}
    for account, jira_id, check_run, deployments_count in zip(
            accounts, jira_ids, check_runs, results[2 * len(accounts):]):
        account_id = account[UserAccount.account_id.name]
        expires_at = account[Account.expires_at.name]
        if expired := expires_at < now and slack is not None:
            await defer(
                report_user_account_expired(
                    uid, account_id, expires_at, sdb, mdb, user_info, slack, cache),
                "report_user_account_expired_to_slack")
        result[account_id] = AccountStatus(
            is_admin=account[UserAccount.is_admin.name],
            expired=expired,
            has_jira=jira_id is not None,
            has_ci=check_run is not None,
            has_deployments=deployments_count > 0,
        )
    return result


async def _return_none() -> None:
    return None


@cached(
    exptime=middle_term_exptime,
    serialize=lambda x: x,
    deserialize=lambda x: x,
    key=lambda user, account, **_: (user, account),
)
async def report_user_account_expired(user: str,
                                      account: int,
                                      expired_at: datetime,
                                      sdb: Database,
                                      mdb: Database,
                                      user_info: Callable[..., Coroutine],
                                      slack: Optional[SlackWebClient],
                                      cache: Optional[aiomcache.Client]):
    """Send a Slack message about the user who accessed an expired account."""
    async def dummy_user():
        return User(login="N/A")

    name, user_info = await gather(
        get_account_name(account, sdb, mdb, cache),
        user_info() if user_info is not None else dummy_user(),
    )
    await slack.post_account("user_account_expired.jinja2",
                             user=user,
                             user_name=user_info.login,
                             user_email=user_info.email,
                             account=account,
                             account_name=name,
                             expired_at=expired_at)
    return b"1"
