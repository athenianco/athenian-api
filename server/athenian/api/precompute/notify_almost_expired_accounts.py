import argparse
from datetime import datetime, timedelta, timezone
from itertools import chain

from sqlalchemy import and_, select

from athenian.api.async_utils import gather
from athenian.api.controllers.account import get_metadata_account_ids_or_empty
from athenian.api.models.metadata.github import Account as GitHubAccount
from athenian.api.models.state.models import Account, UserAccount
from athenian.api.precompute.context import PrecomputeContext


async def main(context: PrecomputeContext, args: argparse.Namespace) -> None:
    """Find accounts which will expire within 1 hour and report them to Slack."""
    log, sdb, mdb, cache = context.log, context.sdb, context.mdb, context.cache
    right = datetime.now(timezone.utc) + timedelta(days=1)
    left = right - timedelta(hours=1)
    accounts = dict(await sdb.fetch_all(
        select([Account.id, Account.expires_at])
        .where(Account.expires_at.between(left, right))))
    if not accounts:
        return
    log.info("Notifying about almost expired accounts: %s", sorted(accounts))
    user_rows, *meta_ids = await gather(
        sdb.fetch_all(
            select([UserAccount.account_id, UserAccount.user_id])
            .where(and_(UserAccount.account_id.in_(accounts),
                        UserAccount.is_admin)),
        ),
        *(get_metadata_account_ids_or_empty(acc, sdb, cache) for acc in accounts),
    )
    users = dict(user_rows)
    name_rows = await mdb.fetch_all(
        select([GitHubAccount.id, GitHubAccount.name])
        .where(GitHubAccount.id.in_(chain.from_iterable(meta_ids))))
    names = dict(name_rows)
    names = {acc: ", ".join(names[i] for i in m) for acc, m in zip(accounts, meta_ids)}
    await gather(*(
        context.slack.post_account(
            "almost_expired.jinja2",
            account=acc,
            name=names[acc],
            user=users[acc],
            expires=expires)
        for acc, expires in accounts.items()
    ))
