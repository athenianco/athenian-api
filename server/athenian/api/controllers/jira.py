import marshal
from typing import Optional

import aiomcache
from sqlalchemy import select

from athenian.api import ResponseError
from athenian.api.cache import cached, max_exptime
from athenian.api.models.state.models import AccountJiraInstallation
from athenian.api.models.web import NoSourceDataError
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import DatabaseLike


@sentry_span
@cached(
    exptime=max_exptime,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda account, **_: (account,),
    refresh_on_access=True,
)
async def get_jira_installation(account: int,
                                sdb: DatabaseLike,
                                cache: Optional[aiomcache.Client],
                                ) -> int:
    """
    Retrieve the JIRA installation ID belonging to the account or raise an exception.

    Raise ResponseError if no installation exists.
    """
    jira_id = await sdb.fetch_val(select([AccountJiraInstallation.id])
                                  .where(AccountJiraInstallation.account_id == account))
    if jira_id is None:
        raise ResponseError(NoSourceDataError(
            detail="JIRA has not been installed to the metadata yet."))
    return jira_id


async def get_jira_installation_or_none(account: int,
                                        sdb: DatabaseLike,
                                        cache: Optional[aiomcache.Client],
                                        ) -> Optional[int]:
    """
    Retrieve the JIRA installation ID belonging to the account or raise an exception.

    Return None if no installation exists.
    """
    try:
        return await get_jira_installation(account, sdb, cache)
    except ResponseError:
        return None
