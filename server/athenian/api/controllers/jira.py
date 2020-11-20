import marshal
from typing import List, Optional, Tuple

import aiomcache
from sqlalchemy import and_, select

from athenian.api import ResponseError
from athenian.api.async_utils import gather
from athenian.api.cache import cached, max_exptime
from athenian.api.models.metadata.jira import Project
from athenian.api.models.state.models import AccountJiraInstallation, JIRAProjectSetting
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
async def get_jira_id(account: int,
                      sdb: DatabaseLike,
                      cache: Optional[aiomcache.Client],
                      ) -> int:
    """
    Retrieve the JIRA installation ID belonging to the account or raise an exception.

    Raise ResponseError if no installation exists.

    :return: JIRA installation ID and the list of enabled JIRA project IDs.
    """
    jira_id = await sdb.fetch_val(select([AccountJiraInstallation.id])
                                  .where(AccountJiraInstallation.account_id == account))
    if jira_id is None:
        raise ResponseError(NoSourceDataError(
            detail="JIRA has not been installed to the metadata yet."))
    return jira_id


@sentry_span
async def get_jira_installation(account: int,
                                sdb: DatabaseLike,
                                mdb: DatabaseLike,
                                cache: Optional[aiomcache.Client],
                                ) -> Tuple[int, List[str]]:
    """
    Retrieve the JIRA installation ID belonging to the account or raise an exception.

    Raise ResponseError if no installation exists.

    :return: JIRA installation ID and the list of enabled JIRA project IDs.
    """
    jira_id = await get_jira_id(account, sdb, cache)
    tasks = [
        mdb.fetch_all(select([Project.id, Project.key]).where(Project.acc_id == jira_id)),
        sdb.fetch_all(select([JIRAProjectSetting.key])
                      .where(and_(JIRAProjectSetting.account_id == account,
                                  JIRAProjectSetting.enabled.is_(False)))),
    ]
    projects, disabled = await gather(*tasks, op="load JIRA projects")
    disabled = {r[0] for r in disabled}
    projects = sorted(r[0] for r in projects if r[1] not in disabled)
    return jira_id, projects


async def get_jira_installation_or_none(account: int,
                                        sdb: DatabaseLike,
                                        mdb: DatabaseLike,
                                        cache: Optional[aiomcache.Client],
                                        ) -> Optional[Tuple[int, List[str]]]:
    """
    Retrieve the JIRA installation ID belonging to the account or raise an exception.

    Return None if no installation exists.

    :return: JIRA installation ID and the list of enabled JIRA project IDs.
    """
    try:
        return await get_jira_installation(account, sdb, mdb, cache)
    except ResponseError:
        return None
