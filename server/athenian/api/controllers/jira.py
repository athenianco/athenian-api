import marshal
from typing import Dict, Iterable, List, Optional, Tuple

import aiomcache
from sqlalchemy import and_, select

from athenian.api.async_utils import gather
from athenian.api.cache import cached, max_exptime
from athenian.api.models.metadata.jira import Project, User as JIRAUser
from athenian.api.models.state.models import AccountJiraInstallation, JIRAProjectSetting, \
    MappedJIRAIdentity
from athenian.api.models.web import NoSourceDataError
from athenian.api.response import ResponseError
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


@sentry_span
async def load_mapped_jira_users(account: int,
                                 github_user_ids: Iterable[str],
                                 sdb: DatabaseLike,
                                 mdb: DatabaseLike,
                                 cache: Optional[aiomcache.Client],
                                 ) -> Dict[str, str]:
    """Fetch the map from GitHub developer IDs to JIRA names."""
    (mapping, cached), sentinel = await gather(
        _load_mapped_jira_users(account, github_user_ids, sdb, mdb, cache),
        load_jira_identity_mapping_sentinel(account, cache))
    if not sentinel and cached:
        mapping, _ = await _load_mapped_jira_users.__wrapped__(
            account, github_user_ids, sdb, mdb, None)
    return mapping


@cached(
    exptime=max_exptime,  # we drop _load_jira_identity_mapping_sentinel when we re-compute
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda account, github_user_ids, **_: (account, sorted(github_user_ids)),
    postprocess=lambda result, **_: (result[0], True),
    refresh_on_access=True,
)
async def _load_mapped_jira_users(account: int,
                                  github_user_ids: Iterable[str],
                                  sdb: DatabaseLike,
                                  mdb: DatabaseLike,
                                  cache: Optional[aiomcache.Client],
                                  ) -> Tuple[Dict[str, str], bool]:
    tasks = [
        sdb.fetch_all(
            select([MappedJIRAIdentity.github_user_id, MappedJIRAIdentity.jira_user_id])
            .where(and_(MappedJIRAIdentity.account_id == account,
                        MappedJIRAIdentity.github_user_id.in_(github_user_ids)))),
        get_jira_installation_or_none(account, sdb, mdb, cache),
    ]
    map_rows, jira_ids = await gather(*tasks)
    if jira_ids is None:
        return {}, False
    jira_user_ids = {r[1]: r[0] for r in map_rows}
    name_rows = await mdb.fetch_all(
        select([JIRAUser.id, JIRAUser.display_name])
        .where(and_(JIRAUser.acc_id == jira_ids[0],
                    JIRAUser.id.in_(jira_user_ids))))
    return {jira_user_ids[row[0]]: row[1] for row in name_rows}, False


@cached(
    exptime=max_exptime,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda account, **_: (account,),
    postprocess=lambda result, account, **_: True,  # if we loaded from cache, return True
    refresh_on_access=True,
)
async def load_jira_identity_mapping_sentinel(account: int,
                                              cache: Optional[aiomcache.Client],
                                              ) -> bool:
    """Load the value indicating whether the JIRA identity mapping cache is valid."""
    # we evaluated => cache miss
    return False
