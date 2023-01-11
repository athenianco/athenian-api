from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import marshal
import pickle
import re
from typing import Iterable, NamedTuple, Optional, TypeVar

import aiomcache
from names_matcher import NamesMatcher
import numpy as np
from pluralizer import Pluralizer
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
import sqlalchemy as sa
from sqlalchemy import and_, func, insert, join, select
from unidecode import unidecode

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.cache import CancelCache, cached, max_exptime
from athenian.api.db import DatabaseLike, ensure_db_datetime_tz, is_postgresql
from athenian.api.internal.miners.github.contributors import load_organization_members
from athenian.api.models.metadata.github import NodePullRequestJiraIssues
from athenian.api.models.metadata.jira import (
    Issue,
    IssueType,
    Priority,
    Progress,
    Project,
    User as JIRAUser,
)
from athenian.api.models.state.models import (
    AccountJiraInstallation,
    JIRAProjectSetting,
    MappedJIRAIdentity,
)
from athenian.api.models.web import InstallationProgress, NoSourceDataError, TableFetchingProgress
from athenian.api.response import ResponseError
from athenian.api.tracing import sentry_span


@sentry_span
@cached(
    exptime=max_exptime,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda account, **_: (account,),
    refresh_on_access=True,
)
async def get_jira_id(
    account: int,
    sdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> int:
    """
    Retrieve the JIRA installation ID belonging to the account or raise an exception.

    Raise ResponseError if no installation exists.

    :return: JIRA installation ID
    """
    jira_id = await get_jira_id_or_none(account, sdb)
    return check_jira_installation(jira_id)


async def get_jira_id_or_none(account: int, sdb: DatabaseLike) -> Optional[int]:
    """Retrieve the JIRA installation ID belonging to the account.

    Return `None` if no installation exists.
    """
    return await sdb.fetch_val(
        select(AccountJiraInstallation.id).where(AccountJiraInstallation.account_id == account),
    )


class JIRAConfig(NamedTuple):
    """JIRA attributes: account ID, project id -> key mapping, epic issue types."""

    acc_id: int
    projects: dict[str, str]
    epics: dict[str, list[str]]

    def epic_candidate_types(self) -> set[str]:
        """Return all possible issue types that can be epics."""
        return set(chain.from_iterable(self.epics.values())).union({"epic"})

    def project_epic_pairs(self) -> np.ndarray:
        """Return project-epic issue types as numpy byte array."""
        rows = []
        for project_id, project_key in self.projects.items():
            types = self.epics.get(project_key, [])
            if "epic" not in types:
                types.append("epic")
            for val in types:
                rows.append(f"{project_id}/{val}")

        return np.array(rows, dtype="S")

    def translate_project_keys(self, project_keys: Iterable[str]) -> list[str]:
        """Translate project keys to project IDs.

        Unknown project keys are ignored.
        """
        key_to_id = {key: id_ for id_, key in self.projects.items()}
        return [id_ for k in project_keys if (id_ := key_to_id.get(k)) is not None]


@sentry_span
async def get_jira_installation(
    account: int,
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> JIRAConfig:
    """
    Retrieve the JIRA installation ID belonging to the account or raise an exception.

    Raise ResponseError if no installation exists.

    :return: JIRA installation ID and the list of enabled JIRA project IDs.
    """
    jira_id = await get_jira_id(account, sdb, cache)
    projects, disabled, epic_rows = await gather(
        mdb.fetch_all(
            select([Project.id, Project.key]).where(
                and_(Project.acc_id == jira_id, Project.is_deleted.is_(False)),
            ),
        ),
        sdb.fetch_all(
            select([JIRAProjectSetting.key]).where(
                and_(
                    JIRAProjectSetting.account_id == account,
                    JIRAProjectSetting.enabled.is_(False),
                ),
            ),
        ),
        mdb.fetch_all(
            select([IssueType.project_id, IssueType.normalized_name])
            .where(and_(IssueType.acc_id == jira_id, IssueType.is_epic))
            .distinct(),
        ),
        op="load JIRA projects",
    )
    disabled = {r[0] for r in disabled}
    projects = {r[0]: r[1] for r in projects if r[1] not in disabled}
    epics = {}
    for row in epic_rows:
        try:
            epics.setdefault(projects[row[IssueType.project_id.name]], []).append(
                row[IssueType.normalized_name.name],
            )
        except KeyError:
            # disabled project
            continue
    return JIRAConfig(jira_id, projects, epics)


async def get_jira_installation_or_none(
    account: int,
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> Optional[JIRAConfig]:
    """
    Retrieve the JIRA installation ID belonging to the account or raise an exception.

    Return None if no installation exists.

    :return: JIRA installation ID and the list of enabled JIRA project IDs.
    """
    try:
        return await get_jira_installation(account, sdb, mdb, cache)
    except ResponseError:
        return None


JIRAConfOrID = TypeVar("JIRAConfOrID", bound=int | JIRAConfig)


def check_jira_installation(jira: Optional[JIRAConfOrID]) -> JIRAConfOrID:
    """Check the JIRAConfig or account id, raise exception if it is None."""
    if jira is None:
        raise ResponseError(
            NoSourceDataError(detail="JIRA has not been installed to the metadata yet."),
        )
    return jira


class JIRAEntitiesMapper:
    """Maps various JIRA issue entities from their labels to their identifiers.

    Issue priorities and issue types are mapped by this object.  Project are instead mapped
    through JIRAConfig.
    """

    def __init__(
        self,
        priority_names_map: dict[str, list[str]],
        types_map: dict[str, list[str]],
    ) -> None:
        """Build a mapper using the given dictionaries as mapping source."""
        self._priority_names_map = priority_names_map
        self._types_map = types_map

    @classmethod
    async def load(cls, jira_acc_id: int, mdb: DatabaseLike) -> JIRAEntitiesMapper:
        """Load the mapper by loading data from DB."""
        priority_stmt = sa.select(Priority.name, Priority.id).where(Priority.acc_id == jira_acc_id)
        type_stmt = (
            sa.select(IssueType.name, IssueType.id)
            .where(IssueType.acc_id == jira_acc_id)
            .distinct()
        )

        priority_rows, type_rows = await gather(
            mdb.fetch_all(priority_stmt), mdb.fetch_all(type_stmt),
        )

        # priority and issue types are not normalized in their table, maps are built normalized
        priority_names_map = defaultdict(list)
        for priority_name, priority_id in priority_rows:
            priority_names_map[normalize_priority(priority_name)].append(priority_id)
        types_map = defaultdict(list)
        for type_, type_id in type_rows:
            types_map[normalize_issue_type(type_)].append(type_id)

        return cls(priority_names_map, types_map)

    def translate_priority_names(self, priority_names: Iterable[str]) -> list[str]:
        """Translate a set of normalized priority names to their identifiers.

        Unknown priority names are ignored.
        """
        map_ = self._priority_names_map
        return list(chain.from_iterable(map_.get(p, ()) for p in priority_names))

    def translate_types(self, types: Iterable[str]) -> list[str]:
        """Translate a set of normalized types to their identifiers.

        Unknown types are ignored.
        """
        return list(chain.from_iterable(self._types_map.get(p, ()) for p in types))


@cached(
    exptime=5,  # matches the webapp poll interval
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda account, **_: (account,),
)
async def fetch_jira_installation_progress(
    account: int,
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> InstallationProgress:
    """Load the JIRA installation progress for the specified account."""
    jira_id = (await get_jira_installation(account, sdb, mdb, cache)).acc_id
    rows = await mdb.fetch_all(
        select([Progress]).where(
            and_(
                Progress.acc_id == jira_id,
                Progress.is_initial,
            ),
        ),
    )
    tables = [
        TableFetchingProgress(
            fetched=r[Progress.current.name],
            total=r[Progress.total.name],
            name=r[Progress.event_type.name],
        )
        for r in rows
    ]
    started_at = min(r[Progress.started_at.name] for r in rows)
    if not all(t.total == t.fetched for t in tables):
        finished_at = None
    else:
        finished_at = max(r[Progress.end_at.name] for r in rows)

    started_at = ensure_db_datetime_tz(started_at, mdb)
    if finished_at is not None:
        finished_at = ensure_db_datetime_tz(finished_at, mdb)
    model = InstallationProgress(
        started_date=started_at,
        finished_date=finished_at,
        tables=tables,
    )
    return model


@sentry_span
async def resolve_projects(
    keys: Iterable[str],
    jira_acc: int,
    mdb: DatabaseLike,
) -> dict[str, str]:
    """Lookup JIRA project IDs by their keys."""
    rows = await mdb.fetch_all(
        select([Project.id, Project.key]).where(
            and_(Project.acc_id == jira_acc, Project.key.in_(keys)),
        ),
    )
    return {r[0]: r[1] for r in rows}


@sentry_span
async def load_mapped_jira_users(
    account: int,
    github_user_ids: Iterable[int],
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> dict[int, str]:
    """Fetch the map from GitHub developer IDs to JIRA names."""
    cache_dropped = not await load_jira_identity_mapping_sentinel(account, cache)
    try:
        return await _load_mapped_jira_users(
            account, github_user_ids, sdb, mdb, cache, cache_dropped,
        )
    except ResponseError:
        return {}


def _postprocess_load_mapped_jira_users(
    result: dict[str, str],
    cache_dropped: bool,
    **_,
) -> dict[str, str]:
    if not cache_dropped:
        return result
    raise CancelCache()


@cached(
    exptime=max_exptime,  # we drop load_jira_identity_mapping_sentinel when we re-compute
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda account, github_user_ids, **_: (account, sorted(github_user_ids)),
    postprocess=_postprocess_load_mapped_jira_users,
    refresh_on_access=True,
)
async def _load_mapped_jira_users(
    account: int,
    github_user_ids: Iterable[int],
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
    cache_dropped: bool,
) -> dict[int, str]:
    tasks = [
        sdb.fetch_all(
            select([MappedJIRAIdentity.github_user_id, MappedJIRAIdentity.jira_user_id]).where(
                and_(
                    MappedJIRAIdentity.account_id == account,
                    MappedJIRAIdentity.github_user_id.in_(github_user_ids),
                ),
            ),
        ),
        get_jira_id(account, sdb, cache),
    ]
    map_rows, jira_id = await gather(*tasks)
    jira_user_ids = {r[1]: r[0] for r in map_rows}
    name_rows = await mdb.fetch_all(
        select([JIRAUser.id, JIRAUser.display_name]).where(
            and_(JIRAUser.acc_id == jira_id, JIRAUser.id.in_(jira_user_ids)),
        ),
    )
    return {jira_user_ids[row[0]]: row[1] for row in name_rows}


@cached(
    exptime=max_exptime,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda account, **_: (account,),
    postprocess=lambda result, account, **_: True,  # if we loaded from cache, return True
    refresh_on_access=True,
)
async def load_jira_identity_mapping_sentinel(
    account: int,
    cache: Optional[aiomcache.Client],
) -> bool:
    """Load the value indicating whether the JIRA identity mapping cache is valid."""
    # we evaluated => cache miss
    return False


async def match_jira_identities(
    account: int,
    meta_ids: tuple[int, ...],
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    slack: Optional[SlackWebClient],
    cache: Optional[aiomcache.Client],
) -> Optional[int]:
    """
    Perform GitHub -> JIRA identity matching and store the results in the state DB.

    :return: Number of the matched identities.
    """
    match_result = await _match_jira_identities(account, meta_ids, sdb, mdb, cache)
    if match_result is not None:
        matched_jira_size, github_size, jira_size, from_scratch = match_result
        if slack is not None and from_scratch and github_size > 0 and jira_size > 0:
            await slack.post_install(
                "matched_jira_identities.jinja2",
                account=account,
                github_users=github_size,
                jira_users=jira_size,
                matched=matched_jira_size,
            )
        return matched_jira_size
    return None


ID_MATCH_MIN_CONFIDENCE = 0.5
ALLOWED_USER_TYPES = ("atlassian", "on-prem")


async def _match_jira_identities(
    account: int,
    meta_ids: tuple[int, ...],
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> Optional[tuple[int, int, int, bool]]:
    log = logging.getLogger("%s.match_jira_identities" % metadata.__package__)
    if (jira_id := await get_jira_installation_or_none(account, sdb, mdb, cache)) is None:
        return None
    progress_row = await mdb.fetch_one(
        select([func.sum(Progress.current), func.sum(Progress.total)]).where(
            and_(
                Progress.acc_id == jira_id[0],
                Progress.is_initial,
                Progress.event_type.in_(["user", "user-fallback"]),
            ),
        ),
    )
    if (current := progress_row[0] or 0) < (total := progress_row[1] or 0):
        log.warning("JIRA fetch progress is not 100%%: %d < %d", current, total)
        return None
    existing_mapping = await sdb.fetch_all(
        select([MappedJIRAIdentity.github_user_id, MappedJIRAIdentity.jira_user_id]).where(
            MappedJIRAIdentity.account_id == account,
        ),
    )
    if not (
        github_users := (await load_organization_members(account, meta_ids, mdb, sdb, log, cache))[
            0
        ]
    ):
        return 0, 0, 0, len(existing_mapping) == 0
    if existing_mapping:
        for row in existing_mapping:
            try:
                del github_users[row[MappedJIRAIdentity.github_user_id.name]]
            except KeyError:
                continue
        log.info("Effective GitHub set size: %d", len(github_users))
    jira_user_rows = await mdb.fetch_all(
        select([JIRAUser.id, JIRAUser.display_name]).where(
            and_(
                JIRAUser.acc_id == jira_id[0],
                JIRAUser.type.in_(ALLOWED_USER_TYPES),
                JIRAUser.display_name.isnot(None),
            ),
        ),
    )
    jira_users = {
        row[JIRAUser.id.name]: [row[JIRAUser.display_name.name]] for row in jira_user_rows
    }
    log.info("JIRA set size: %d", len(jira_users))
    if existing_mapping:
        for row in existing_mapping:
            try:
                del jira_users[row[MappedJIRAIdentity.jira_user_id.name]]
            except KeyError:
                continue
        log.info("Effective JIRA set size: %d", len(jira_users))
    if not jira_users:
        return 0, len(github_users), 0, len(existing_mapping) == 0
    if not github_users:
        return 0, 0, len(jira_users), len(existing_mapping) == 0
    matches, confidences = NamesMatcher()(github_users.values(), jira_users.values())
    jira_users_keys = list(jira_users)
    db_records = []
    for github_user, match_index, confidence in zip(github_users, matches, confidences):
        if match_index >= 0 and confidence >= ID_MATCH_MIN_CONFIDENCE:
            jira_user_id = jira_users_keys[match_index]
            log.debug("%s -> %s", github_users[github_user], jira_users[jira_user_id][0])
            db_records.append(
                MappedJIRAIdentity(
                    account_id=account,
                    github_user_id=github_user,
                    jira_user_id=jira_user_id,
                    confidence=confidence,
                )
                .create_defaults()
                .explode(with_primary_keys=True),
            )
    if db_records:
        log.info("Storing %d matches", len(db_records))
        async with sdb.connection() as sdb_conn:
            async with sdb_conn.transaction():
                await sdb_conn.execute_many(insert(MappedJIRAIdentity), db_records)
        await load_jira_identity_mapping_sentinel.reset_cache(account, cache)
    return len(db_records), len(github_users), len(jira_users), len(existing_mapping) == 0


nonalphanum_re = re.compile(r"[^a-zA-Z0-9]+")
pluralizer = Pluralizer()


def normalize_issue_type(name: str) -> str:
    """Normalize the JIRA issue type name."""
    return pluralizer.singular(nonalphanum_re.sub("", unidecode(name.lower())))


def normalize_priority(name: str) -> str:
    """Normalize the JIRA priority name."""
    return name.lower()


def normalize_user_type(type_: str) -> str:
    """Normalize the JIRA user type name."""
    if type_ == "on-prem":
        return "atlassian"
    return type_


def parse_request_priorities(req_priorities: Optional[list[str]]) -> Optional[list[str]]:
    """Parse the raw Jira priorities received in a request."""
    if req_priorities is None:
        return None
    return sorted({normalize_priority(p) for p in req_priorities})


def parse_request_issue_types(req_issue_types: Optional[list[str]]) -> Optional[list[str]]:
    """Parse the raw Jira issue types received in a request."""
    if req_issue_types is None:
        return None
    return sorted({normalize_issue_type(t) for t in req_issue_types})


async def disable_empty_projects(
    account: int,
    meta_ids: tuple[int, ...],
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    slack: Optional[SlackWebClient],
    cache: Optional[aiomcache.Client],
) -> int:
    """
    Disable jira unused projects by creating entries in JIRAProjectSetting table.

    A project is disabled when:
    - doesn't contain issues with mapped PRs
    - it has issues and its first issue is not recent (more than a month ago)
    - it is not explicitly enabled in JIRAProjectSetting table

    """
    if (jira_id := await get_jira_installation_or_none(account, sdb, mdb, cache)) is None:
        return 0
    log = logging.getLogger("%s.disable_empty_projects" % metadata.__package__)
    mapped_projects, all_projects, settings = await gather(
        mdb.fetch_all(
            select(Issue.project_id)
            .select_from(
                join(
                    NodePullRequestJiraIssues,
                    Issue,
                    and_(
                        NodePullRequestJiraIssues.jira_acc == Issue.acc_id,
                        NodePullRequestJiraIssues.jira_id == Issue.id,
                    ),
                ),
            )
            .where(
                and_(
                    NodePullRequestJiraIssues.node_acc.in_(meta_ids),
                    NodePullRequestJiraIssues.jira_acc == jira_id[0],
                ),
            )
            .distinct(),
        ),
        mdb.fetch_all(
            select(Project.id, Project.key, func.min(Issue.created))
            .select_from(
                join(
                    Project,
                    Issue,
                    and_(Project.acc_id == Issue.acc_id, Project.id == Issue.project_id),
                    isouter=True,
                ),
            )
            .where(and_(Project.acc_id == jira_id[0], Project.is_deleted.is_(False)))
            .group_by(Project.id, Project.key),
        ),
        sdb.fetch_all(
            select(JIRAProjectSetting.key).where(JIRAProjectSetting.account_id == account),
        ),
    )
    mapped_projects = {r[0] for r in mapped_projects}
    settings = {r[0] for r in settings}
    one_month_ago = datetime.now(timezone.utc) - timedelta(days=30)
    if not await is_postgresql(mdb):
        one_month_ago = one_month_ago.replace(tzinfo=None)

    # multiple project_id can have the same project_key, but projects can be disabled in
    # JIRAProjectSetting by key and not by id; so a project key will be disabled only when
    # all the projects with that key are to be disabled
    keys_with_disabled_projs = set()
    keys_with_enabled_projs = set()
    for project_id, project_key, first_issue in all_projects:
        if (
            project_id not in mapped_projects
            and project_key not in settings
            and first_issue is not None
            and first_issue <= one_month_ago
        ):
            keys_with_disabled_projs.add(project_key)
        else:
            keys_with_enabled_projs.add(project_key)

    disabled = keys_with_disabled_projs - keys_with_enabled_projs

    project_setting_values = [
        JIRAProjectSetting(key=key, enabled=False, account_id=account)
        .create_defaults()
        .explode(with_primary_keys=True)
        for key in disabled
    ]
    await sdb.execute_many(insert(JIRAProjectSetting), project_setting_values)
    log.warning("Disabled JIRA projects on account %d: %s", account, sorted(disabled))
    if slack is not None and len(project_setting_values) > 0 and len(settings) == 0:
        await slack.post_install(
            "disabled_jira_projects.jinja2",
            account=account,
            disabled=len(disabled),
            total=len(all_projects),
        )
    return len(disabled)
