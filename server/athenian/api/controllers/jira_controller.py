import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import pickle
from typing import Any, Collection, Dict, List, Mapping, Optional, Set, Tuple, Type, Union

from aiohttp import web
import aiomcache
import numpy as np
import numpy.typing as npt
import pandas as pd
from sqlalchemy import and_, select, union_all
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_utils import gather, list_with_yield, read_sql_query
from athenian.api.balancing import weight
from athenian.api.cache import cached, expires_header, short_term_exptime
from athenian.api.controllers.filter_controller import web_pr_from_struct, webify_deployment
from athenian.api.db import Database
from athenian.api.internal.account import get_account_repositories, get_metadata_account_ids
from athenian.api.internal.datetime_utils import split_to_time_intervals
from athenian.api.internal.features.entries import UnsupportedMetricError, make_calculator
from athenian.api.internal.features.github.pull_request_filter import (
    PullRequestListMiner,
    unwrap_pull_requests,
)
from athenian.api.internal.features.histogram import HistogramParameters, Scale
from athenian.api.internal.jira import (
    JIRAConfig,
    get_jira_installation,
    load_mapped_jira_users,
    normalize_issue_type,
    normalize_user_type,
    resolve_projects,
)
from athenian.api.internal.logical_repos import drop_logical_repo
from athenian.api.internal.miners.filters import LabelFilter
from athenian.api.internal.miners.github.bots import bots
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.deployment_light import fetch_repository_environments
from athenian.api.internal.miners.github.precomputed_prs import DonePRFactsLoader
from athenian.api.internal.miners.jira.epic import filter_epics
from athenian.api.internal.miners.jira.issue import (
    ISSUE_PR_IDS,
    ISSUE_PRS_BEGAN,
    ISSUE_PRS_COUNT,
    ISSUE_PRS_RELEASED,
    fetch_jira_issues,
    participant_columns,
    resolve_work_began_and_resolved,
)
from athenian.api.internal.miners.types import Deployment
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings, Settings
from athenian.api.internal.with_ import fetch_teams_map
from athenian.api.models.metadata.github import Branch, PullRequest
from athenian.api.models.metadata.jira import (
    AthenianIssue,
    Component,
    Issue,
    IssueType,
    Priority,
    Status,
    User,
)
from athenian.api.models.state.models import MappedJIRAIdentity
from athenian.api.models.web import (
    CalculatedJIRAHistogram,
    CalculatedJIRAMetricValues,
    CalculatedLinearMetricValues,
    DeploymentNotification as WebDeploymentNotification,
    FilteredJIRAStuff,
    FilterJIRAStuff,
    Interquartile,
    InvalidRequestError,
    JIRAEpic,
    JIRAEpicChild,
    JIRAFilterReturn,
    JIRAFilterWith,
    JIRAHistogramsRequest,
    JIRAIssue,
    JIRAIssueType,
    JIRALabel,
    JIRAMetricsRequest,
    JIRAPriority,
    JIRAStatus,
    JIRAUser,
    PullRequest as WebPullRequest,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response
from athenian.api.tracing import sentry_span
from athenian.api.unordered_unique import unordered_unique


@expires_header(short_term_exptime)
@weight(2.0)
async def filter_jira_stuff(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find JIRA epics and labels used in the specified date interval."""
    try:
        filt = FilterJIRAStuff.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError(getattr(e, "path", "?"), detail=str(e)))
    (
        meta_ids,
        jira_ids,
        branches,
        default_branches,
        release_settings,
        logical_settings,
        prefixer,
    ) = await _collect_ids(filt.account, request, request.sdb, request.mdb, request.cache)
    if filt.projects is not None:
        projects = await resolve_projects(filt.projects, jira_ids.acc_id, request.mdb)
        projects = {k: projects[k] for k in jira_ids.projects if k in projects}
        jira_ids = JIRAConfig(jira_ids.acc_id, projects, jira_ids.epics)
    if filt.date_from is None or filt.date_to is None:
        if (filt.date_from is None) != (filt.date_to is None):
            raise ResponseError(
                InvalidRequestError(
                    ".date_from",
                    detail="date_from and date_to must be either both not null or both null",
                ),
            )
        time_from = time_to = None
    else:
        time_from, time_to = filt.resolve_time_from_and_to()
    label_filter = LabelFilter.from_iterables(filt.labels_include, filt.labels_exclude)
    if filt.with_ is not None:
        reporters = [p.lower() for p in (filt.with_.reporters or [])]
        assignees = [(p.lower() if p is not None else None) for p in (filt.with_.assignees or [])]
        commenters = [p.lower() for p in (filt.with_.commenters or [])]
    else:
        reporters = assignees = commenters = []
    filt.priorities = [p.lower() for p in (filt.priorities or [])]
    filt.types = {normalize_issue_type(p) for p in (filt.types or [])}
    return_ = set(filt.return_ or JIRAFilterReturn)
    if not filt.return_:
        return_.remove(JIRAFilterReturn.ONLY_FLYING)
    sdb, mdb, pdb, rdb = request.sdb, request.mdb, request.pdb, request.rdb
    cache = request.cache
    tasks = [
        _epic_flow(
            return_,
            jira_ids,
            time_from,
            time_to,
            filt.exclude_inactive,
            label_filter,
            filt.priorities,
            reporters,
            assignees,
            commenters,
            default_branches,
            release_settings,
            logical_settings,
            filt.account,
            meta_ids,
            mdb,
            pdb,
            cache,
        ),
        _issue_flow(
            return_,
            filt.account,
            jira_ids,
            time_from,
            time_to,
            filt.exclude_inactive,
            label_filter,
            filt.priorities,
            filt.types,
            reporters,
            assignees,
            commenters,
            branches,
            default_branches,
            release_settings,
            logical_settings,
            prefixer,
            meta_ids,
            sdb,
            mdb,
            pdb,
            rdb,
            cache,
        ),
    ]
    (
        (epics, epic_priorities, epic_statuses),
        (issues, labels, issue_users, issue_types, issue_priorities, issue_statuses, deps),
    ) = await gather(*tasks, op="forked flows")
    if epic_priorities is None:
        priorities = issue_priorities
    elif issue_priorities is None:
        priorities = epic_priorities
    else:
        priorities = sorted(set(epic_priorities).union(issue_priorities))
    if epic_statuses is None:
        statuses = issue_statuses
    elif issue_statuses is None:
        statuses = epic_statuses
    else:
        statuses = sorted(set(epic_statuses).union(issue_statuses))
    return model_response(
        FilteredJIRAStuff(
            epics=epics,
            issues=issues,
            labels=labels,
            issue_types=issue_types,
            priorities=priorities,
            users=issue_users,
            statuses=statuses,
            deployments=deps,
        ),
    )


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda return_, time_from, time_to, exclude_inactive, label_filter, priorities, reporters, assignees, commenters, default_branches, release_settings, logical_settings, **_: (  # noqa
        JIRAFilterReturn.EPICS in return_,
        JIRAFilterReturn.PRIORITIES in return_,
        JIRAFilterReturn.STATUSES in return_,
        time_from.timestamp() if time_from else "-",
        time_to.timestamp() if time_to else "-",
        exclude_inactive,
        label_filter,
        ",".join(sorted(priorities)),
        ",".join(sorted(reporters)),
        ",".join(sorted((ass if ass is not None else "<None>") for ass in assignees)),
        ",".join(sorted(commenters)),
        ",".join("%s:%s" % db for db in sorted(default_branches.items())),
        release_settings,
        logical_settings,
    ),
)
async def _epic_flow(
    return_: Set[str],
    jira_ids: JIRAConfig,
    time_from: Optional[datetime],
    time_to: Optional[datetime],
    exclude_inactive: bool,
    label_filter: LabelFilter,
    priorities: Collection[str],
    reporters: Collection[str],
    assignees: Collection[Optional[str]],
    commenters: Collection[str],
    default_branches: Dict[str, str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    account: int,
    meta_ids: Tuple[int, ...],
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> Tuple[Optional[List[JIRAEpic]], Optional[List[JIRAPriority]], Optional[List[JIRAStatus]]]:
    """Fetch various information related to JIRA epics."""
    if JIRAFilterReturn.EPICS not in return_:
        return None, None, None
    log = logging.getLogger("%s.filter_jira_stuff/epic" % metadata.__package__)
    extra_columns = [
        Issue.key,
        Issue.title,
        Issue.reporter_display_name,
        Issue.assignee_display_name,
        Issue.project_id,
        Issue.components,
        Issue.assignee_id,
        Issue.reporter_id,
        Issue.commenters_ids,
        Issue.priority_id,
        Issue.status_id,
        Issue.type_id,
        Issue.comments_count,
        Issue.url,
    ]
    if JIRAFilterReturn.USERS in return_:
        extra_columns.extend(participant_columns)
    epics_df, children_df, subtask_counts, epic_children_map = await filter_epics(
        jira_ids,
        time_from,
        time_to,
        exclude_inactive,
        label_filter,
        priorities,
        reporters,
        assignees,
        commenters,
        default_branches,
        release_settings,
        logical_settings,
        account,
        meta_ids,
        mdb,
        pdb,
        cache,
        extra_columns=extra_columns,
    )
    children_columns = {k: children_df[k].values for k in children_df.columns}
    children_columns[Issue.id.name] = children_df.index.values
    epics = []
    issue_by_id = {}
    issue_type_ids = {}
    children_by_type = defaultdict(list)
    epics_by_type = defaultdict(list)
    now = datetime.utcnow()
    for (
        epic_id,
        project_id,
        epic_key,
        epic_title,
        epic_created,
        epic_updated,
        epic_prs_began,
        epic_work_began,
        epic_prs_released,
        epic_resolved,
        epic_reporter,
        epic_assignee,
        epic_priority,
        epic_status,
        epic_type,
        epic_prs,
        epic_comments,
        epic_url,
    ) in zip(
        epics_df.index.values,
        *(
            epics_df[column].values
            for column in (
                Issue.project_id.name,
                Issue.key.name,
                Issue.title.name,
                Issue.created.name,
                AthenianIssue.updated.name,
                ISSUE_PRS_BEGAN,
                AthenianIssue.work_began.name,
                ISSUE_PRS_RELEASED,
                AthenianIssue.resolved.name,
                Issue.reporter_display_name.name,
                Issue.assignee_display_name.name,
                Issue.priority_name.name,
                Issue.status.name,
                Issue.type_id.name,
                ISSUE_PRS_COUNT,
                Issue.comments_count.name,
                Issue.url.name,
            )
        ),
    ):
        work_began, resolved = resolve_work_began_and_resolved(
            epic_work_began, epic_prs_began, epic_resolved, epic_prs_released,
        )
        epics.append(
            epic := JIRAEpic(
                id=epic_key,
                project=project_id.decode(),
                children=[],
                title=epic_title,
                created=epic_created,
                updated=epic_updated,
                work_began=work_began,
                resolved=resolved,
                reporter=epic_reporter,
                assignee=epic_assignee,
                comments=epic_comments,
                priority=epic_priority,
                status=epic_status,
                type=epic_type,
                prs=epic_prs,
                url=epic_url,
            ),
        )
        epics_by_type[(project_id, epic_type)].append(epic)
        children_indexes = epic_children_map.get(epic_id, [])
        project_type_ids = issue_type_ids.setdefault(project_id, set())
        project_type_ids.add(epic_type)
        for (
            child_id,
            child_key,
            child_title,
            child_created,
            child_updated,
            child_prs_began,
            child_work_began,
            child_prs_released,
            child_resolved,
            child_comments,
            child_reporter,
            child_assignee,
            child_priority,
            child_status,
            child_prs,
            child_type,
            child_url,
        ) in zip(
            *(
                children_columns[column][children_indexes]
                for column in (
                    Issue.id.name,
                    Issue.key.name,
                    Issue.title.name,
                    Issue.created.name,
                    AthenianIssue.updated.name,
                    ISSUE_PRS_BEGAN,
                    AthenianIssue.work_began.name,
                    ISSUE_PRS_RELEASED,
                    AthenianIssue.resolved.name,
                    Issue.comments_count.name,
                    Issue.reporter_display_name.name,
                    Issue.assignee_display_name.name,
                    Issue.priority_name.name,
                    Issue.status.name,
                    ISSUE_PRS_COUNT,
                    Issue.type_id.name,
                    Issue.url.name,
                )
            ),
        ):  # noqa(E123)
            epic.prs += child_prs
            work_began, resolved = resolve_work_began_and_resolved(
                child_work_began, child_prs_began, child_resolved, child_prs_released,
            )
            if work_began is not None:
                epic.work_began = min(epic.work_began or work_began, work_began)
            if resolved is not None:
                lead_time = resolved - work_began
                life_time = resolved - child_created
            else:
                life_time = now - pd.to_datetime(child_created)
                if work_began is not None:
                    lead_time = now - pd.to_datetime(work_began)
                else:
                    lead_time = None
            if resolved is None:
                epic.resolved = None
            project_type_ids.add(child_type)
            epic.children.append(
                child := JIRAEpicChild(
                    id=child_key,
                    title=child_title,
                    created=child_created,
                    updated=child_updated,
                    work_began=work_began,
                    lead_time=lead_time,
                    life_time=life_time,
                    resolved=resolved,
                    reporter=child_reporter,
                    assignee=child_assignee,
                    comments=child_comments,
                    priority=child_priority,
                    status=child_status,
                    prs=child_prs,
                    type=child_type,
                    subtasks=0,
                    url=child_url,
                ),
            )
            issue_by_id[child_id] = child
            children_by_type[(project_id, child_type)].append(child)
            if len(issue_by_id) % 200 == 0:
                await asyncio.sleep(0)
        if epic.resolved is not None:
            epic.lead_time = epic.resolved - epic.work_began
            epic.life_time = epic.resolved - epic.created
        else:
            epic.life_time = now - pd.to_datetime(epic.created)
            if epic.work_began is not None:
                epic.lead_time = now - pd.to_datetime(epic.work_began)
    if JIRAFilterReturn.PRIORITIES in return_:
        priority_ids = unordered_unique(
            np.concatenate(
                [
                    epics_df[Issue.priority_id.name].values,
                    children_columns[Issue.priority_id.name],
                ],
            ),
        )
    else:
        priority_ids = np.array([], dtype="S")
    if JIRAFilterReturn.STATUSES in return_:
        # status IDs are account-wide unique
        status_ids = unordered_unique(
            np.concatenate(
                [
                    epics_df[Issue.status_id.name].values,
                    children_columns[Issue.status_id.name],
                ],
            ),
        )
        status_project_map = defaultdict(set)
        for status_id, project_id in chain(
            zip(
                epics_df[Issue.status_id.name].values,
                epics_df[Issue.project_id.name].values,
            ),
            zip(
                children_columns[Issue.status_id.name],
                children_columns[Issue.project_id.name],
            ),
        ):
            status_project_map[status_id].add(project_id)
    else:
        status_ids = np.array([], dtype="S")
        status_project_map = {}
    priorities, statuses, types = await gather(
        _fetch_priorities(priority_ids, jira_ids[0], return_, mdb),
        _fetch_statuses(status_ids, status_project_map, jira_ids[0], return_, mdb),
        _fetch_types(
            issue_type_ids,
            jira_ids[0],
            return_,
            mdb,
            columns=[IssueType.id, IssueType.project_id, IssueType.name],
        ),
        op="epic epilog",
    )
    invalid_parent_id = []
    for issue_id, count in subtask_counts:
        try:
            issue_by_id[issue_id].subtasks = count
        except KeyError:
            invalid_parent_id.append(issue_id)
    if invalid_parent_id:
        log.error("issues are parents of children outside of the epics: %s", invalid_parent_id)
    for row in types:
        name = row[IssueType.name.name]
        key = row[IssueType.project_id.name].encode(), row[IssueType.id.name].encode()
        for epic in epics_by_type.get(key, []):
            epic.type = name
        for child in children_by_type.get(key, []):
            child.type = name
    return epics, priorities, statuses


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda return_, time_from, time_to, exclude_inactive, label_filter, priorities, reporters, assignees, commenters, default_branches, release_settings, logical_settings, **_: (  # noqa
        JIRAFilterReturn.ISSUES in return_,
        JIRAFilterReturn.ISSUE_BODIES in return_,
        JIRAFilterReturn.LABELS in return_,
        JIRAFilterReturn.USERS in return_,
        JIRAFilterReturn.ISSUE_TYPES in return_,
        JIRAFilterReturn.PRIORITIES in return_,
        JIRAFilterReturn.STATUSES in return_,
        JIRAFilterReturn.ONLY_FLYING in return_,
        time_from.timestamp() if time_from else "-",
        time_to.timestamp() if time_to else "-",
        exclude_inactive,
        label_filter,
        ",".join(sorted(priorities)),
        ",".join(sorted(reporters)),
        ",".join(sorted((ass if ass is not None else "<None>") for ass in assignees)),
        ",".join(sorted(commenters)),
        ",".join("%s:%s" % db for db in sorted(default_branches.items())),
        release_settings,
        logical_settings,
    ),
)
async def _issue_flow(
    return_: Set[str],
    account: int,
    jira_ids: JIRAConfig,
    time_from: Optional[datetime],
    time_to: Optional[datetime],
    exclude_inactive: bool,
    label_filter: LabelFilter,
    priorities: Collection[str],
    types: Collection[str],
    reporters: Collection[str],
    assignees: Collection[Optional[str]],
    commenters: Collection[str],
    branches: pd.DataFrame,
    default_branches: Dict[str, str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    meta_ids: Tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> Tuple[
    Optional[JIRAIssue],
    Optional[JIRALabel],
    Optional[JIRAUser],
    Optional[JIRAIssueType],
    Optional[JIRAPriority],
    Optional[JIRAStatus],
    Optional[Dict[str, WebDeploymentNotification]],
]:
    """Fetch various information related to JIRA issues."""
    if JIRAFilterReturn.ISSUES not in return_:
        return (None,) * 7
    log = logging.getLogger("%s.filter_jira_stuff/issue" % metadata.__package__)
    extra_columns = [
        Issue.project_id,
        Issue.components,
        Issue.assignee_id,
        Issue.reporter_id,
        Issue.commenters_ids,
        Issue.priority_id,
        Issue.status_id,
        Issue.type_id,
        Issue.comments_count,
    ]
    if JIRAFilterReturn.ISSUE_BODIES in return_:
        extra_columns.extend(
            [
                Issue.key,
                Issue.title,
                Issue.reporter_display_name,
                Issue.assignee_display_name,
                Issue.url,
            ],
        )
    if JIRAFilterReturn.USERS in return_:
        extra_columns.extend(participant_columns)
    epics = [] if JIRAFilterReturn.ONLY_FLYING not in return_ else False
    issues = await fetch_jira_issues(
        jira_ids,
        time_from,
        time_to,
        exclude_inactive,
        label_filter,
        # priorities are already lower-cased and de-None-d
        priorities,
        types,
        epics,
        reporters,
        assignees,
        commenters,
        False,
        default_branches,
        release_settings,
        logical_settings,
        account,
        meta_ids,
        mdb,
        pdb,
        cache,
        extra_columns=extra_columns,
    )
    if JIRAFilterReturn.LABELS in return_:
        components = Counter(chain.from_iterable(_nonzero(issues[Issue.components.name].values)))
    else:
        components = None
    if JIRAFilterReturn.USERS in return_:
        people = np.unique(
            np.concatenate(
                [
                    _nonzero(issues[Issue.reporter_id.name].values),
                    _nonzero(issues[Issue.assignee_id.name].values),
                    list(chain.from_iterable(_nonzero(issues[Issue.commenters_ids.name].values))),
                ],
            ),
        )
        # we can leave None because `IN (null)` is always false
    else:
        people = None
    if JIRAFilterReturn.PRIORITIES in return_:
        priorities = issues[Issue.priority_id.name].unique()
    else:
        priorities = []
    if JIRAFilterReturn.STATUSES in return_:
        statuses = issues[Issue.status_id.name].unique()
        status_project_map = defaultdict(set)
        for status_id, project_id in zip(
            issues[Issue.status_id.name].values,
            issues[Issue.project_id.name].values,
        ):
            status_project_map[status_id].add(project_id)
    else:
        statuses = []
        status_project_map = {}
    if JIRAFilterReturn.ISSUE_TYPES in return_ or JIRAFilterReturn.ISSUE_BODIES in return_:
        issue_type_counts = defaultdict(int)
        issue_type_projects = defaultdict(set)
        for project_id, issue_type_id in zip(
            issues[Issue.project_id.name].values,
            issues[Issue.type_id.name].values,
        ):
            issue_type_projects[project_id].add(issue_type_id)
            issue_type_counts[(project_id, issue_type_id)] += 1
    else:
        issue_type_counts = issue_type_projects = None
    if JIRAFilterReturn.ISSUE_BODIES in return_:
        if not issues.empty:
            pr_ids = np.concatenate(issues[ISSUE_PR_IDS].values, dtype=int, casting="unsafe")
        else:
            pr_ids = []
    else:
        pr_ids = None

    @sentry_span
    async def fetch_components():
        if JIRAFilterReturn.LABELS not in return_:
            return []
        return await mdb.fetch_all(
            select([Component.id, Component.name]).where(
                and_(
                    Component.id.in_(components),
                    Component.acc_id == jira_ids[0],
                ),
            ),
        )

    @sentry_span
    async def fetch_users():
        if JIRAFilterReturn.USERS not in return_:
            return []
        return await mdb.fetch_all(
            select([User.display_name, User.avatar_url, User.type, User.id])
            .where(
                and_(
                    User.id.in_(people),
                    User.acc_id == jira_ids[0],
                ),
            )
            .order_by(User.display_name),
        )

    @sentry_span
    async def fetch_mapped_identities():
        if JIRAFilterReturn.USERS not in return_:
            return []
        return await sdb.fetch_all(
            select([MappedJIRAIdentity.github_user_id, MappedJIRAIdentity.jira_user_id]).where(
                and_(
                    MappedJIRAIdentity.account_id == account,
                    MappedJIRAIdentity.jira_user_id.in_(people),
                ),
            ),
        )

    @sentry_span
    async def extract_labels():
        if JIRAFilterReturn.LABELS in return_:
            labels_column = issues[Issue.labels.name].values
            if label_filter:
                labels_column = (ils for ils in labels_column if label_filter.match(ils))
            labels = Counter(chain.from_iterable(labels_column))
            if None in labels:
                del labels[None]
            labels = {
                k: JIRALabel(title=k, kind="regular", issues_count=v) for k, v in labels.items()
            }
            for updated, issue_labels in zip(
                issues[Issue.updated.name],
                issues[Issue.labels.name].values,
            ):
                for label in issue_labels or ():
                    try:
                        label = labels[label]  # type: JIRALabel
                    except KeyError:
                        continue
                    if label.last_used is None or label.last_used < updated:
                        label.last_used = updated
            if mdb.url.dialect == "sqlite":
                for label in labels.values():
                    label.last_used = label.last_used.replace(tzinfo=timezone.utc)
        else:
            labels = None
        return labels

    @sentry_span
    async def _fetch_prs() -> Tuple[
        Optional[Dict[str, WebPullRequest]],
        Optional[Dict[str, Deployment]],
    ]:
        if JIRAFilterReturn.ISSUE_BODIES not in return_:
            return None, None
        if len(pr_ids) == 0:
            return {}, {}
        prs_df, (facts, ambiguous), account_bots = await gather(
            read_sql_query(
                select([PullRequest])
                .where(
                    and_(
                        PullRequest.acc_id.in_(meta_ids),
                        PullRequest.node_id.in_(pr_ids),
                    ),
                )
                .order_by(PullRequest.node_id.name),
                mdb,
                PullRequest,
                index=PullRequest.node_id.name,
            ),
            DonePRFactsLoader.load_precomputed_done_facts_ids(
                pr_ids,
                default_branches,
                release_settings,
                prefixer,
                account,
                pdb,
                panic_on_missing_repositories=False,
            ),
            bots(account, meta_ids, mdb, sdb, cache),
        )
        existing_mask = (
            prs_df[PullRequest.repository_full_name.name].isin(release_settings.native).values
        )
        if not existing_mask.all():
            prs_df = prs_df.take(np.flatnonzero(existing_mask))
        found_repos = set(prs_df[PullRequest.repository_full_name.name].unique())
        if ambiguous.keys() - found_repos:
            # there are archived or disabled repos
            ambiguous = {k: v for k, v in ambiguous.items() if k in found_repos}
        related_branches = branches.take(
            np.flatnonzero(
                np.in1d(
                    branches[Branch.repository_full_name.name].values.astype("S"),
                    prs_df[PullRequest.repository_full_name.name].unique().astype("S"),
                ),
            ),
        )
        (mined_prs, dfs, facts, _, deployments_task), repo_envs = await gather(
            unwrap_pull_requests(
                prs_df,
                facts,
                ambiguous,
                False,
                related_branches,
                default_branches,
                account_bots,
                release_settings,
                logical_settings,
                prefixer,
                account,
                meta_ids,
                mdb,
                pdb,
                rdb,
                cache,
            ),
            fetch_repository_environments(
                prs_df[PullRequest.repository_full_name.name].unique(),
                prefixer,
                account,
                rdb,
                cache,
            ),
        )

        miner = PullRequestListMiner(
            mined_prs,
            dfs,
            facts,
            set(),
            set(),
            datetime(1970, 1, 1, tzinfo=timezone.utc),
            datetime.now(timezone.utc),
            False,
            None,
            repo_envs,
        )
        pr_list_items = await list_with_yield(miner, "PullRequestListMiner.__iter__")
        if missing_repo_indexes := [
            i
            for i, pr in enumerate(pr_list_items)
            if drop_logical_repo(pr.repository) not in prefixer.repo_name_to_prefixed_name
        ]:
            log.error(
                "Discarded %d PRs because their repositories are gone: %s",
                len(missing_repo_indexes),
                {pr_list_items[i].repository for i in missing_repo_indexes},
            )
            for i in reversed(missing_repo_indexes):
                pr_list_items.pop(i)
        if deployments_task is not None:
            await deployments_task
            deployments = deployments_task.result()
        else:
            deployments = None
        prs = dict(web_pr_from_struct(pr_list_items, prefixer, log, lambda w, pr: (pr.node_id, w)))
        return prs, deployments

    (
        (prs, deps),
        component_names,
        users,
        mapped_identities,
        priorities,
        statuses,
        issue_types,
        labels,
    ) = await gather(
        _fetch_prs(),
        fetch_components(),
        fetch_users(),
        fetch_mapped_identities(),
        _fetch_priorities(priorities, jira_ids[0], return_, mdb),
        _fetch_statuses(statuses, status_project_map, jira_ids[0], return_, mdb),
        _fetch_types(issue_type_projects, jira_ids[0], return_, mdb),
        extract_labels(),
    )
    components = {
        row[0]: JIRALabel(title=row[1], kind="component", issues_count=components[row[0]])
        for row in component_names
    }
    mapped_identities = {
        r[MappedJIRAIdentity.jira_user_id.name]: r[MappedJIRAIdentity.github_user_id.name]
        for r in mapped_identities
    }
    users = [
        JIRAUser(
            avatar=row[User.avatar_url.name],
            name=row[User.display_name.name],
            type=normalize_user_type(row[User.type.name]),
            developer=mapped_identities.get(row[User.id.name]),
        )
        for row in users
    ] or None
    if deps is not None:
        prefix_logical_repo = prefixer.prefix_logical_repo
        deps = {
            key: webify_deployment(val, prefix_logical_repo) for key, val in sorted(deps.items())
        }
    if issue_types is not None:
        issue_types.sort(
            key=lambda row: (
                row[IssueType.normalized_name.name],
                issue_type_counts[
                    (row[IssueType.project_id.name].encode(), row[IssueType.id.name].encode())
                ],
            ),
        )
    else:
        issue_types = []
    # fmt: off
    issue_type_names = {
        (row[IssueType.project_id.name].encode(), row[IssueType.id.name].encode()):
            row[IssueType.name.name]
        for row in issue_types
    }
    # fmt: on
    issue_types = [
        JIRAIssueType(
            name=row[IssueType.name.name],
            image=row[IssueType.icon_url.name],
            count=issue_type_counts[
                (row[IssueType.project_id.name].encode(), row[IssueType.id.name].encode())
            ],
            project=row[IssueType.project_id.name],
            is_subtask=row[IssueType.is_subtask.name],
            is_epic=row[IssueType.is_epic.name],
            normalized_name=row[IssueType.normalized_name.name],
        )
        for row in issue_types
    ] or None
    if JIRAFilterReturn.LABELS in return_:
        for updated, issue_components in zip(
            issues[Issue.updated.name],
            issues[Issue.components.name].values,
        ):
            for component in issue_components or ():
                try:
                    label = components[component]  # type: JIRALabel
                except KeyError:
                    log.error("Missing JIRA component: %s" % component)
                    continue
                if label.last_used is None or label.last_used < updated:
                    label.last_used = updated
        if mdb.url.dialect == "sqlite":
            for label in components.values():
                label.last_used = label.last_used.replace(tzinfo=timezone.utc)

        labels = sorted(chain(components.values(), labels.values()))
    if JIRAFilterReturn.ISSUE_BODIES in return_:
        issue_models = []
        now = datetime.utcnow()
        for (
            issue_key,
            issue_title,
            issue_created,
            issue_updated,
            issue_prs_began,
            issue_work_began,
            issue_prs_released,
            issue_resolved,
            issue_reporter,
            issue_assignee,
            issue_priority,
            issue_status,
            issue_prs,
            issue_type,
            issue_project,
            issue_comments,
            issue_url,
        ) in zip(
            *(
                issues[column].values
                for column in (
                    Issue.key.name,
                    Issue.title.name,
                    Issue.created.name,
                    AthenianIssue.updated.name,
                    ISSUE_PRS_BEGAN,
                    AthenianIssue.work_began.name,
                    ISSUE_PRS_RELEASED,
                    AthenianIssue.resolved.name,
                    Issue.reporter_display_name.name,
                    Issue.assignee_display_name.name,
                    Issue.priority_name.name,
                    Issue.status.name,
                    ISSUE_PR_IDS,
                    Issue.type_id.name,
                    Issue.project_id.name,
                    Issue.comments_count.name,
                    Issue.url.name,
                )
            ),
        ):
            work_began, resolved = resolve_work_began_and_resolved(
                issue_work_began, issue_prs_began, issue_resolved, issue_prs_released,
            )
            if resolved:
                lead_time = resolved - work_began
                life_time = resolved - issue_created
            else:
                life_time = now - pd.to_datetime(issue_created)
                if work_began:
                    lead_time = now - pd.to_datetime(work_began)
                else:
                    lead_time = None
            issue_models.append(
                JIRAIssue(
                    id=issue_key,
                    title=issue_title,
                    created=issue_created,
                    updated=issue_updated,
                    work_began=work_began,
                    resolved=resolved,
                    lead_time=lead_time,
                    life_time=life_time,
                    reporter=issue_reporter,
                    assignee=issue_assignee,
                    comments=issue_comments,
                    priority=issue_priority,
                    status=issue_status,
                    project=issue_project.decode(),
                    type=issue_type_names[(issue_project, issue_type)],
                    prs=[prs[node_id] for node_id in issue_prs if node_id in prs],
                    url=issue_url,
                ),
            )
    else:
        issue_models = None
    return issue_models, labels, users, issue_types, priorities, statuses, deps


@sentry_span
async def _fetch_priorities(
    priorities: npt.NDArray[bytes],
    acc_id: int,
    return_: Set[str],
    mdb: Database,
) -> Optional[List[JIRAPriority]]:
    if JIRAFilterReturn.PRIORITIES not in return_:
        return None
    if len(priorities) == 0:
        return []
    rows = await mdb.fetch_all(
        select([Priority.name, Priority.icon_url, Priority.rank, Priority.status_color])
        .where(
            and_(
                Priority.id.in_(priorities),
                Priority.acc_id == acc_id,
            ),
        )
        .order_by(Priority.rank),
    )
    return [
        JIRAPriority(
            name=row[Priority.name.name],
            image=row[Priority.icon_url.name],
            rank=row[Priority.rank.name],
            color=row[Priority.status_color.name],
        )
        for row in rows
    ]


@sentry_span
async def _fetch_statuses(
    statuses: npt.NDArray[bytes],
    status_project_map: Dict[bytes, Set[bytes]],
    acc_id: int,
    return_: Set[str],
    mdb: Database,
) -> Optional[List[JIRAStatus]]:
    if JIRAFilterReturn.STATUSES not in return_:
        return None
    if len(statuses) == 0:
        return []
    rows = await mdb.fetch_all(
        select([Status.id, Status.name, Status.category_name])
        .where(
            and_(
                Status.id.in_(statuses),
                Status.acc_id == acc_id,
            ),
        )
        .order_by(Status.name),
    )
    # status IDs are account-wide unique
    return [
        JIRAStatus(
            name=row[Status.name.name],
            stage=row[Status.category_name.name],
            project=project.decode(),
        )
        for row in rows
        for project in status_project_map[row[Status.id.name].encode()]
    ]


@sentry_span
async def _fetch_types(
    issue_type_projects: Mapping[bytes, set[bytes]],
    acc_id: int,
    return_: Set[str],
    mdb: Database,
    columns: Optional[List[InstrumentedAttribute]] = None,
) -> Optional[List[Mapping[str, Any]]]:
    if (
        JIRAFilterReturn.ISSUE_TYPES not in return_
        and JIRAFilterReturn.ISSUE_BODIES not in return_
        and JIRAFilterReturn.EPICS not in return_
    ):
        return None
    if len(issue_type_projects) == 0:
        return []
    if columns is None:
        columns = [
            IssueType.name,
            IssueType.normalized_name,
            IssueType.id,
            IssueType.project_id,
            IssueType.icon_url,
            IssueType.is_subtask,
            IssueType.is_epic,
        ]
    queries = [
        select(columns).where(
            and_(
                IssueType.id.in_(np.fromiter(ids, "S8", len(ids))),
                IssueType.acc_id == acc_id,
                IssueType.project_id == project_id.decode(),
            ),
        )
        for project_id, ids in issue_type_projects.items()
    ]
    return await mdb.fetch_all(union_all(*queries))


def _nonzero(arr: np.ndarray) -> np.ndarray:
    return arr[arr.nonzero()[0]]


async def _collect_ids(
    account: int,
    request: AthenianWebRequest,
    sdb: Database,
    mdb: Database,
    cache: Optional[aiomcache.Client],
) -> Tuple[
    Tuple[int, ...],
    JIRAConfig,
    pd.DataFrame,
    Dict[str, str],
    ReleaseSettings,
    LogicalRepositorySettings,
    Prefixer,
]:
    repos, jira_ids, meta_ids = await gather(
        get_account_repositories(account, True, sdb),
        get_jira_installation(account, sdb, mdb, cache),
        get_metadata_account_ids(account, sdb, cache),
        op="sdb/ids",
    )
    settings = Settings.from_request(request, account)
    prefixer = await Prefixer.load(meta_ids, mdb, cache)
    (branches, default_branches), logical_settings = await gather(
        BranchMiner.extract_branches(repos, prefixer, meta_ids, mdb, cache, strip=True),
        settings.list_logical_repositories(prefixer, repos),
        op="sdb/branches and releases",
    )
    repos = logical_settings.append_logical_prs(repos)
    release_settings = await settings.list_release_matches(repos)
    return (
        meta_ids,
        jira_ids,
        branches,
        default_branches,
        release_settings,
        logical_settings,
        prefixer,
    )


async def _calc_jira_entry(
    request: AthenianWebRequest,
    body: dict,
    model: Union[Type[JIRAMetricsRequest], Type[JIRAHistogramsRequest]],
) -> Union[
    Tuple[
        JIRAMetricsRequest,
        List[List[datetime]],
        timedelta,
        np.ndarray,
        np.ndarray,
    ],
    Tuple[
        JIRAHistogramsRequest,
        Dict[HistogramParameters, List[str]],
        np.ndarray,
    ],
]:
    try:
        filt = model.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError(getattr(e, "path", "?"), detail=str(e))) from None
    (
        meta_ids,
        jira_ids,
        _,
        default_branches,
        release_settings,
        logical_settings,
        _,
    ) = await _collect_ids(filt.account, request, request.sdb, request.mdb, request.cache)
    if filt.projects is not None:
        projects = await resolve_projects(filt.projects, jira_ids.acc_id, request.mdb)
        projects = {k: projects[k] for k in jira_ids.projects if k in projects}
        jira_ids = JIRAConfig(jira_ids.acc_id, projects, jira_ids.epics)
    time_intervals, tzoffset = split_to_time_intervals(
        filt.date_from, filt.date_to, getattr(filt, "granularities", ["all"]), filt.timezone,
    )
    with_ = await _dereference_teams(
        filt.with_, filt.account, request.sdb, request.mdb, request.cache,
    )
    label_filter = LabelFilter.from_iterables(filt.labels_include, filt.labels_exclude)
    calculator = make_calculator(
        filt.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache,
    )
    if issubclass(model, JIRAMetricsRequest):
        metric_values, split_labels = await calculator.calc_jira_metrics_line_github(
            filt.metrics,
            time_intervals,
            filt.quantiles or (0, 1),
            [g.as_participants() for g in (with_ or [])],
            label_filter,
            filt.group_by_jira_label,
            [p.lower() for p in (filt.priorities or [])],
            {normalize_issue_type(p) for p in (filt.types or [])},
            filt.epics or [],
            filt.exclude_inactive,
            release_settings,
            logical_settings,
            default_branches,
            jira_ids,
        )
        return filt, time_intervals, tzoffset, metric_values, split_labels
    defs = defaultdict(list)
    for h in filt.histograms or []:
        defs[
            HistogramParameters(
                scale=Scale[h.scale.upper()] if h.scale is not None else None,
                bins=h.bins,
                ticks=tuple(h.ticks) if h.ticks is not None else None,
            )
        ].append(h.metric)
    try:
        histograms = await calculator.calc_jira_histograms(
            defs,
            time_intervals[0][0],
            time_intervals[0][1],
            filt.quantiles or (0, 1),
            [g.as_participants() for g in (with_ or [])],
            label_filter,
            [p.lower() for p in (filt.priorities or [])],
            {normalize_issue_type(p) for p in (filt.types or [])},
            filt.epics or [],
            filt.exclude_inactive,
            release_settings,
            logical_settings,
            default_branches,
            jira_ids,
        )
    except UnsupportedMetricError as e:
        raise ResponseError(InvalidRequestError("Unsupported metric: %s" % e)) from None
    return filt, defs, histograms


@expires_header(short_term_exptime)
@weight(5)
async def calc_metrics_jira_linear(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics over JIRA issue activities."""
    filt, time_intervals, tzoffset, metric_values, split_labels = await _calc_jira_entry(
        request, body, JIRAMetricsRequest,
    )
    mets = list(
        chain.from_iterable(
            (
                CalculatedJIRAMetricValues(
                    granularity=granularity,
                    with_=with_group,
                    jira_label=label,
                    values=[
                        CalculatedLinearMetricValues(
                            date=(dt - tzoffset).date(),
                            values=[v.value for v in vals],
                            confidence_mins=[v.confidence_min for v in vals],
                            confidence_maxs=[v.confidence_max for v in vals],
                            confidence_scores=[v.confidence_score() for v in vals],
                        )
                        for dt, vals in zip(ts, ts_values)
                    ],
                )
                for label, group_metric_values in zip(split_labels, label_metric_values)
                for granularity, ts, ts_values in zip(
                    filt.granularities,
                    time_intervals,
                    group_metric_values,
                )
            )
            for with_group, label_metric_values in zip(filt.with_ or [None], metric_values)
        ),
    )
    return model_response(mets)


async def _dereference_teams(
    with_: Optional[List[JIRAFilterWith]],
    account: int,
    sdb: Database,
    mdb: Database,
    cache: Optional[aiomcache.Client],
) -> Optional[List[JIRAFilterWith]]:
    if not with_:
        return with_
    teams = set()
    for i, group in enumerate(with_):
        for topic in ("assignees", "reporters", "commenters"):
            for j, dev in enumerate(getattr(group, topic, []) or []):
                if dev is not None and dev.startswith("{"):
                    try:
                        if not dev.endswith("}"):
                            raise ValueError
                        teams.add(int(dev[1:-1]))
                    except ValueError:
                        raise ResponseError(
                            InvalidRequestError(
                                pointer=f".with[{i}].{topic}[{j}]",
                                detail=f"Invalid team ID: {dev}",
                            ),
                        )
    teams_map = await fetch_teams_map(teams, account, sdb)
    all_team_members = set(chain.from_iterable(teams_map.values()))
    jira_map = await load_mapped_jira_users(account, all_team_members, sdb, mdb, cache)
    del all_team_members
    deref = []
    for group in with_:
        new_group = {}
        changed = False
        for topic in ("assignees", "reporters", "commenters"):
            if topic_devs := getattr(group, topic):
                new_topic_devs = []
                topic_changed = False
                for dev in topic_devs:
                    if dev is not None and dev.startswith("{"):
                        topic_changed = True
                        for member in teams_map[int(dev[1:-1])]:
                            try:
                                new_topic_devs.append(jira_map[member])
                            except KeyError:
                                continue
                    else:
                        new_topic_devs.append(dev)
                if topic_changed:
                    changed = True
                    new_group[topic] = new_topic_devs
                else:
                    new_group[topic] = topic_devs
        deref.append(JIRAFilterWith(**new_group) if changed else group)
    return deref


@expires_header(short_term_exptime)
@weight(1.5)
async def calc_histogram_jira(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate histograms over JIRA issue activities."""
    filt, defs, histograms = await _calc_jira_entry(request, body, JIRAHistogramsRequest)
    result = []
    for metrics, def_hists in zip(defs.values(), histograms):
        for with_, with_hists in zip(filt.with_ or [None], def_hists):
            for metric, histogram in zip(metrics, with_hists[0][0]):
                result.append(
                    CalculatedJIRAHistogram(
                        with_=with_,
                        metric=metric,
                        scale=histogram.scale.name.lower(),
                        ticks=histogram.ticks,
                        frequencies=histogram.frequencies,
                        interquartile=Interquartile(*histogram.interquartile),
                    ),
                )
    return model_response(result)
