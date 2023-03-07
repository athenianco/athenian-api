import asyncio
from collections import Counter, defaultdict
import dataclasses
from datetime import datetime, timedelta, timezone
from itertools import chain, repeat
import logging
from typing import Any, Collection, Mapping, Optional

from aiohttp import web
import aiomcache
import numpy as np
import numpy.typing as npt
import pandas as pd
import sentry_sdk
from sqlalchemy import select, union_all
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_utils import gather, list_with_yield, read_sql_query
from athenian.api.balancing import weight
from athenian.api.cache import cached, expires_header, short_term_exptime
from athenian.api.controllers.filter_controller import webify_deployment
from athenian.api.controllers.jira_controller.common import (
    AccountInfo,
    build_issue_web_models,
    collect_account_info,
    fetch_issues_prs,
    fetch_issues_users,
    web_prs_map_from_struct,
)
from athenian.api.controllers.jira_controller.get_jira_issues import get_jira_issues
from athenian.api.db import Database
from athenian.api.internal.datetime_utils import split_to_time_intervals
from athenian.api.internal.features.entries import UnsupportedMetricError, make_calculator
from athenian.api.internal.features.github.pull_request_filter import (
    PullRequestListMiner,
    fetch_pr_deployments,
)
from athenian.api.internal.features.histogram import HistogramParameters, Scale
from athenian.api.internal.jira import JIRAConfig, normalize_issue_type, normalize_priority
from athenian.api.internal.logical_repos import drop_logical_repo
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.deployment_light import fetch_repository_environments
from athenian.api.internal.miners.jira.epic import filter_epics
from athenian.api.internal.miners.jira.issue import (
    ISSUE_PR_IDS,
    ISSUE_PRS_BEGAN,
    ISSUE_PRS_COUNT,
    ISSUE_PRS_RELEASED,
    fetch_jira_issues,
    participant_columns,
    query_jira_raw,
    resolve_resolved,
    resolve_work_began,
)
from athenian.api.internal.miners.types import Deployment
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings, Settings
from athenian.api.internal.with_ import resolve_jira_with
from athenian.api.models.metadata.github import Branch, PullRequest
from athenian.api.models.metadata.jira import (
    AthenianIssue,
    Component,
    Issue,
    IssueType,
    Priority,
    Status,
)
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
from athenian.api.models.web_model_io import deserialize_models, serialize_models
from athenian.api.request import AthenianWebRequest, model_from_body
from athenian.api.response import ResponseError, model_response
from athenian.api.tracing import sentry_span
from athenian.api.unordered_unique import unordered_unique


@expires_header(short_term_exptime)
@weight(2.0)
async def filter_jira_stuff(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find JIRA epics and labels used in the specified date interval."""
    filt = model_from_body(FilterJIRAStuff, body)
    return_ = set(filt.return_ or JIRAFilterReturn)
    if not filt.return_:
        return_.remove(JIRAFilterReturn.ONLY_FLYING)
    account_info = await collect_account_info(
        filt.account,
        request,
        JIRAFilterReturn.ISSUE_BODIES in return_ or JIRAFilterReturn.EPICS in return_,
    )
    if filt.projects is not None:
        jira_conf = account_info.jira_conf
        projects = jira_conf.project_ids_map(filt.projects)
        jira_conf = JIRAConfig(jira_conf.acc_id, projects, jira_conf.epics)
        account_info = dataclasses.replace(account_info, jira_conf=jira_conf)
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
    filt.priorities = {normalize_priority(p) for p in (filt.priorities or [])}
    filt.types = {normalize_issue_type(p) for p in (filt.types or [])}
    sdb, mdb, pdb, rdb = request.sdb, request.mdb, request.pdb, request.rdb
    cache = request.cache
    (
        (epics, epic_priorities, epic_statuses),
        (issues, labels, issue_users, issue_types, issue_priorities, issue_statuses, deps),
    ) = await gather(
        _epic_flow(
            return_,
            account_info,
            time_from,
            time_to,
            filt.exclude_inactive,
            label_filter,
            filt.priorities,
            reporters,
            assignees,
            commenters,
            mdb,
            pdb,
            cache,
        ),
        _issue_flow(
            return_,
            account_info,
            time_from,
            time_to,
            filt.exclude_inactive,
            label_filter,
            filt.priorities,
            filt.types,
            reporters,
            assignees,
            commenters,
            sdb,
            mdb,
            pdb,
            rdb,
            cache,
        ),
        op="forked flows",
    )
    if not epic_priorities:
        priorities = issue_priorities
    elif not issue_priorities:
        priorities = epic_priorities
    else:
        priorities = sorted(set(epic_priorities).union(issue_priorities))
    if not epic_statuses:
        statuses = issue_statuses
    elif not issue_statuses:
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
        native=True,
    )


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=serialize_models,
    deserialize=deserialize_models,
    key=lambda return_, account_info, time_from, time_to, exclude_inactive, label_filter, priorities, reporters, assignees, commenters, **_: (  # noqa
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
        ",".join("%s:%s" % db for db in sorted(account_info.default_branches.items())),
        account_info.release_settings,
        account_info.logical_settings,
    ),
)
async def _epic_flow(
    return_: set[str],
    account_info: AccountInfo,
    time_from: Optional[datetime],
    time_to: Optional[datetime],
    exclude_inactive: bool,
    label_filter: LabelFilter,
    priorities: Collection[str],
    reporters: Collection[str],
    assignees: Collection[Optional[str]],
    commenters: Collection[str],
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> tuple[list[JIRAEpic], list[JIRAPriority], list[JIRAStatus]]:
    """Fetch various information related to JIRA epics."""
    if JIRAFilterReturn.EPICS not in return_:
        return [], [], []
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
        Issue.priority_name,
        Issue.status_id,
        Issue.status,
        Issue.type_id,
        Issue.comments_count,
        Issue.url,
        Issue.story_points,
    ]
    if JIRAFilterReturn.USERS in return_:
        extra_columns.extend(participant_columns)
    epics_df, children_df, subtask_counts, epic_children_map = await filter_epics(
        account_info.jira_conf,
        time_from,
        time_to,
        exclude_inactive,
        label_filter,
        priorities,
        reporters,
        assignees,
        commenters,
        account_info.default_branches,
        account_info.release_settings,
        account_info.logical_settings,
        account_info.account,
        account_info.meta_ids,
        mdb,
        pdb,
        cache,
        extra_columns=extra_columns,
    )
    with sentry_sdk.start_span(op="materialize models", description=str(len(epics_df))):
        children_columns = {k: children_df[k].values for k in children_df.columns}
        children_columns[Issue.id.name] = children_df.index.values
        epics = []
        issue_by_id = {}
        issue_type_ids = {}
        children_by_type = defaultdict(list)
        epics_by_type = defaultdict(list)
        now = np.datetime64(datetime.utcnow())

        epics_work_began = resolve_work_began(
            epics_df[AthenianIssue.work_began.name].values, epics_df[ISSUE_PRS_BEGAN].values,
        )
        epics_resolved = resolve_resolved(
            epics_df[AthenianIssue.resolved.name].values,
            epics_df[ISSUE_PRS_BEGAN].values,
            epics_df[ISSUE_PRS_RELEASED].values,
        )
        for (
            epic_id,
            project_id,
            epic_key,
            epic_title,
            epic_created,
            epic_updated,
            epic_reporter,
            epic_assignee,
            epic_priority,
            epic_status,
            epic_type,
            epic_prs,
            epic_comments,
            epic_url,
            epic_story_points,
            epic_work_began,
            epic_resolved,
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
                    Issue.reporter_display_name.name,
                    Issue.assignee_display_name.name,
                    Issue.priority_name.name,
                    Issue.status.name,
                    Issue.type_id.name,
                    ISSUE_PRS_COUNT,
                    Issue.comments_count.name,
                    Issue.url.name,
                    Issue.story_points.name,
                )
            ),
            epics_work_began,
            epics_resolved,
        ):
            epic_work_began = epic_work_began if epic_work_began == epic_work_began else None
            epic_resolved = epic_resolved if epic_resolved == epic_resolved else None
            epics.append(
                epic := JIRAEpic(
                    id=epic_key,
                    project=project_id.decode(),
                    children=[],
                    title=epic_title,
                    created=epic_created,
                    updated=epic_updated,
                    work_began=epic_work_began,
                    resolved=epic_resolved,
                    reporter=epic_reporter,
                    assignee=epic_assignee,
                    comments=epic_comments,
                    priority=epic_priority,
                    status=epic_status,
                    type=epic_type,
                    prs=epic_prs,
                    url=epic_url,
                    life_time=timedelta(0),
                    story_points=epic_story_points,
                ),
            )
            epics_by_type[(project_id, epic_type)].append(epic)
            children_indexes = epic_children_map.get(epic_id, [])
            project_type_ids = issue_type_ids.setdefault(project_id, set())
            project_type_ids.add(epic_type)

            children_work_began = resolve_work_began(
                children_columns[AthenianIssue.work_began.name][children_indexes],
                children_columns[ISSUE_PRS_BEGAN][children_indexes],
            )
            children_resolved = resolve_resolved(
                children_columns[AthenianIssue.resolved.name][children_indexes],
                children_columns[ISSUE_PRS_BEGAN][children_indexes],
                children_columns[ISSUE_PRS_RELEASED][children_indexes],
            )

            for (
                child_id,
                child_key,
                child_title,
                child_created,
                child_updated,
                child_comments,
                child_reporter,
                child_assignee,
                child_priority,
                child_status,
                child_prs,
                child_type,
                child_url,
                child_story_points,
                child_work_began,
                child_resolved,
            ) in zip(
                *(
                    children_columns[column][children_indexes]
                    for column in (
                        Issue.id.name,
                        Issue.key.name,
                        Issue.title.name,
                        Issue.created.name,
                        AthenianIssue.updated.name,
                        Issue.comments_count.name,
                        Issue.reporter_display_name.name,
                        Issue.assignee_display_name.name,
                        Issue.priority_name.name,
                        Issue.status.name,
                        ISSUE_PRS_COUNT,
                        Issue.type_id.name,
                        Issue.url.name,
                        Issue.story_points.name,
                    )
                ),
                children_work_began,
                children_resolved,
            ):
                epic.prs += child_prs
                child_work_began = (
                    child_work_began if child_work_began == child_work_began else None
                )
                child_resolved = child_resolved if child_resolved == child_resolved else None
                if child_work_began is not None:
                    epic.work_began = min(epic.work_began or child_work_began, child_work_began)
                if child_resolved is not None:
                    lead_time = child_resolved - child_work_began
                    life_time = child_resolved - child_created
                else:
                    life_time = now - child_created
                    if child_work_began is not None:
                        lead_time = now - child_work_began
                    else:
                        lead_time = None
                if child_resolved is None:
                    epic.resolved = None
                project_type_ids.add(child_type)
                epic.children.append(
                    child := JIRAEpicChild(
                        id=child_key,
                        title=child_title,
                        created=child_created,
                        updated=child_updated,
                        work_began=child_work_began,
                        lead_time=lead_time,
                        life_time=life_time,
                        resolved=child_resolved,
                        reporter=child_reporter,
                        assignee=child_assignee,
                        comments=child_comments,
                        priority=child_priority,
                        status=child_status,
                        prs=child_prs,
                        type=child_type,
                        subtasks=0,
                        url=child_url,
                        story_points=child_story_points,
                    ),
                )
                issue_by_id[child_id] = child
                children_by_type[(project_id, child_type)].append(child)
                if len(issue_by_id) % 1000 == 0:
                    await asyncio.sleep(0)
            if epic.resolved is not None:
                epic.lead_time = epic.resolved - epic.work_began
                epic.life_time = epic.resolved - epic.created
            else:
                epic.life_time = now - epic.created
                if epic.work_began is not None:
                    epic.lead_time = now - epic.work_began
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
    jira_acc_id = account_info.jira_conf.acc_id
    priorities, statuses, types = await gather(
        _fetch_priorities(priority_ids, jira_acc_id, return_, mdb),
        _fetch_statuses(status_ids, status_project_map, jira_acc_id, return_, mdb),
        _fetch_types(
            issue_type_ids,
            jira_acc_id,
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
    serialize=serialize_models,
    deserialize=deserialize_models,
    key=lambda return_, account_info, time_from, time_to, exclude_inactive, label_filter, priorities, reporters, assignees, commenters, **_: (  # noqa
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
        ",".join("%s:%s" % db for db in sorted(account_info.default_branches.items())),
        account_info.release_settings,
        account_info.logical_settings,
    ),
)
async def _issue_flow(
    return_: set[str],
    account_info: AccountInfo,
    time_from: Optional[datetime],
    time_to: Optional[datetime],
    exclude_inactive: bool,
    label_filter: LabelFilter,
    priorities: Collection[str],
    types: Collection[str],
    reporters: Collection[str],
    assignees: Collection[Optional[str]],
    commenters: Collection[str],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> tuple[
    list[JIRAIssue],
    list[JIRALabel],
    list[JIRAUser],
    list[JIRAIssueType],
    list[JIRAPriority],
    list[JIRAStatus],
    Optional[dict[str, WebDeploymentNotification]],
]:
    """Fetch various information related to JIRA issues."""
    if JIRAFilterReturn.ISSUES not in return_ or return_ == {JIRAFilterReturn.ISSUES}:
        return [], [], [], [], [], [], None
    extra_columns = [
        Issue.epic_id,
        Issue.labels,
        Issue.project_id,
        Issue.components,
        Issue.assignee_id,
        Issue.reporter_id,
        Issue.commenters_ids,
        Issue.priority_id,
        Issue.priority_name,
        Issue.status_id,
        Issue.status,
        Issue.type_id,
        Issue.comments_count,
        Issue.story_points,
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
    jira_filter = JIRAFilter.from_jira_config(account_info.jira_conf).replace(
        epics=epics,
        labels=label_filter,
        issue_types=types,
        # priorities are already lower-cased and de-None-d
        priorities=priorities,
    )
    jira_acc_id = account_info.jira_conf.acc_id

    if full_fetch := return_.difference(
        {
            JIRAFilterReturn.ISSUES,
            JIRAFilterReturn.PRIORITIES,
            JIRAFilterReturn.ISSUE_TYPES,
            JIRAFilterReturn.STATUSES,
        },
    ):
        issues = await fetch_jira_issues(
            time_from,
            time_to,
            jira_filter,
            exclude_inactive,
            reporters,
            assignees,
            commenters,
            False,
            account_info.default_branches,
            account_info.release_settings,
            account_info.logical_settings,
            account_info.account,
            account_info.meta_ids,
            mdb,
            pdb,
            cache,
            extra_columns=extra_columns,
            adjust_timestamps_using_prs=JIRAFilterReturn.ISSUE_BODIES in return_,
        )
    else:
        # fast lane: it's enough to select distinct ID values
        distinct_columns = [Issue.project_id]
        if JIRAFilterReturn.PRIORITIES in return_:
            distinct_columns.append(Issue.priority_id)
        if JIRAFilterReturn.ISSUE_TYPES in return_:
            distinct_columns.append(Issue.type_id)
        if JIRAFilterReturn.STATUSES in return_:
            distinct_columns.append(Issue.status_id)
        issues = await query_jira_raw(
            distinct_columns,
            time_from,
            time_to,
            jira_filter,
            exclude_inactive,
            reporters,
            assignees,
            commenters,
            False,
            mdb,
            cache,
            distinct=True,
        )
    if JIRAFilterReturn.LABELS in return_:
        components = Counter(chain.from_iterable(_nonzero(issues[Issue.components.name].values)))
    else:
        components = []
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
        for project_id, issue_type_id, count in zip(
            issues[Issue.project_id.name].values,
            issues[Issue.type_id.name].values,
            repeat(1) if full_fetch else issues["count"].values,
        ):
            issue_type_projects[project_id].add(issue_type_id)
            issue_type_counts[(project_id, issue_type_id)] += count
    else:
        issue_type_counts = issue_type_projects = []

    if JIRAFilterReturn.ISSUE_BODIES in return_:
        if not issues.empty:
            pr_ids = np.concatenate(issues[ISSUE_PR_IDS].values, dtype=int, casting="unsafe")
        else:
            pr_ids = []
    else:
        pr_ids = []

    @sentry_span
    async def fetch_components():
        if JIRAFilterReturn.LABELS not in return_:
            return []
        return await mdb.fetch_all(
            select(Component.id, Component.name).where(
                Component.id.in_(components), Component.acc_id == jira_acc_id,
            ),
        )

    @sentry_span
    async def fetch_users():
        if JIRAFilterReturn.USERS not in return_:
            return []
        return await fetch_issues_users(issues, account_info, sdb, mdb)

    @sentry_span
    async def extract_labels():
        if JIRAFilterReturn.LABELS in return_:
            labels_column = issues[Issue.labels.name].values
            if label_filter:
                labels_column = (ils for ils in labels_column if label_filter.match(ils))
            labels = Counter(chain.from_iterable(labels_column))
            if None in labels:
                del labels[None]
            last_useds = {}
            for updated, issue_labels in zip(
                issues[Issue.updated.name],
                issues[Issue.labels.name].values,
            ):
                for label in issue_labels or ():
                    if last_useds.setdefault(label, updated) < updated:
                        last_useds[label] = updated
            labels = {
                k: JIRALabel(title=k, kind="regular", issues_count=v, last_used=last_useds[k])
                for k, v in labels.items()
            }
            if mdb.url.dialect == "sqlite":
                for label in labels.values():
                    label.last_used = label.last_used.replace(tzinfo=timezone.utc)
        else:
            labels = []
        return labels

    async def _fetch_prs() -> tuple[
        dict[str, WebPullRequest],
        dict[str, Deployment],
    ]:
        if JIRAFilterReturn.ISSUE_BODIES not in return_ or len(pr_ids) == 0:
            return {}, {}
        prs, deployments = await fetch_issues_prs(pr_ids, account_info, sdb, mdb, pdb, rdb, cache)
        web_prs = web_prs_map_from_struct(prs, account_info.prefixer)
        return web_prs, deployments

    (
        (prs, deps),
        component_names,
        users,
        priorities,
        statuses,
        issue_types,
        labels,
    ) = await gather(
        _fetch_prs(),
        fetch_components(),
        fetch_users(),
        _fetch_priorities(priorities, jira_acc_id, return_, mdb),
        _fetch_statuses(statuses, status_project_map, jira_acc_id, return_, mdb),
        _fetch_types(issue_type_projects, jira_acc_id, return_, mdb),
        extract_labels(),
    )
    if deps is not None:
        prefix_logical_repo = account_info.prefixer.prefix_logical_repo
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

    web_issue_types = [
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
    ]
    if JIRAFilterReturn.LABELS in return_:
        last_useds = {}
        for updated, issue_components in zip(
            issues[Issue.updated.name],
            issues[Issue.components.name].values,
        ):
            for component in issue_components or ():
                if last_useds.setdefault(component, updated) < updated:
                    last_useds[component] = updated
        components = {
            row[0]: JIRALabel(
                title=row[1],
                kind="component",
                issues_count=components[row[0]],
                last_used=last_useds[row[0]],
            )
            for row in component_names
        }
        if mdb.url.dialect == "sqlite":
            for label in components.values():
                label.last_used = label.last_used.replace(tzinfo=timezone.utc)

        labels = sorted(chain(components.values(), labels.values()))
    if JIRAFilterReturn.ISSUE_BODIES in return_:
        issue_models = build_issue_web_models(issues, prs, issue_types)
    else:
        issue_models = []
    return issue_models, labels, users, web_issue_types, priorities, statuses, deps


@sentry_span
async def _fetch_priorities(
    priorities: npt.NDArray[bytes],
    acc_id: int,
    return_: set[str],
    mdb: Database,
) -> list[JIRAPriority] | None:
    if JIRAFilterReturn.PRIORITIES not in return_ or len(priorities) == 0:
        return []
    rows = await mdb.fetch_all(
        select(Priority.name, Priority.icon_url, Priority.rank, Priority.status_color)
        .where(Priority.acc_id == acc_id, Priority.id.in_(priorities))
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
    status_project_map: dict[bytes, set[bytes]],
    acc_id: int,
    return_: set[str],
    mdb: Database,
) -> Optional[list[JIRAStatus]]:
    if JIRAFilterReturn.STATUSES not in return_ or len(statuses) == 0:
        return []
    rows = await mdb.fetch_all(
        select(Status.id, Status.name, Status.category_name)
        .where(Status.acc_id == acc_id, Status.id.in_(statuses))
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
    return_: set[str],
    mdb: Database,
    columns: Optional[list[InstrumentedAttribute]] = None,
) -> Optional[list[Mapping[str, Any]]]:
    if (
        JIRAFilterReturn.ISSUE_TYPES not in return_
        and JIRAFilterReturn.ISSUE_BODIES not in return_
        and JIRAFilterReturn.EPICS not in return_
    ) or len(issue_type_projects) == 0:
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
        select(*columns).where(
            IssueType.acc_id == acc_id,
            IssueType.id.in_(np.fromiter(ids, "S8", len(ids))),
            IssueType.project_id == project_id.decode(),
        )
        for project_id, ids in issue_type_projects.items()
    ]
    return await mdb.fetch_all(union_all(*queries))


def _nonzero(arr: np.ndarray) -> np.ndarray:
    return arr[arr.nonzero()[0]]


async def _calc_linear_entry(
    request: AthenianWebRequest,
    metrics_req: JIRAMetricsRequest,
) -> tuple[list[list[datetime]], timedelta, np.ndarray, np.ndarray]:
    account_info = await collect_account_info(metrics_req.account, request, True)
    time_intervals, tzoffset = _request_time_intervals(metrics_req)
    participants = await resolve_jira_with(
        metrics_req.with_, metrics_req.account, request.sdb, request.mdb, request.cache,
    )
    meta_ids = account_info.meta_ids
    calculator = make_calculator(
        metrics_req.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache,
    )
    groups = _jira_metrics_request_read_groups(metrics_req, account_info.jira_conf)
    metric_values, split_labels = await calculator.calc_jira_metrics_line_github(
        metrics_req.metrics,
        time_intervals,
        metrics_req.quantiles or (0, 1),
        participants,
        groups,
        metrics_req.group_by_jira_label,
        metrics_req.exclude_inactive,
        account_info.release_settings,
        account_info.logical_settings,
        account_info.default_branches,
    )
    return time_intervals, tzoffset, metric_values, split_labels


async def _calc_histogram_entry(
    request: AthenianWebRequest,
    hist_req: JIRAHistogramsRequest,
) -> tuple[dict[HistogramParameters, list[str]], np.ndarray]:
    account_info = await collect_account_info(hist_req.account, request, True)
    meta_ids = account_info.meta_ids
    jira_conf = account_info.jira_conf
    time_intervals, tzoffset = _request_time_intervals(hist_req)
    participants = await resolve_jira_with(
        hist_req.with_, hist_req.account, request.sdb, request.mdb, request.cache,
    )
    label_filter = LabelFilter.from_iterables(hist_req.labels_include, hist_req.labels_exclude)

    calculator = make_calculator(
        hist_req.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache,
    )
    defs = defaultdict(list)

    if hist_req.projects is not None:
        projects = jira_conf.project_ids_map(hist_req.projects)
        jira_conf = JIRAConfig(jira_conf.acc_id, projects, jira_conf.epics)

    for h in hist_req.histograms or []:
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
            hist_req.quantiles or (0, 1),
            participants,
            label_filter,
            {normalize_priority(p) for p in (hist_req.priorities or [])},
            {normalize_issue_type(t) for t in (hist_req.types or [])},
            hist_req.epics or [],
            hist_req.exclude_inactive,
            account_info.release_settings,
            account_info.logical_settings,
            account_info.default_branches,
            jira_conf,
        )
    except UnsupportedMetricError as e:
        raise ResponseError(InvalidRequestError("Unsupported metric: %s" % e)) from None
    return defs, histograms


def _request_time_intervals(
    req: JIRAMetricsRequest | JIRAHistogramsRequest,
) -> tuple[list[list[datetime]], timedelta]:
    granularities = req.granularities if isinstance(req, JIRAMetricsRequest) else ["all"]
    return split_to_time_intervals(req.date_from, req.date_to, granularities, req.timezone)


def _jira_metrics_request_read_groups(
    request: JIRAMetricsRequest,
    jira_conf: JIRAConfig,
) -> list[JIRAFilter]:
    if request.for_:
        # this incompatibilty is not expressed by the spec and must be checked manually
        not_compatible_fields = (
            "priorities",
            "types",
            "projects",
            "labels_include",
            "labels_exclude",
            "group_by_jira_label",
        )
        if not_compatible := [f for f in not_compatible_fields if getattr(request, f)]:
            fields_repr = ",".join(f"`{f}`" for f in not_compatible)
            msg = f"`for` cannot be used with {fields_repr}"
            raise ResponseError(InvalidRequestError(pointer=".for", detail=msg))
        return [
            JIRAFilter.from_web(group, jira_conf) if group
            # we don't want JIRAFilter.empty() for empty groups like when using JIRAFilter for PRs
            # we want a JIRAFilter with a valid account, see calc_jira_metrics_line_github
            else JIRAFilter.from_jira_config(jira_conf)
            for group in request.for_
        ]
    else:
        group = JIRAFilter.from_jira_config(jira_conf).replace(
            labels=LabelFilter.from_iterables(request.labels_include, request.labels_exclude),
            priorities=frozenset([normalize_priority(p) for p in (request.priorities or [])]),
            issue_types=frozenset([normalize_issue_type(t) for t in (request.types or [])]),
            epics=frozenset([s.upper() for s in (request.epics or [])]),
            custom_projects=False,
        )
        if request.projects is not None:
            projects = jira_conf.project_ids_map(request.projects)
            group = group.replace(projects=frozenset(projects), custom_projects=True)

        return [group]


@expires_header(short_term_exptime)
@weight(5)
async def calc_metrics_jira_linear(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics over JIRA issue activities."""
    metrics_req = model_from_body(JIRAMetricsRequest, body)
    time_intervals, tzoffset, metric_values, split_labels = await _calc_linear_entry(
        request, metrics_req,
    )

    # response different depending on usage of groups vs group_by_jira_label
    if metrics_req.for_:
        res_groups = metrics_req.for_
        split_labels = np.full(len(metrics_req.for_), None)
    else:
        res_groups = [None] * len(split_labels)

    mets = [
        CalculatedJIRAMetricValues(
            granularity=gran,
            with_=with_group,
            jira_label=label,
            for_=res_group,
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
        for with_group, label_values in zip(metrics_req.with_ or [None], metric_values)
        for label, res_group, group_values in zip(split_labels, res_groups, label_values)
        for gran, ts, ts_values in zip(metrics_req.granularities, time_intervals, group_values)
    ]
    return model_response(mets)


@expires_header(short_term_exptime)
@weight(1.5)
async def calc_histogram_jira(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate histograms over JIRA issue activities."""
    hist_req = model_from_body(JIRAHistogramsRequest, body)
    defs, histograms = await _calc_histogram_entry(request, hist_req)
    result = []
    for metrics, def_hists in zip(defs.values(), histograms):
        for with_, with_hists in zip(hist_req.with_ or [None], def_hists):
            for metric, histogram in zip(metrics, with_hists[0][0]):
                result.append(
                    CalculatedJIRAHistogram(
                        with_=with_,
                        metric=metric,
                        scale=histogram.scale.name.lower(),
                        ticks=histogram.ticks,
                        frequencies=histogram.frequencies,
                        interquartile=Interquartile(
                            left=histogram.interquartile[0], right=histogram.interquartile[1],
                        ),
                    ),
                )
    return model_response(result)
