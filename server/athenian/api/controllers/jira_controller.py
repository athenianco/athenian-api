import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from functools import partial
from itertools import chain
import logging
import pickle
from typing import Collection, Dict, List, Optional, Set, Tuple, Type, Union

from aiohttp import web
import aiomcache
import databases
import numpy as np
import pandas as pd
from sqlalchemy import and_, func, select, union_all

from athenian.api import list_with_yield, metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.balancing import weight
from athenian.api.cache import cached
from athenian.api.controllers.account import get_account_repositories, get_metadata_account_ids
from athenian.api.controllers.datetime_utils import split_to_time_intervals
from athenian.api.controllers.features.github.pull_request_filter import PullRequestListMiner, \
    unwrap_pull_requests
from athenian.api.controllers.features.histogram import HistogramParameters, Scale
from athenian.api.controllers.features.jira.issue_metrics import JIRABinnedHistogramCalculator, \
    JIRABinnedMetricCalculator
from athenian.api.controllers.features.metric_calculator import group_to_indexes
from athenian.api.controllers.filter_controller import web_pr_from_struct
from athenian.api.controllers.jira import get_jira_installation
from athenian.api.controllers.miners.filters import LabelFilter
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.precomputed_prs import load_precomputed_done_facts_ids
from athenian.api.controllers.miners.jira.epic import filter_epics
from athenian.api.controllers.miners.jira.issue import fetch_jira_issues, ISSUE_PR_IDS, \
    ISSUE_PRS_BEGAN, \
    ISSUE_PRS_COUNT, ISSUE_PRS_RELEASED, resolve_work_began_and_resolved
from athenian.api.controllers.settings import ReleaseMatchSetting, Settings
from athenian.api.models.metadata.github import Branch, PullRequest
from athenian.api.models.metadata.jira import AthenianIssue, Component, Issue, IssueType, \
    Priority, Status, User
from athenian.api.models.state.models import MappedJIRAIdentity
from athenian.api.models.web import CalculatedJIRAHistogram, CalculatedJIRAMetricValues, \
    CalculatedLinearMetricValues, FilteredJIRAStuff, FilterJIRAStuff, Interquartile, \
    InvalidRequestError, JIRAEpic, JIRAEpicChild, JIRAFilterReturn, JIRAFilterWith, \
    JIRAHistogramsRequest, JIRAIssue, JIRAIssueType, JIRALabel, JIRAMetricsRequest, JIRAPriority, \
    JIRAStatus, JIRAUser, PullRequest as WebPullRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError
from athenian.api.tracing import sentry_span


@weight(1.0)
async def filter_jira_stuff(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find JIRA epics and labels used in the specified date interval."""
    try:
        filt = FilterJIRAStuff.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    meta_ids, jira_ids, branches, default_branches, release_settings = await _collect_ids(
        filt.account, request, request.sdb, request.mdb, request.cache)
    if filt.date_from is None or filt.date_to is None:
        if (filt.date_from is None) != (filt.date_to is None):
            raise ResponseError(InvalidRequestError(
                ".date_from",
                detail="date_from and date_to must be either both not null or both null"))
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
    filt.types = [p.lower() for p in (filt.types or [])]
    return_ = set(filt.return_ or JIRAFilterReturn)
    sdb, mdb, pdb, rdb = request.sdb, request.mdb, request.pdb, request.rdb
    cache = request.cache
    tasks = [
        _epic_flow(return_, jira_ids, time_from, time_to, filt.exclude_inactive, label_filter,
                   filt.priorities, reporters, assignees, commenters, default_branches,
                   release_settings, meta_ids, mdb, pdb, cache),
        _issue_flow(return_, filt.account, jira_ids, time_from, time_to, filt.exclude_inactive,
                    label_filter, filt.priorities, filt.types, reporters, assignees, commenters,
                    branches, default_branches, release_settings, meta_ids,
                    sdb, mdb, pdb, rdb, cache),
    ]
    ((epics, epic_priorities, epic_statuses),
     (issues, labels, issue_users, issue_types, issue_priorities, issue_statuses),
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
    return model_response(FilteredJIRAStuff(
        epics=epics,
        issues=issues,
        labels=labels,
        issue_types=issue_types,
        priorities=priorities,
        users=issue_users,
        statuses=statuses,
    ))


@sentry_span
@cached(
    exptime=5 * 60,  # 5 min
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda return_, time_from, time_to, exclude_inactive, label_filter, priorities, reporters, assignees, commenters, default_branches, release_settings, **_: (  # noqa
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
    ),
    version=2,
)
async def _epic_flow(return_: Set[str],
                     jira_ids: Tuple[int, List[str]],
                     time_from: Optional[datetime],
                     time_to: Optional[datetime],
                     exclude_inactive: bool,
                     label_filter: LabelFilter,
                     priorities: Collection[str],
                     reporters: Collection[str],
                     assignees: Collection[Optional[str]],
                     commenters: Collection[str],
                     default_branches: Dict[str, str],
                     release_settings: Dict[str, ReleaseMatchSetting],
                     meta_ids: Tuple[int, ...],
                     mdb: databases.Database,
                     pdb: databases.Database,
                     cache: Optional[aiomcache.Client],
                     ) -> Tuple[Optional[List[JIRAEpic]],
                                Optional[List[JIRAPriority]],
                                Optional[List[JIRAStatus]]]:
    """Fetch various information related to JIRA epics."""
    if JIRAFilterReturn.EPICS not in return_:
        return None, None, None
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
        Issue.comments_count,
        Issue.url,
    ]
    if JIRAFilterReturn.USERS in return_:
        extra_columns.extend(_participant_columns)
    epics_df, children_df, subtask_coro, epic_children_map = await filter_epics(
        jira_ids, time_from, time_to, exclude_inactive, label_filter,
        priorities, reporters, assignees, commenters, default_branches,
        release_settings, meta_ids, mdb, pdb, cache, extra_columns=extra_columns,
    )
    children_columns = {k: children_df[k].values for k in children_df.columns}
    children_columns[Issue.id.key] = children_df.index.values
    epics = []
    issue_by_id = {}
    now = datetime.utcnow()
    for epic_id, project_id, epic_key, epic_title, epic_created, epic_updated, epic_prs_began,\
        epic_work_began, epic_prs_released, epic_resolved, epic_reporter, epic_assignee, \
        epic_priority, epic_status, epic_prs, epic_comments, epic_url in zip(
            epics_df.index.values, *(epics_df[column].values for column in (
            Issue.project_id.key,
            Issue.key.key,
            Issue.title.key,
            Issue.created.key,
            AthenianIssue.updated.key,
            ISSUE_PRS_BEGAN,
            AthenianIssue.work_began.key,
            ISSUE_PRS_RELEASED,
            AthenianIssue.resolved.key,
            Issue.reporter_display_name.key,
            Issue.assignee_display_name.key,
            Issue.priority_name.key,
            Issue.status.key,
            ISSUE_PRS_COUNT,
            Issue.comments_count.key,
            Issue.url.key,
            ))):
        work_began, resolved = resolve_work_began_and_resolved(
            epic_work_began, epic_prs_began, epic_resolved, epic_prs_released)
        epics.append(epic := JIRAEpic(
            id=epic_key,
            project=project_id,
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
            prs=epic_prs,
            url=epic_url,
        ))
        children_indexes = epic_children_map.get(epic_id, [])
        for child_id, child_key, child_title, child_created, child_updated, child_prs_began, \
            child_work_began, child_prs_released, child_resolved, child_comments, child_reporter, \
            child_assignee, child_priority, child_status, child_prs, child_type, child_url \
            in zip(*(children_columns[column][children_indexes] for column in (
                Issue.id.key,
                Issue.key.key,
                Issue.title.key,
                Issue.created.key,
                AthenianIssue.updated.key,
                ISSUE_PRS_BEGAN,
                AthenianIssue.work_began.key,
                ISSUE_PRS_RELEASED,
                AthenianIssue.resolved.key,
                Issue.comments_count.key,
                Issue.reporter_display_name.key,
                Issue.assignee_display_name.key,
                Issue.priority_name.key,
                Issue.status.key,
                ISSUE_PRS_COUNT,
                Issue.type.key,
                Issue.url.key,
                ))):  # noqa(E123)
            epic.prs += child_prs
            work_began, resolved = resolve_work_began_and_resolved(
                child_work_began, child_prs_began, child_resolved, child_prs_released)
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
            epic.children.append(child := JIRAEpicChild(
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
            ))
            issue_by_id[child_id] = child
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
        priority_ids = np.unique(np.concatenate([epics_df[Issue.priority_id.key].values,
                                                 children_columns[Issue.priority_id.key]]))
    else:
        priority_ids = []
    if JIRAFilterReturn.STATUSES in return_:
        status_ids = np.unique(np.concatenate([epics_df[Issue.status_id.key].values,
                                               children_columns[Issue.status_id.key]]))
        status_project_map = dict(zip(epics_df[Issue.status_id.key].values,
                                      epics_df[Issue.project_id.key].values))
        status_project_map.update(zip(children_columns[Issue.status_id.key],
                                      children_columns[Issue.project_id.key]))
    else:
        status_ids = []
        status_project_map = {}
    tasks = [
        subtask_coro,
        _fetch_priorities(priority_ids, jira_ids[0], return_, mdb),
        _fetch_statuses(status_ids, status_project_map, jira_ids[0], return_, mdb),
    ]
    subtask_counts, priorities, statuses = await gather(*tasks, op="epic epilog")
    for row in subtask_counts:
        issue_by_id[row[Issue.parent_id.key]].subtasks = row["subtasks"]
    return epics, priorities, statuses


@sentry_span
@cached(
    exptime=5 * 60,  # 5 min
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda return_, time_from, time_to, exclude_inactive, label_filter, priorities, reporters, assignees, commenters, default_branches, release_settings, **_: (  # noqa
        JIRAFilterReturn.ISSUES in return_,
        JIRAFilterReturn.ISSUE_BODIES in return_,
        JIRAFilterReturn.LABELS in return_,
        JIRAFilterReturn.USERS in return_,
        JIRAFilterReturn.ISSUE_TYPES in return_,
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
    ),
    version=2,
)
async def _issue_flow(return_: Set[str],
                      account: int,
                      jira_ids: Tuple[int, List[str]],
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
                      release_settings: Dict[str, ReleaseMatchSetting],
                      meta_ids: Tuple[int, ...],
                      sdb: databases.Database,
                      mdb: databases.Database,
                      pdb: databases.Database,
                      rdb: databases.Database,
                      cache: Optional[aiomcache.Client],
                      ) -> Tuple[Optional[JIRAIssue],
                                 Optional[JIRALabel],
                                 Optional[JIRAUser],
                                 Optional[JIRAIssueType],
                                 Optional[JIRAPriority],
                                 Optional[JIRAStatus]]:
    """Fetch various information related to JIRA issues."""
    if JIRAFilterReturn.ISSUES not in return_:
        return None, None, None, None, None, None
    log = logging.getLogger("%s.filter_jira_stuff" % metadata.__package__)
    extra_columns = [
        Issue.project_id,
        Issue.components,
        Issue.assignee_id,
        Issue.reporter_id,
        Issue.commenters_ids,
        Issue.priority_id,
        Issue.status_id,
        Issue.comments_count,
    ]
    if JIRAFilterReturn.ISSUE_BODIES in return_:
        extra_columns.extend([
            Issue.key,
            Issue.title,
            Issue.reporter_display_name,
            Issue.assignee_display_name,
            Issue.url,
        ])
    if JIRAFilterReturn.USERS in return_:
        extra_columns.extend(_participant_columns)
    issues = await fetch_jira_issues(
        jira_ids, time_from, time_to, exclude_inactive, label_filter,
        # priorities are already lower-cased and de-None-d
        priorities, types, [], reporters, assignees, commenters,
        default_branches, release_settings, meta_ids, mdb, pdb, cache,
        extra_columns=extra_columns)
    if JIRAFilterReturn.LABELS in return_:
        components = Counter(chain.from_iterable(
            _nonzero(issues[Issue.components.key].values)))
    else:
        components = None
    if JIRAFilterReturn.USERS in return_:
        people = np.unique(np.concatenate([
            _nonzero(issues[Issue.reporter_id.key].values),
            _nonzero(issues[Issue.assignee_id.key].values),
            list(chain.from_iterable(_nonzero(issues[Issue.commenters_ids.key].values))),
        ]))
        # we can leave None because `IN (null)` is always false
    else:
        people = None
    if JIRAFilterReturn.PRIORITIES in return_:
        priorities = issues[Issue.priority_id.key].unique()
    else:
        priorities = []
    if JIRAFilterReturn.STATUSES in return_:
        statuses = issues[Issue.status_id.key].unique()
        status_project_map = dict(zip(issues[Issue.status_id.key].values,
                                      issues[Issue.project_id.key].values))
    else:
        statuses = []
        status_project_map = {}
    if JIRAFilterReturn.ISSUE_TYPES in return_:
        issue_type_counts = Counter(issues[Issue.type.key].values)
        issue_type_projects = defaultdict(set)
        project_counts = defaultdict(int)
        for project_id, issue_type in zip(issues[Issue.project_id.key].values,
                                          issues[Issue.type.key].values):
            issue_type_projects[project_id].add(issue_type)
            project_counts[project_id] += 1
    else:
        issue_type_counts = issue_type_projects = project_counts = None
    if JIRAFilterReturn.ISSUE_BODIES in return_:
        if not issues.empty:
            pr_ids = np.concatenate(issues[ISSUE_PR_IDS].values)
        else:
            pr_ids = []
    else:
        pr_ids = None

    @sentry_span
    async def fetch_components():
        if JIRAFilterReturn.LABELS not in return_:
            return []
        return await mdb.fetch_all(
            select([Component.id, Component.name])
            .where(and_(
                Component.id.in_(components),
                Component.acc_id == jira_ids[0],
            )))

    @sentry_span
    async def fetch_users():
        if JIRAFilterReturn.USERS not in return_:
            return []
        return await mdb.fetch_all(
            select([User.display_name, User.avatar_url, User.type, User.id])
            .where(and_(
                User.id.in_(people),
                User.acc_id == jira_ids[0],
            ))
            .order_by(User.display_name))

    @sentry_span
    async def fetch_mapped_identities():
        if JIRAFilterReturn.USERS not in return_:
            return []
        return await sdb.fetch_all(
            select([MappedJIRAIdentity.github_user_id, MappedJIRAIdentity.jira_user_id])
            .where(and_(
                MappedJIRAIdentity.account_id == account,
                MappedJIRAIdentity.jira_user_id.in_(people),
            )))

    @sentry_span
    async def fetch_types():
        if JIRAFilterReturn.ISSUE_TYPES not in return_:
            return []
        queries = [
            select([IssueType.name, IssueType.project_id,
                    IssueType.icon_url, IssueType.is_subtask])
            .where(and_(
                IssueType.name.in_(names),
                IssueType.acc_id == jira_ids[0],
                IssueType.project_id == project_id,
            ))
            for project_id, names in issue_type_projects.items()
        ]
        return await mdb.fetch_all(union_all(*queries))

    @sentry_span
    async def extract_labels():
        if JIRAFilterReturn.LABELS in return_:
            label_iter = issues[Issue.labels.key].values
            if label_filter:
                label_iter = (ils for ils in label_iter if label_filter.match(ils))
            labels = Counter(chain.from_iterable(label_iter))
            if None in labels:
                del labels[None]
            labels = {k: JIRALabel(title=k, kind="regular", issues_count=v)
                      for k, v in labels.items()}
            for updated, issue_labels in zip(issues[Issue.updated.key],
                                             issues[Issue.labels.key].values):
                for label in (issue_labels or ()):
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
    async def _fetch_prs() -> Optional[Dict[str, WebPullRequest]]:
        if JIRAFilterReturn.ISSUE_BODIES not in return_:
            return None
        if len(pr_ids) == 0:
            return {}
        tasks = [
            read_sql_query(
                select([PullRequest]).where(and_(
                    PullRequest.acc_id.in_(meta_ids),
                    PullRequest.node_id.in_(pr_ids),
                )).order_by(PullRequest.node_id.key),
                mdb, PullRequest, index=PullRequest.node_id.key),
            load_precomputed_done_facts_ids(pr_ids, default_branches, release_settings, pdb),
        ]
        prs_df, (facts, ambiguous) = await gather(*tasks)
        related_branches = branches.take(np.nonzero(np.in1d(
            branches[Branch.repository_full_name.key].values.astype("U"),
            prs_df[PullRequest.repository_full_name.key].unique().astype("U")))[0])
        mined_prs, dfs, facts, _ = await unwrap_pull_requests(
            prs_df, facts, ambiguous, False, related_branches, default_branches, release_settings,
            account, meta_ids, mdb, pdb, rdb, cache)
        miner = PullRequestListMiner(
            mined_prs, dfs, facts, set(), set(),
            datetime(1970, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc), False)
        pr_list_items = await list_with_yield(miner, "PullRequestListMiner.__iter__")
        return dict(zip((pr.node_id for pr in pr_list_items),
                        (web_pr_from_struct(pr) for pr in pr_list_items)))

    prs, component_names, users, mapped_identities, priorities, statuses, issue_types, labels = \
        await gather(
            _fetch_prs(), fetch_components(), fetch_users(), fetch_mapped_identities(),
            _fetch_priorities(priorities, jira_ids[0], return_, mdb),
            _fetch_statuses(statuses, status_project_map, jira_ids[0], return_, mdb),
            fetch_types(), extract_labels())
    components = {
        row[0]: JIRALabel(title=row[1], kind="component", issues_count=components[row[0]])
        for row in component_names
    }
    mapped_identities = {
        r[MappedJIRAIdentity.jira_user_id.key]: r[MappedJIRAIdentity.github_user_id.key]
        for r in mapped_identities
    }
    users = [JIRAUser(avatar=row[User.avatar_url.key],
                      name=row[User.display_name.key],
                      type=row[User.type.key],
                      developer=mapped_identities.get(row[User.id.key]),
                      )
             for row in users] or None
    # take the issue type URL corresponding to the project with the most issues
    max_issue_types = {}
    for row in issue_types:
        name = row[IssueType.name.key]
        max_count = max_issue_types.get(name, (0, "", ""))[0]
        if (count := project_counts[row[IssueType.project_id.key]]) > max_count:
            max_issue_types[name] = (
                count,
                row[IssueType.icon_url.key],
                row[IssueType.project_id.key],
                row[IssueType.is_subtask.key],
            )
    issue_types = [
        JIRAIssueType(name=name,
                      image=image,
                      count=issue_type_counts[name],
                      project=project,
                      is_subtask=is_subtask,
                      normalized_name=name)
        for name, (_, image, project, is_subtask) in sorted(max_issue_types.items())] or None
    if JIRAFilterReturn.LABELS in return_:
        for updated, issue_components in zip(issues[Issue.updated.key],
                                             issues[Issue.components.key].values):
            for component in (issue_components or ()):
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
        for issue_key, issue_title, issue_created, issue_updated, issue_prs_began, \
            issue_work_began, issue_prs_released, issue_resolved, issue_reporter, \
            issue_assignee, issue_priority, issue_status, issue_prs, issue_type, issue_project, \
            issue_comments, issue_url in zip(*(
                issues[column].values for column in (
                Issue.key.key,
                Issue.title.key,
                Issue.created.key,
                AthenianIssue.updated.key,
                ISSUE_PRS_BEGAN,
                AthenianIssue.work_began.key,
                ISSUE_PRS_RELEASED,
                AthenianIssue.resolved.key,
                Issue.reporter_display_name.key,
                Issue.assignee_display_name.key,
                Issue.priority_name.key,
                Issue.status.key,
                ISSUE_PR_IDS,
                Issue.type.key,
                Issue.project_id.key,
                Issue.comments_count.key,
                Issue.url.key,
                ))):
            work_began, resolved = resolve_work_began_and_resolved(
                issue_work_began, issue_prs_began, issue_resolved, issue_prs_released)
            if resolved:
                lead_time = resolved - work_began
                life_time = resolved - issue_created
            else:
                life_time = now - pd.to_datetime(issue_created)
                if work_began:
                    lead_time = now - pd.to_datetime(work_began)
                else:
                    lead_time = None
            issue_models.append(JIRAIssue(
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
                project=issue_project,
                type=issue_type,
                prs=[prs[node_id] for node_id in issue_prs if node_id in prs],
                url=issue_url,
            ))
    else:
        issue_models = None
    return issue_models, labels, users, issue_types, priorities, statuses


@sentry_span
async def _fetch_priorities(priorities: Collection[str],
                            acc_id: int,
                            return_: Set[str],
                            mdb: databases.Database,
                            ) -> Optional[List[JIRAPriority]]:
    if JIRAFilterReturn.PRIORITIES not in return_:
        return None
    rows = await mdb.fetch_all(
        select([Priority.name, Priority.icon_url, Priority.rank, Priority.status_color])
        .where(and_(
            Priority.id.in_(priorities),
            Priority.acc_id == acc_id,
        ))
        .order_by(Priority.rank))
    return [JIRAPriority(name=row[Priority.name.key],
                         image=row[Priority.icon_url.key],
                         rank=row[Priority.rank.key],
                         color=row[Priority.status_color.key])
            for row in rows]


@sentry_span
async def _fetch_statuses(statuses: Collection[str],
                          status_project_map: Dict[str, str],
                          acc_id: int,
                          return_: Set[str],
                          mdb: databases.Database,
                          ) -> Optional[List[JIRAStatus]]:
    if JIRAFilterReturn.STATUSES not in return_:
        return None
    rows = await mdb.fetch_all(
        select([Status.id, Status.name, Status.category_name])
        .where(and_(
            Status.id.in_(statuses),
            Status.acc_id == acc_id,
        ))
        .order_by(Status.name))
    return [JIRAStatus(name=row[Status.name.key],
                       stage=row[Status.category_name.key],
                       project=status_project_map[row[Status.id.key]])
            for row in rows]


_participant_columns = [
    func.lower(Issue.reporter_display_name).label("reporter"),
    func.lower(Issue.assignee_display_name).label("assignee"),
    Issue.commenters_display_names.label("commenters"),
]


def _nonzero(arr: np.ndarray) -> np.ndarray:
    return arr[arr.nonzero()[0]]


async def _collect_ids(account: int,
                       request: AthenianWebRequest,
                       sdb: databases.Database,
                       mdb: databases.Database,
                       cache: Optional[aiomcache.Client],
                       ) -> Tuple[Tuple[int, ...],
                                  Tuple[int, List[str]],
                                  pd.DataFrame,
                                  Dict[str, str],
                                  Dict[str, ReleaseMatchSetting]]:
    tasks = [
        get_account_repositories(account, sdb),
        get_jira_installation(account, sdb, mdb, cache),
        get_metadata_account_ids(account, sdb, cache),
    ]
    repos, jira_ids, meta_ids = await gather(*tasks, op="sdb/ids")
    tasks = [
        extract_branches(
            [r.split("/", 1)[1] for r in repos], meta_ids, mdb, cache),
        Settings.from_request(request, account).list_release_matches(repos),
    ]
    (branches, default_branches), release_settings = await gather(
        *tasks, op="sdb/branches and releases")
    return meta_ids, jira_ids, branches, default_branches, release_settings


async def _calc_jira_entry(request: AthenianWebRequest,
                           body: dict,
                           model: Union[Type[JIRAMetricsRequest], Type[JIRAHistogramsRequest]],
                           ) -> Tuple[Union[JIRAMetricsRequest, JIRAHistogramsRequest],
                                      List[List[datetime]],
                                      pd.DataFrame,
                                      timedelta,
                                      LabelFilter]:
    try:
        filt = model.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e))) from None
    meta_ids, jira_ids, _, default_branches, release_settings = await _collect_ids(
        filt.account, request, request.sdb, request.mdb, request.cache)
    time_intervals, tzoffset = split_to_time_intervals(
        filt.date_from, filt.date_to, getattr(filt, "granularities", ["all"]), filt.timezone)
    reporters = list(set(chain.from_iterable(
        ([p.lower() for p in (g.reporters or [])]) for g in (filt.with_ or []))))
    assignees = list(set(chain.from_iterable(
        ([(p.lower() if p is not None else None) for p in (g.assignees or [])])
        for g in (filt.with_ or []))))
    commenters = list(set(chain.from_iterable(
        ([p.lower() for p in (g.commenters or [])]) for g in (filt.with_ or []))))
    label_filter = LabelFilter.from_iterables(filt.labels_include, filt.labels_exclude)
    issues = await fetch_jira_issues(
        jira_ids,
        time_intervals[0][0], time_intervals[0][-1], filt.exclude_inactive,
        label_filter,
        [p.lower() for p in (filt.priorities or [])],
        [p.lower() for p in (filt.types or [])],
        filt.epics or [],
        reporters, assignees, commenters,
        default_branches, release_settings,
        meta_ids, request.mdb, request.pdb, request.cache,
        extra_columns=_participant_columns if len(filt.with_ or []) > 1 else (),
    )
    return filt, time_intervals, issues, tzoffset, label_filter


@weight(2.5)
async def calc_metrics_jira_linear(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics over JIRA issue activities."""
    filt, time_intervals, issues, tzoffset, label_filter = await _calc_jira_entry(
        request, body, JIRAMetricsRequest)
    calc = JIRABinnedMetricCalculator(filt.metrics, filt.quantiles or [0, 1])
    label_splitter = _IssuesLabelSplitter(filt.group_by_jira_label, label_filter)
    groupers = partial(_split_issues_by_with, filt.with_), label_splitter
    groups = group_to_indexes(issues, *groupers)
    metric_values = calc(issues, time_intervals, groups)
    mets = list(chain.from_iterable((
        CalculatedJIRAMetricValues(
            granularity=granularity, with_=with_group, jira_label=label, values=[
                CalculatedLinearMetricValues(
                    date=(dt - tzoffset).date(),
                    values=[v.value for v in vals],
                    confidence_mins=[v.confidence_min for v in vals],
                    confidence_maxs=[v.confidence_max for v in vals],
                    confidence_scores=[v.confidence_score() for v in vals])
                for dt, vals in zip(ts, ts_values)
            ])
        for label, group_metric_values in zip(label_splitter.labels, label_metric_values)
        for granularity, ts, ts_values in zip(filt.granularities, time_intervals,
                                              group_metric_values)
    ) for with_group, label_metric_values in zip(filt.with_ or [None], metric_values)))
    return model_response(mets)


def _split_issues_by_with(with_: Optional[List[JIRAFilterWith]],
                          issues: pd.DataFrame,
                          ) -> List[np.ndarray]:
    result = []
    if len(with_ or []) < 2:
        return [np.arange(len(issues))]
    for group in with_:
        mask = np.full(len(issues), False)
        if group.assignees:
            # None will become "None" and will match; nobody is going to name a user "None"
            # except for to troll Athenian.
            assignees = np.char.lower(np.array(group.assignees, dtype="U"))
            mask |= np.in1d(issues["assignee"].values.astype("U"), assignees)
        if group.reporters:
            reporters = np.char.lower(np.array(group.reporters, dtype="U"))
            mask |= np.in1d(issues["reporter"].values.astype("U"), reporters)
        if group.commenters:
            commenters = np.char.lower(np.array(group.commenters, dtype="U"))
            issue_commenters = issues["commenters"]
            merged_issue_commenters = np.concatenate(issue_commenters).astype("U")
            offsets = np.zeros(len(issue_commenters) + 1, dtype=int)
            np.cumsum(issue_commenters.apply(len).values, out=offsets[1:])
            indexes = np.unique(np.searchsorted(
                offsets, np.nonzero(np.in1d(merged_issue_commenters, commenters))[0],
                side="right") - 1)
            mask[indexes] = True
        result.append(np.nonzero(mask)[0])
    return result


class _IssuesLabelSplitter:
    def __init__(self, enabled: bool, label_filter: LabelFilter):
        self._labels = np.array([None], dtype=object)
        self.enabled = enabled
        self._filter = label_filter

    @property
    def labels(self):
        return self._labels

    def __call__(self, issues: pd.DataFrame) -> List[np.ndarray]:
        if not self.enabled or issues.empty:
            return [np.arange(len(issues))]
        labels_column = issues[Issue.labels.key].tolist()
        rows_all_labels = np.repeat(np.arange(len(labels_column), dtype=int),
                                    [len(labels) for labels in labels_column])
        all_labels = np.concatenate(labels_column).astype("U")
        del labels_column
        all_labels_order = np.argsort(all_labels)
        ordered_rows_all_labels = rows_all_labels[all_labels_order]
        unique_labels, unique_counts = np.unique(all_labels[all_labels_order], return_counts=True)
        del all_labels
        groups = np.array(np.split(ordered_rows_all_labels, np.cumsum(unique_counts)),
                          dtype=object)
        unique_labels_order = np.argsort(-unique_counts)
        unique_labels = unique_labels[unique_labels_order]
        groups = groups[unique_labels_order]
        if self._filter.exclude and len(unique_labels):
            exclude = np.array(sorted(self._filter.exclude), dtype="U")
            mask = np.in1d(unique_labels, exclude, assume_unique=True, invert=True)
            unique_labels = unique_labels[mask]
            groups = groups[mask]
        if self._filter.include:
            if len(unique_labels):
                singles, multiples = LabelFilter.split(self._filter.include)
                include = set(singles)
                for labels in multiples:
                    include.update(labels)
                include = np.array(sorted(self._filter.include), dtype="U")
                mask = np.in1d(unique_labels, include, assume_unique=True)
                unique_labels = unique_labels[mask]
                groups = groups[mask]
        else:
            # no include filter => append another group of issues with empty labels
            unique_labels = np.concatenate([unique_labels, [None]])
            empty_labels_group = np.nonzero(~issues[Issue.labels.key].astype(bool).values)[0]
            groups = list(groups) + [empty_labels_group]
        if not isinstance(groups, list):
            groups = groups.tolist()
        self._labels = unique_labels
        return groups


@weight(1.5)
async def calc_histogram_jira(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate histograms over JIRA issue activities."""
    filt, time_intervals, issues, _, _ = await _calc_jira_entry(
        request, body, JIRAHistogramsRequest)
    defs = defaultdict(list)
    for h in (filt.histograms or []):
        defs[HistogramParameters(
            scale=Scale[h.scale.upper()] if h.scale is not None else None,
            bins=h.bins,
            ticks=tuple(h.ticks) if h.ticks is not None else None,
        )].append(h.metric)
    try:
        calc = JIRABinnedHistogramCalculator(defs.values(), filt.quantiles or [0, 1])
    except KeyError as e:
        raise ResponseError(InvalidRequestError("Unsupported metric: %s" % e)) from None
    with_groups = group_to_indexes(issues, partial(_split_issues_by_with, filt.with_))
    histograms = calc(issues, time_intervals, with_groups, defs)
    result = []
    for metrics, def_hists in zip(defs.values(), histograms):
        for with_, with_hists in zip(filt.with_ or [None], def_hists):
            for metric, histogram in zip(metrics, with_hists[0][0]):
                result.append(CalculatedJIRAHistogram(
                    with_=with_,
                    metric=metric,
                    scale=histogram.scale.name.lower(),
                    ticks=histogram.ticks,
                    frequencies=histogram.frequencies,
                    interquartile=Interquartile(*histogram.interquartile),
                ))
    return model_response(result)
