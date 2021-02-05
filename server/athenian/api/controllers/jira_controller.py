from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from functools import partial
from itertools import chain, groupby
import logging
from operator import itemgetter
from typing import List, Optional, Tuple, Type, Union

from aiohttp import web
import numpy as np
import pandas as pd
from sqlalchemy import and_, outerjoin, select, true, union_all
from sqlalchemy.sql.functions import coalesce

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.controllers.account import get_account_repositories, get_metadata_account_ids
from athenian.api.controllers.datetime_utils import split_to_time_intervals
from athenian.api.controllers.features.histogram import HistogramParameters, Scale
from athenian.api.controllers.features.jira.issue_metrics import JIRABinnedHistogramCalculator, \
    JIRABinnedMetricCalculator
from athenian.api.controllers.features.metric_calculator import group_to_indexes
from athenian.api.controllers.jira import get_jira_installation
from athenian.api.controllers.miners.filters import LabelFilter
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.jira.issue import fetch_jira_issues
from athenian.api.controllers.settings import Settings
from athenian.api.models.metadata.jira import AthenianIssue, Component, Issue, IssueType, \
    Priority, User
from athenian.api.models.state.models import MappedJIRAIdentity
from athenian.api.models.web import CalculatedJIRAHistogram, CalculatedJIRAMetricValues, \
    CalculatedLinearMetricValues, FilterJIRAStuff, FoundJIRAStuff, Interquartile, \
    InvalidRequestError, JIRAEpic, JIRAEpicChild, JIRAFilterReturn, JIRAFilterWith, \
    JIRAHistogramsRequest, JIRAIssueType, JIRALabel, JIRAMetricsRequest, JIRAPriority, JIRAUser
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError
from athenian.api.tracing import sentry_span


@weight(0.5)
async def filter_jira_stuff(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find JIRA epics and labels used in the specified date interval."""
    try:
        filt = FilterJIRAStuff.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    if filt.date_from is None or filt.date_to is None:
        if (filt.date_from is None) != (filt.date_to is None):
            raise ResponseError(InvalidRequestError(
                ".date_from",
                detail="date_from and date_to must be either both not null or both null"))
        time_from = time_to = None
    else:
        time_from, time_to = filt.resolve_time_from_and_to()
    return_ = set(filt.return_ or FoundJIRAStuff.openapi_types)
    jira_ids = await get_jira_installation(filt.account, request.sdb, request.mdb, request.cache)
    mdb = request.mdb
    sdb = request.sdb
    log = logging.getLogger("%s.filter_jira_stuff" % metadata.__package__)

    def append_time_filters(filters: list) -> None:
        if time_to is not None:
            filters.append(Issue.created < time_to)
        if time_from is not None:
            filters.append(coalesce(AthenianIssue.resolved >= time_from, true()))
            if filt.exclude_inactive:
                filters.append(Issue.updated >= time_from)

    @sentry_span
    async def epic_flow() -> Optional[List[JIRAEpic]]:
        if JIRAFilterReturn.EPICS not in return_:
            return None
        filters = [
            Issue.acc_id == jira_ids[0],
            Issue.project_id.in_(jira_ids[1]),
            Issue.type == "Epic",
        ]
        append_time_filters(filters)
        epic_rows = await mdb.fetch_all(
            select([Issue.id, Issue.key, Issue.title, Issue.updated])
            .select_from(outerjoin(Issue, AthenianIssue, and_(Issue.acc_id == AthenianIssue.acc_id,
                                                              Issue.id == AthenianIssue.id)))
            .where(and_(*filters)))
        epic_ids = [r[Issue.id.key] for r in epic_rows]
        children_rows = await mdb.fetch_all(
            select([Issue.epic_id, Issue.key, Issue.status, Issue.type,
                    AthenianIssue.work_began, AthenianIssue.resolved, Issue.updated])
            .select_from(outerjoin(Issue, AthenianIssue, and_(Issue.acc_id == AthenianIssue.acc_id,
                                                              Issue.id == AthenianIssue.id)))
            .where(and_(Issue.epic_id.in_(epic_ids),
                        Issue.acc_id == jira_ids[0]))
            .order_by(Issue.epic_id))
        children = {k: sorted((i[1] for i in g), key=lambda c: c[0]) for k, g in groupby(
            ((r[0], [r[i] for i in range(1, len(r))]) for r in children_rows), key=itemgetter(0))}
        if mdb.url.dialect == "sqlite":
            for cs in children.values():
                for c in cs:
                    for i in (-3, -2):
                        if c[i] is not None:
                            c[i] = c[i].replace(tzinfo=timezone.utc)
        epics = sorted(JIRAEpic(id=r[Issue.key.key],
                                title=r[Issue.title.key],
                                updated=max(chain(
                                    (r[Issue.updated.key],),
                                    (c[-1] for c in children.get(r[Issue.id.key], [])))),
                                children=[JIRAEpicChild(*c[:-1]) for c in children.get(
                                    r[Issue.id.key], [])])
                       for r in epic_rows)
        if mdb.url.dialect == "sqlite":
            for epic in epics:
                epic.updated = epic.updated.replace(tzinfo=timezone.utc)
        return epics

    @sentry_span
    async def issue_flow():
        if not {JIRAFilterReturn.LABELS, JIRAFilterReturn.ISSUE_TYPES, JIRAFilterReturn.PRIORITIES,
                JIRAFilterReturn.STATUSES, JIRAFilterReturn.USERS}.intersection(return_):
            return None, None, None, None
        filters = [
            Issue.acc_id == jira_ids[0],
            Issue.project_id.in_(jira_ids[1]),
        ]
        append_time_filters(filters)
        property_rows = await mdb.fetch_all(
            select([Issue.id,
                    Issue.project_id,
                    Issue.labels,
                    Issue.components,
                    Issue.type,
                    Issue.updated,
                    Issue.assignee_id,
                    Issue.reporter_id,
                    Issue.commenters_ids,
                    Issue.priority_id])
            .select_from(outerjoin(Issue, AthenianIssue, and_(Issue.acc_id == AthenianIssue.acc_id,
                                                              Issue.id == AthenianIssue.id)))
            .where(and_(*filters)))
        if JIRAFilterReturn.LABELS in return_:
            components = Counter(chain.from_iterable(
                (r[Issue.components.key] or ()) for r in property_rows))
        else:
            components = None
        if JIRAFilterReturn.USERS in return_:
            people = set(r[Issue.reporter_id.key] for r in property_rows)
            people.update(r[Issue.assignee_id.key] for r in property_rows)
            people.update(chain.from_iterable(
                (r[Issue.commenters_ids.key] or []) for r in property_rows))
        else:
            people = None
        if JIRAFilterReturn.PRIORITIES in return_:
            priorities = set(r[Issue.priority_id.key] for r in property_rows)
        else:
            priorities = None
        if JIRAFilterReturn.ISSUE_TYPES in return_:
            issue_type_counts = Counter(r[Issue.type.key] for r in property_rows)
            issue_type_projects = defaultdict(set)
            for r in property_rows:
                issue_type_projects[r[Issue.project_id.key]].add(r[Issue.type.key])
            project_counts = Counter(r[Issue.project_id.key] for r in property_rows)
        else:
            issue_type_counts = issue_type_projects = project_counts = None

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
                    MappedJIRAIdentity.account_id == filt.account,
                    MappedJIRAIdentity.jira_user_id.in_(people),
                )))

        @sentry_span
        async def fetch_priorities():
            if JIRAFilterReturn.PRIORITIES not in return_:
                return []
            return await mdb.fetch_all(
                select([Priority.name, Priority.icon_url, Priority.rank, Priority.status_color])
                .where(and_(
                    Priority.id.in_(priorities),
                    Priority.acc_id == jira_ids[0],
                ))
                .order_by(Priority.rank))

        @sentry_span
        async def fetch_types():
            if JIRAFilterReturn.ISSUE_TYPES not in return_:
                return []
            queries = [
                select([IssueType.name, IssueType.project_id, IssueType.icon_url])
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
                labels = Counter(chain.from_iterable(
                    (r[Issue.labels.key] or ()) for r in property_rows))
                labels = {k: JIRALabel(title=k, kind="regular", issues_count=v)
                          for k, v in labels.items()}
                for row in property_rows:
                    updated = row[Issue.updated.key]
                    for label in (row[Issue.labels.key] or ()):
                        label = labels[label]  # type: JIRALabel
                        if label.last_used is None or label.last_used < updated:
                            label.last_used = updated
                if mdb.url.dialect == "sqlite":
                    for label in labels.values():
                        label.last_used = label.last_used.replace(tzinfo=timezone.utc)
            else:
                labels = None
            return labels

        component_names, users, mapped_identities, priorities, issue_types, labels = await gather(
            fetch_components(), fetch_users(), fetch_mapped_identities(), fetch_priorities(),
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
        priorities = [JIRAPriority(name=row[Priority.name.key],
                                   image=row[Priority.icon_url.key],
                                   rank=row[Priority.rank.key],
                                   color=row[Priority.status_color.key])
                      for row in priorities] or None
        # take the issue type URL corresponding to the project with the most issues
        max_issue_types = {}
        for row in issue_types:
            name = row[IssueType.name.key]
            max_count, _ = max_issue_types.get(name, (0, ""))
            if (count := project_counts[row[IssueType.project_id.key]]) > max_count:
                max_issue_types[name] = count, row[IssueType.icon_url.key]
        issue_types = [JIRAIssueType(name=name,
                                     image=image,
                                     count=issue_type_counts[name],
                                     project="<not implemented>")
                       for name, (_, image) in sorted(max_issue_types.items())] or None
        if JIRAFilterReturn.LABELS in return_:
            for row in property_rows:
                updated = row[Issue.updated.key]
                for component in (row[Issue.components.key] or ()):
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
        return labels, users, issue_types, priorities

    epics, (labels, users, types, priorities) = await gather(epic_flow(), issue_flow(), op="mdb")
    return model_response(FoundJIRAStuff(
        epics=epics,
        labels=labels,
        issue_types=types,
        priorities=priorities,
        users=users,
    ))


async def _calc_jira_entry(request: AthenianWebRequest,
                           body: dict,
                           model: Union[Type[JIRAMetricsRequest], Type[JIRAHistogramsRequest]],
                           ) -> Tuple[Union[JIRAMetricsRequest, JIRAHistogramsRequest],
                                      List[List[datetime]],
                                      pd.DataFrame,
                                      timedelta]:
    try:
        filt = model.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e))) from None
    tasks = [
        get_account_repositories(filt.account, request.sdb),
        get_jira_installation(filt.account, request.sdb, request.mdb, request.cache),
        get_metadata_account_ids(filt.account, request.sdb, request.cache),
    ]
    repos, jira_ids, meta_ids = await gather(*tasks, op="sdb")
    time_intervals, tzoffset = split_to_time_intervals(
        filt.date_from, filt.date_to, getattr(filt, "granularities", ["all"]), filt.timezone)
    tasks = [
        extract_branches(
            [r.split("/", 1)[1] for r in repos], meta_ids, request.mdb, request.cache),
        Settings.from_request(
            request, filt.account).list_release_matches(repos),
    ]
    (_, default_branches), release_settings = await gather(
        *tasks, op="branches and release settings")
    reporters = list(set(chain.from_iterable(
        ([p.lower() for p in (g.reporters or [])]) for g in (filt.with_ or []))))
    assignees = list(set(chain.from_iterable(
        ([(p.lower() if p is not None else None) for p in (g.assignees or [])])
        for g in (filt.with_ or []))))
    commenters = list(set(chain.from_iterable(
        ([p.lower() for p in (g.commenters or [])]) for g in (filt.with_ or []))))
    issues = await fetch_jira_issues(
        jira_ids,
        time_intervals[0][0], time_intervals[0][-1], filt.exclude_inactive,
        LabelFilter.from_iterables(filt.labels_include, filt.labels_exclude),
        [p.lower() for p in (filt.priorities or [])],
        [p.lower() for p in (filt.types or [])],
        filt.epics or [],
        reporters, assignees, commenters, len(filt.with_ or []) > 1,
        default_branches, release_settings,
        meta_ids, request.mdb, request.pdb, request.cache,
    )
    return filt, time_intervals, issues, tzoffset


@weight(2.5)
async def calc_metrics_jira_linear(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics over JIRA issue activities."""
    filt, time_intervals, issues, tzoffset = await _calc_jira_entry(
        request, body, JIRAMetricsRequest)
    calc = JIRABinnedMetricCalculator(filt.metrics, filt.quantiles or [0, 1])
    label_splitter = _IssuesLabelSplitter(filt.group_by_jira_label)
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
    def __init__(self, enabled: bool):
        self._labels = np.array([None], dtype=object)
        self.enabled = enabled

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
        self._labels = unique_labels[unique_labels_order]
        groups = groups[unique_labels_order]
        return groups


@weight(1.5)
async def calc_histogram_jira(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate histograms over JIRA issue activities."""
    filt, time_intervals, issues, _ = await _calc_jira_entry(request, body, JIRAHistogramsRequest)
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


async def filter_jira_epics(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find JIRA epics which were active in the specified date interval."""
    raise NotImplementedError
