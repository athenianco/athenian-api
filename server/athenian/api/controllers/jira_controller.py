from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import chain, groupby
import logging
from operator import itemgetter
from typing import List, Optional, Tuple, Type, Union

from aiohttp import web
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, outerjoin, select, true
from sqlalchemy.sql.functions import coalesce

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.controllers.account import get_account_repositories, get_user_account_status
from athenian.api.controllers.datetime_utils import split_to_time_intervals
from athenian.api.controllers.features.histogram import HistogramParameters, Scale
from athenian.api.controllers.features.jira.issue_metrics import JIRABinnedHistogramCalculator, \
    JIRABinnedMetricCalculator
from athenian.api.controllers.jira import get_jira_installation
from athenian.api.controllers.miners.filters import LabelFilter
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.jira.issue import fetch_jira_issues
from athenian.api.controllers.settings import Settings
from athenian.api.models.metadata.jira import AthenianIssue, Component, Issue, Priority, User
from athenian.api.models.web import CalculatedJIRAHistogram, CalculatedJIRAMetricValues, \
    CalculatedLinearMetricValues, FilterJIRAStuff, FoundJIRAStuff, Interquartile, \
    InvalidRequestError, JIRAEpic, JIRAHistogramsRequest, JIRALabel, JIRAMetricsRequest, \
    JIRAPriority, JIRAUser
from athenian.api.models.web.jira_epic_child import JIRAEpicChild
from athenian.api.models.web.jira_metrics_request_with import JIRAMetricsRequestWith
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError
from athenian.api.tracing import sentry_span


async def filter_jira_stuff(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find JIRA epics and labels used in the specified date interval."""
    try:
        filt = FilterJIRAStuff.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    time_from, time_to = filt.resolve_time_from_and_to()
    with sentry_sdk.start_span(op="sdb"):
        async with request.sdb.connection() as conn:
            await get_user_account_status(request.uid, filt.account, conn, request.cache)
            jira_ids = await get_jira_installation(
                filt.account, request.sdb, request.mdb, request.cache)
    mdb = request.mdb
    log = logging.getLogger("%s.filter_jira_stuff" % metadata.__package__)

    @sentry_span
    async def epic_flow():
        filters = [
            Issue.acc_id == jira_ids[0],
            Issue.project_id.in_(jira_ids[1]),
            Issue.type == "Epic",
            Issue.created < time_to,
            coalesce(AthenianIssue.resolved >= time_from, true()),
        ]
        if filt.exclude_inactive:
            filters.append(Issue.updated >= time_from)
        epic_rows = await mdb.fetch_all(
            select([Issue.id, Issue.key, Issue.title, Issue.updated])
            .select_from(outerjoin(Issue, AthenianIssue, and_(Issue.acc_id == AthenianIssue.acc_id,
                                                              Issue.id == AthenianIssue.id)))
            .where(and_(*filters)))
        epic_ids = [r[Issue.id.key] for r in epic_rows]
        children_rows = await mdb.fetch_all(
            select([Issue.epic_id, Issue.key, Issue.status, Issue.type, Issue.updated])
            .where(Issue.epic_id.in_(epic_ids))
            .order_by(Issue.epic_id))
        children = {k: [i[1] for i in g] for k, g in groupby(
            ((r[0], [r[i] for i in range(1, 5)]) for r in children_rows), key=itemgetter(0))}
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
        filters = [
            Issue.acc_id == jira_ids[0],
            Issue.project_id.in_(jira_ids[1]),
            Issue.created < time_to,
            coalesce(AthenianIssue.resolved >= time_from, true()),
        ]
        if filt.exclude_inactive:
            filters.append(Issue.updated >= time_from)
        property_rows = await mdb.fetch_all(
            select([Issue.id, Issue.labels, Issue.components, Issue.type, Issue.updated,
                    Issue.assignee_id, Issue.reporter_id, Issue.commenters_ids, Issue.priority_id])
            .select_from(outerjoin(Issue, AthenianIssue, and_(Issue.acc_id == AthenianIssue.acc_id,
                                                              Issue.id == AthenianIssue.id)))
            .where(and_(*filters)))
        components = Counter(chain.from_iterable(
            (r[Issue.components.key] or ()) for r in property_rows))
        people = set(r[Issue.reporter_id.key] for r in property_rows)
        people.update(r[Issue.assignee_id.key] for r in property_rows)
        people.update(chain.from_iterable(
            (r[Issue.commenters_ids.key] or []) for r in property_rows))
        priorities = set(r[Issue.priority_id.key] for r in property_rows)

        @sentry_span
        async def fetch_components():
            return await mdb.fetch_all(
                select([Component.id, Component.name])
                .where(and_(
                    Component.id.in_(components),
                    Component.acc_id == jira_ids[0],
                )))

        @sentry_span
        async def fetch_users():
            return await mdb.fetch_all(
                select([User.display_name, User.avatar_url, User.type])
                .where(and_(
                    User.id.in_(people),
                    User.acc_id == jira_ids[0],
                ))
                .order_by(User.display_name))

        @sentry_span
        async def fetch_priorities():
            return await mdb.fetch_all(
                select([Priority.name, Priority.icon_url, Priority.rank, Priority.status_color])
                .where(and_(
                    Priority.id.in_(priorities),
                    Priority.acc_id == jira_ids[0],
                ))
                .order_by(Priority.rank))

        @sentry_span
        async def main_flow():
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
            types = sorted(set(r[Issue.type.key] for r in property_rows))
            if mdb.url.dialect == "sqlite":
                for label in labels.values():
                    label.last_used = label.last_used.replace(tzinfo=timezone.utc)
            return labels, types

        component_names, users, priorities, (labels, types) = await gather(
            fetch_components(), fetch_users(), fetch_priorities(), main_flow())
        components = {
            row[0]: JIRALabel(title=row[1], kind="component", issues_count=components[row[0]])
            for row in component_names
        }
        users = [JIRAUser(avatar=row[User.avatar_url.key],
                          name=row[User.display_name.key],
                          type=row[User.type.key])
                 for row in users]
        priorities = [JIRAPriority(name=row[Priority.name.key],
                                   image=row[Priority.icon_url.key],
                                   rank=row[Priority.rank.key],
                                   color=row[Priority.status_color.key])
                      for row in priorities]
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
        return labels, users, types, priorities

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
                                      pd.DataFrame]:
    try:
        filt = model.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e))) from None
    tasks = [
        get_user_account_status(request.uid, filt.account, request.sdb, request.cache),
        get_account_repositories(filt.account, request.sdb),
        get_jira_installation(filt.account, request.sdb, request.mdb, request.cache),
    ]
    status, repos, jira_ids = await gather(*tasks, op="sdb")
    time_intervals, _ = split_to_time_intervals(
        filt.date_from, filt.date_to, getattr(filt, "granularities", ["all"]), filt.timezone)
    tasks = [
        extract_branches([r.split("/", 1)[1] for r in repos], request.mdb, request.cache),
        Settings.from_request(request, filt.account).list_release_matches(repos),
    ]
    (_, default_branches), release_settings = await gather(
        *tasks, op="branches and release settings")
    reporters = list(set(chain.from_iterable(
        ([p.lower() for p in (g.reporters or [])]) for g in (filt.with_ or []))))
    assignees = list(set(chain.from_iterable(
        ([p.lower() for p in (g.assignees or [])]) for g in (filt.with_ or []))))
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
        request.mdb, request.pdb, request.cache,
    )
    return filt, time_intervals, issues


async def calc_metrics_jira_linear(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics over JIRA issue activities."""
    filt, time_intervals, issues = await _calc_jira_entry(request, body, JIRAMetricsRequest)
    calc = JIRABinnedMetricCalculator(filt.metrics, filt.quantiles or [0, 1])
    with_groups = _split_issues_by_with(issues, filt.with_)
    metric_values = calc(issues, time_intervals, with_groups)
    mets = list(chain.from_iterable((
        CalculatedJIRAMetricValues(granularity=granularity, with_=group, values=[
            CalculatedLinearMetricValues(date=dt.date(),
                                         values=[v.value for v in vals],
                                         confidence_mins=[v.confidence_min for v in vals],
                                         confidence_maxs=[v.confidence_max for v in vals],
                                         confidence_scores=[v.confidence_score() for v in vals])
            for dt, vals in zip(ts, ts_values)
        ])
        for granularity, ts, ts_values in zip(filt.granularities, time_intervals,
                                              group_metric_values)
    ) for group, group_metric_values in zip(filt.with_ or [None], metric_values)))
    return model_response(mets)


def _split_issues_by_with(issues: pd.DataFrame,
                          with_: Optional[List[JIRAMetricsRequestWith]],
                          ) -> List[np.ndarray]:
    result = []
    if len(with_ or []) < 2:
        return [np.arange(len(issues))]
    for group in with_:
        mask = np.full(len(issues), False)
        if group.assignees:
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


async def calc_histogram_jira(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate histograms over JIRA issue activities."""
    filt, time_intervals, issues = await _calc_jira_entry(request, body, JIRAHistogramsRequest)
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
    with_groups = _split_issues_by_with(issues, filt.with_)
    histograms = calc(issues, time_intervals, with_groups, [k.__dict__ for k in defs])
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
