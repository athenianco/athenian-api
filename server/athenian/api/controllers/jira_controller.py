import asyncio
from collections import Counter
from datetime import timezone
from itertools import chain, groupby
import logging
import marshal
from operator import itemgetter
from typing import Optional

from aiohttp import web
import aiomcache
import sentry_sdk
from sqlalchemy import and_, or_, select

from athenian.api import metadata
from athenian.api.cache import cached, max_exptime
from athenian.api.controllers.account import get_account_repositories, get_user_account_status
from athenian.api.controllers.datetime_utils import split_to_time_intervals
from athenian.api.controllers.features.github.jira_metrics import JIRABinnedMetricCalculator
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.jira.issue import fetch_jira_issues
from athenian.api.controllers.settings import Settings
from athenian.api.models.metadata.jira import Component, Issue, User
from athenian.api.models.state.models import AccountJiraInstallation
from athenian.api.models.web import CalculatedJIRAMetricValues, CalculatedLinearMetricValues, \
    FilterJIRAStuff, FoundJIRAStuff, \
    InvalidRequestError, \
    JIRAEpic, JIRALabel, JIRAMetricsRequest, JIRAUser, NoSourceDataError
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError
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
                                cache: Optional[aiomcache.Client]) -> int:
    """Retrieve the JIRA installation ID belonging to the account or raise an exception."""
    jira_id = await sdb.fetch_val(select([AccountJiraInstallation.id])
                                  .where(AccountJiraInstallation.account_id == account))
    if jira_id is None:
        raise ResponseError(NoSourceDataError(
            detail="JIRA has not been installed to the metadata yet."))
    return jira_id


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
            jira_id = await get_jira_installation(filt.account, request.sdb, request.cache)
    mdb = request.mdb
    log = logging.getLogger("%s.filter_jira_stuff" % metadata.__package__)

    @sentry_span
    async def epic_flow():
        epic_rows = await mdb.fetch_all(
            select([Issue.id, Issue.key, Issue.title, Issue.updated])
            .where(and_(Issue.acc_id == jira_id,
                        Issue.type == "Epic",
                        Issue.created < time_to,
                        or_(Issue.resolved.is_(None), Issue.resolved >= time_from),
                        )))
        epic_ids = [r[Issue.id.key] for r in epic_rows]
        children_rows = await mdb.fetch_all(
            select([Issue.epic_id, Issue.key])
            .where(Issue.epic_id.in_(epic_ids))
            .order_by(Issue.epic_id))
        children = {k: [i[1] for i in g] for k, g in groupby(
            ((r[0], r[1]) for r in children_rows), key=itemgetter(0))}
        epics = sorted(JIRAEpic(id=r[Issue.key.key],
                                title=r[Issue.title.key],
                                updated=r[Issue.updated.key],
                                children=children.get(r[Issue.id.key], []))
                       for r in epic_rows)
        if mdb.url.dialect == "sqlite":
            for epic in epics:
                epic.updated = epic.updated.replace(tzinfo=timezone.utc)
        return epics

    @sentry_span
    async def issue_flow():
        property_rows = await mdb.fetch_all(
            select([Issue.id, Issue.labels, Issue.components, Issue.type, Issue.updated,
                    Issue.assignee_id, Issue.reporter_id, Issue.commenters_ids, Issue.priority_id])
            .where(and_(Issue.acc_id == jira_id,
                        Issue.created < time_to,
                        or_(Issue.resolved.is_(None), Issue.resolved >= time_from),
                        )))
        components = Counter(chain.from_iterable(
            (r[Issue.components.key] or ()) for r in property_rows))
        people = set(r[Issue.reporter_id.key] for r in property_rows)
        people.update(r[Issue.assignee_id.key] for r in property_rows)
        people.update(chain.from_iterable(r[Issue.commenters_ids.key] for r in property_rows))
        # priorities = set(r[Issue.priority_id.key] for r in property_rows)

        @sentry_span
        async def fetch_components():
            return await mdb.fetch_all(
                select([Component.id, Component.name])
                .where(and_(
                    Component.id.in_(components),
                    Component.acc_id == jira_id,
                )))

        @sentry_span
        async def fetch_users():
            return await mdb.fetch_all(
                select([User.display_name, User.avatar_url, User.type])
                .where(and_(
                    User.id.in_(people),
                    User.acc_id == jira_id,
                ))
                .order_by(User.display_name))

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

        component_names, users, issues = await asyncio.gather(
            fetch_components(), fetch_users(), main_flow(), return_exceptions=True)
        for e in (components, users, issues):
            if isinstance(e, Exception):
                raise e from None
        labels, types = issues

        components = {
            row[0]: JIRALabel(title=row[1], kind="component", issues_count=components[row[0]])
            for row in component_names
        }
        users = [JIRAUser(avatar=row[User.avatar_url.key],
                          name=row[User.display_name.key],
                          type=row[User.type.key])
                 for row in users]
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
        return labels, users, types

    with sentry_sdk.start_span(op="mdb"):
        epics, issues = await asyncio.gather(epic_flow(), issue_flow(), return_exceptions=True)
        for r in (epics, issues):
            if isinstance(r, Exception):
                raise r from None
    labels, users, types = issues
    return model_response(FoundJIRAStuff(
        epics=epics,
        labels=labels,
        issue_types=types,
        priorities=[],
        users=users,
    ))


async def calc_metrics_jira_linear(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate metrics over JIRA issue activities."""
    try:
        filt = JIRAMetricsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    with sentry_sdk.start_span(op="sdb"):
        tasks = [
            get_user_account_status(request.uid, filt.account, request.sdb, request.cache),
            get_account_repositories(filt.account, request.sdb),
            get_jira_installation(filt.account, request.sdb, request.cache),
        ]
        status, repos, jira_id = await asyncio.gather(*tasks, return_exceptions=True)
        for r in (status, repos, jira_id):
            if isinstance(r, Exception):
                raise r from None
    time_intervals, tzoffset = split_to_time_intervals(
        filt.date_from, filt.date_to, filt.granularities, filt.timezone)
    tasks = [
        extract_branches([r.split("/", 1)[1] for r in repos], request.mdb, request.cache),
        Settings.from_request(request, filt.account).list_release_matches(repos),
    ]
    with sentry_sdk.start_span(op="branches and release settings"):
        branches, release_settings = await asyncio.gather(*tasks, return_exceptions=True)
    for r in (branches, release_settings):
        if isinstance(r, Exception):
            raise r from None
    _, default_branches = branches

    issues = await fetch_jira_issues(
        jira_id,
        time_intervals[0][0], time_intervals[0][-1], filt.exclude_inactive,
        [p.lower() for p in (filt.priorities or [])],
        [p.lower() for p in (filt.reporters or [])],
        [p.lower() for p in (filt.assignees or [])],
        [p.lower() for p in (filt.commenters or [])],
        default_branches, release_settings,
        request.mdb, request.pdb, request.cache,
    )
    calc = JIRABinnedMetricCalculator(filt.metrics, filt.quantiles or [0, 1])
    metric_values = calc(issues, time_intervals)
    mets = [
        CalculatedJIRAMetricValues(granularity=granularity, values=[
            CalculatedLinearMetricValues(date=dt.date(),
                                         values=[v.value for v in vals],
                                         confidence_mins=[v.confidence_min for v in vals],
                                         confidence_maxs=[v.confidence_max for v in vals],
                                         confidence_scores=[v.confidence_score() for v in vals])
            for dt, vals in zip(ts, ts_values)
        ])
        for granularity, ts, ts_values in zip(filt.granularities, time_intervals, metric_values)
    ]
    return model_response(mets)
