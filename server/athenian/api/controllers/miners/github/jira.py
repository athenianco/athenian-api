from datetime import datetime
from itertools import chain
import pickle
from typing import Collection, List, Optional

import aiomcache
import databases
import numpy as np
import pandas as pd
from sqlalchemy import sql
from sqlalchemy.orm import aliased
from sqlalchemy.sql import ClauseElement

from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.models.metadata.github import NodePullRequestJiraIssues, PullRequest
from athenian.api.models.metadata.jira import Component, Issue
from athenian.api.tracing import sentry_span


async def generate_jira_prs_query(filters: List[ClauseElement],
                                  jira: JIRAFilter,
                                  mdb: databases.Database,
                                  columns=PullRequest) -> sql.Select:
    """Produce SQLAlchemy statement to fetch PRs that satisfy JIRA conditions."""
    assert jira
    if columns is PullRequest:
        columns = [PullRequest]
    _map = aliased(NodePullRequestJiraIssues, name="m")
    _issue = aliased(Issue, name="j")
    if jira.labels:
        all_labels = set()
        for label in chain(jira.labels.include, jira.labels.exclude):
            for part in label.split(","):
                all_labels.add(part.strip())
        rows = await mdb.fetch_all(sql.select([Component.id, Component.name]).where(sql.and_(
            sql.func.lower(Component.name).in_(all_labels),
            Component.acc_id == jira.account,
        )))
        components = {r[1].lower(): r[0] for r in rows}
    if mdb.url.dialect in ("postgres", "postgresql"):
        if jira.labels.include:
            singles, multiples = LabelFilter.split(jira.labels.include)
            or_items = []
            if singles:
                or_items.append(_issue.labels.overlap(singles))
            or_items.extend(_issue.labels.contains(m) for m in multiples)
            if components:
                if singles:
                    cinc = [components[s] for s in singles if s in components]
                    if cinc:
                        or_items.append(_issue.components.overlap(cinc))
                if multiples:
                    cinc = [[components[c] for c in g if c in components] for g in multiples]
                    or_items.extend(_issue.components.contains(g) for g in cinc if g)
            filters.append(sql.or_(*or_items))
        if jira.labels.exclude:
            filters.append(sql.not_(_issue.labels.overlap(jira.labels.exclude)))
            if components:
                filters.append(sql.not_(_issue.components.overlap(
                    [components[s] for s in jira.labels.exclude if s in components])))
    else:
        # neither 100% correct nor efficient, but enough for local development
        if jira.labels.include:
            or_items = []
            singles, multiples = LabelFilter.split(jira.labels.include)
            or_items.extend(_issue.labels.like("%%%s%%" % s) for s in singles)
            or_items.extend(
                sql.and_(*(_issue.labels.like("%%%s%%" % s) for s in g)) for g in multiples)
            if components:
                if singles:
                    or_items.extend(
                        _issue.components.like("%%%s%%" % components[s])
                        for s in singles if s in components)
                if multiples:
                    or_items.extend(
                        sql.and_(*(_issue.components.like("%%%s%%" % components[s]) for s in g
                                   if s in components))
                        for g in multiples)
            filters.append(sql.or_(*or_items))
        if jira.labels.exclude:
            filters.append(sql.not_(sql.or_(*(
                _issue.labels.like("%%%s%%" % s) for s in jira.labels.exclude))))
            if components:
                filters.append(sql.not_(sql.or_(*(
                    _issue.components.like("%%%s%%" % components[s])
                    for s in jira.labels.exclude if s in components))))
    if jira.issue_types:
        filters.append(sql.func.lower(_issue.type).in_(jira.issue_types))
    if not jira.epics:
        return sql.select(columns).select_from(sql.join(
            PullRequest, sql.join(_map, _issue, _map.jira_id == _issue.id),
            PullRequest.node_id == _map.node_id,
        )).where(sql.and_(*filters))
    _issue_epic = aliased(Issue, name="e")
    filters.append(_issue_epic.key.in_(jira.epics))
    return sql.select(columns).select_from(sql.join(
        PullRequest, sql.join(
            _map, sql.join(_issue, _issue_epic, _issue.epic_id == _issue_epic.id),
            _map.jira_id == _issue.id),
        PullRequest.node_id == _map.node_id,
    )).where(sql.and_(*filters))


@sentry_span
@cached(
    exptime=5 * 60,  # 5 minutes
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, exclude_inactive, priorities, reporters, assignees, commenters, **_: (  # noqa
        time_from.timestamp(), time_to.timestamp(),
        exclude_inactive,
        ",".join(sorted(priorities)),
        ",".join(sorted(reporters)),
        ",".join(sorted(assignees)),
        ",".join(sorted(commenters)),
    ),
)
async def fetch_jira_issues(account: int,
                            time_from: datetime,
                            time_to: datetime,
                            exclude_inactive: bool,
                            priorities: Collection[str],
                            reporters: Collection[str],
                            assignees: Collection[str],
                            commenters: Collection[str],
                            mdb: databases.Database,
                            cache: Optional[aiomcache.Client],
                            ) -> pd.DataFrame:
    """
    Load JIRA issues following the specified filters.

    The aggregation is OR between the participation roles.

    :param time_from: Issues should not be resolved before this timestamp.
    :param time_to: Issues should be opened before this timestamp.
    :param exclude_inactive: Issues must be updated after `time_from`.
    :param priorities: List of lower-case priorities.
    :param reporters: List of lower-case issue reporters.
    :param assignees: List of lower-case issue assignees.
    :param commenters: List of lower-case issue commenters.
    """
    postgres = mdb.url.dialect in ("postgres", "postgresql")
    columns = [
        Issue.created, Issue.resolved, Issue.updated, Issue.priority_name, Issue.epic_id,
        Issue.status,
    ]
    and_filters = [
        Issue.acc_id == account,
        sql.func.coalesce(Issue.resolved >= time_from, sql.true()),
        Issue.created < time_to,
    ]
    if exclude_inactive:
        and_filters.append(Issue.updated >= time_from)
    if priorities:
        and_filters.append(sql.func.lower(Issue.priority_name).in_(priorities))
    or_filters = []
    if reporters and (postgres or not commenters):
        or_filters.append(sql.func.lower(Issue.reporter_display_name).in_(reporters))
    if assignees and (postgres or not commenters):
        or_filters.append(sql.func.lower(Issue.assignee_display_name).in_(assignees))
    if commenters:
        if postgres:
            or_filters.append(Issue.commenters_display_names.overlap(commenters))
        else:
            if reporters:
                columns.append(Issue.reporter_display_name)
            if assignees:
                columns.append(Issue.assignee_display_name)
            columns.append(Issue.commenters_display_names)
    if or_filters:
        query = sql.union(*(sql.select(columns).where(sql.and_(or_filter, *and_filters))
                            for or_filter in or_filters))
    else:
        query = sql.select(columns).where(sql.and_(*and_filters))
    df = await read_sql_query(query, mdb, columns)
    if postgres or not commenters:
        return df
    passed = np.full(len(df), False)
    if reporters:
        passed |= df[Issue.reporter_display_name.key].str.lower().isin(reporters).values
    if assignees:
        passed |= df[Issue.assignee_display_name.key].str.lower().isin(assignees).values
    # don't go hardcore vectorized here, we don't have to with SQLite
    for i, issue_commenters in enumerate(df[Issue.commenters_display_names.key].values):
        if len(np.intersect1d(issue_commenters, commenters)):
            passed[i] = True
    return df.take(np.nonzero(passed)[0])
