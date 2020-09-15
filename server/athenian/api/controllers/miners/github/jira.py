from itertools import chain
from typing import List

import databases
from sqlalchemy import sql
from sqlalchemy.orm import aliased
from sqlalchemy.sql import ClauseElement

from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.models.metadata.github import NodePullRequestJiraIssues, PullRequest
from athenian.api.models.metadata.jira import Component, Issue


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
