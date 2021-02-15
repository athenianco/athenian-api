from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import chain
import pickle
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import aiomcache
import databases
import numpy as np
import pandas as pd
from sqlalchemy import sql
from sqlalchemy.orm import aliased
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement

from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.precomputed_prs import triage_by_release_match
from athenian.api.controllers.miners.types import PullRequestFacts
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import NodePullRequestJiraIssues, PullRequest
from athenian.api.models.metadata.jira import AthenianIssue, Component, Epic, Issue
from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import DatabaseLike


async def generate_jira_prs_query(filters: List[ClauseElement],
                                  jira: JIRAFilter,
                                  mdb: databases.Database,
                                  cache: Optional[aiomcache.Client],
                                  columns=PullRequest,
                                  seed=PullRequest,
                                  on=(PullRequest.node_id, PullRequest.acc_id),
                                  ) -> sql.Select:
    """
    Produce SQLAlchemy statement to fetch PRs that satisfy JIRA conditions.

    :param filters: Extra WHERE conditions.
    :param columns: SELECT these columns.
    :param seed: JOIN with this object.
    :param on: JOIN by these two columns: node ID-like and acc_id-like.
    """
    assert jira
    if columns is PullRequest:
        columns = [PullRequest]
    _map = aliased(NodePullRequestJiraIssues, name="m")
    if jira.unmapped:
        return sql.select(columns).select_from(sql.outerjoin(
            seed, _map, sql.and_(on[0] == _map.node_id, on[1] == _map.node_acc),
        )).where(sql.and_(*filters, _map.node_id.is_(None)))
    _issue = aliased(Issue, name="j")
    filters.append(_issue.acc_id == jira.account)
    filters.append(_issue.project_id.in_(jira.projects))
    if jira.labels:
        components = await _load_components(jira.labels, jira.account, mdb, cache)
        _append_label_filters(
            jira.labels, components, mdb.url.dialect in ("postgres", "postgresql"), filters,
            model=_issue)
    if jira.issue_types:
        filters.append(sql.func.lower(_issue.type).in_(jira.issue_types))
    if not jira.epics:
        return sql.select(columns).select_from(sql.join(
            seed, sql.join(_map, _issue, _map.jira_id == _issue.id),
            sql.and_(on[0] == _map.node_id, on[1] == _map.node_acc),
        )).where(sql.and_(*filters))
    _issue_epic = aliased(Issue, name="e")
    filters.append(_issue_epic.key.in_(jira.epics))
    return sql.select(columns).select_from(sql.join(
        seed, sql.join(
            _map, sql.join(_issue, _issue_epic, _issue.epic_id == _issue_epic.id),
            _map.jira_id == _issue.id),
        sql.and_(on[0] == _map.node_id, on[1] == _map.node_acc),
    )).where(sql.and_(*filters))


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda labels, account, **_: (labels, account),
    refresh_on_access=True,
)
async def _load_components(labels: LabelFilter,
                           account: int,
                           mdb: databases.Database,
                           cache: Optional[aiomcache.Client],
                           ) -> Dict[str, str]:
    all_labels = set()
    for label in chain(labels.include, labels.exclude):
        for part in label.split(","):
            all_labels.add(part.strip())
    rows = await mdb.fetch_all(sql.select([Component.id, Component.name]).where(sql.and_(
        sql.func.lower(Component.name).in_(all_labels),
        Component.acc_id == account,
    )))
    return {r[1].lower(): r[0] for r in rows}


def _append_label_filters(labels: LabelFilter,
                          components: Dict[str, str],
                          postgres: bool,
                          filters: List[ClauseElement],
                          model=Issue):
    if postgres:
        if labels.include:
            singles, multiples = LabelFilter.split(labels.include)
            or_items = []
            if singles:
                or_items.append(model.labels.overlap(singles))
            or_items.extend(model.labels.contains(m) for m in multiples)
            if components:
                if singles:
                    cinc = [components[s] for s in singles if s in components]
                    if cinc:
                        or_items.append(model.components.overlap(cinc))
                if multiples:
                    cinc = [[components[c] for c in g if c in components] for g in multiples]
                    or_items.extend(model.components.contains(g) for g in cinc if g)
            filters.append(sql.or_(*or_items))
        if labels.exclude:
            filters.append(sql.not_(model.labels.overlap(labels.exclude)))
            if components:
                filters.append(sql.not_(model.components.overlap(
                    [components[s] for s in labels.exclude if s in components])))
    else:
        # neither 100% correct nor efficient, but enough for local development
        if labels.include:
            or_items = []
            singles, multiples = LabelFilter.split(labels.include)
            or_items.extend(model.labels.like("%%%s%%" % s) for s in singles)
            or_items.extend(
                sql.and_(*(model.labels.like("%%%s%%" % s) for s in g)) for g in multiples)
            if components:
                if singles:
                    or_items.extend(
                        model.components.like("%%%s%%" % components[s])
                        for s in singles if s in components)
                if multiples:
                    or_items.extend(
                        sql.and_(*(model.components.like("%%%s%%" % components[s]) for s in g
                                   if s in components))
                        for g in multiples)
            filters.append(sql.or_(*or_items))
        if labels.exclude:
            filters.append(sql.not_(sql.or_(*(
                model.labels.like("%%%s%%" % s) for s in labels.exclude))))
            if components:
                filters.append(sql.not_(sql.or_(*(
                    model.components.like("%%%s%%" % components[s])
                    for s in labels.exclude if s in components))))


ISSUE_PRS_BEGAN = "prs_began"
ISSUE_PRS_RELEASED = "prs_released"
ISSUE_PRS_COUNT = "prs_count"


@sentry_span
@cached(
    exptime=5 * 60,  # 5 minutes
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda installation_ids, time_from, time_to, exclude_inactive, labels, priorities, types, epics, reporters, assignees, commenters, **kwargs: (  # noqa
        installation_ids[0],
        ",".join(installation_ids[1]),
        time_from.timestamp() if time_from else "-",
        time_to.timestamp() if time_to else "-",
        exclude_inactive,
        labels,
        ",".join(sorted(priorities)),
        ",".join(sorted(types)),
        ",".join(sorted(epics)),
        ",".join(sorted(reporters)),
        ",".join(sorted((ass if ass is not None else "<None>") for ass in assignees)),
        ",".join(sorted(commenters)),
        ",".join(c.key for c in kwargs.get("extra_columns", ())),
    ),
)
async def fetch_jira_issues(installation_ids: Tuple[int, List[str]],
                            time_from: Optional[datetime],
                            time_to: Optional[datetime],
                            exclude_inactive: bool,
                            labels: LabelFilter,
                            priorities: Collection[str],
                            types: Collection[str],
                            epics: Collection[str],
                            reporters: Collection[str],
                            assignees: Collection[Optional[str]],
                            commenters: Collection[str],
                            default_branches: Dict[str, str],
                            release_settings: Dict[str, ReleaseMatchSetting],
                            meta_ids: Tuple[int, ...],
                            mdb: databases.Database,
                            pdb: databases.Database,
                            cache: Optional[aiomcache.Client],
                            extra_columns: Iterable[InstrumentedAttribute] = (),
                            ) -> pd.DataFrame:
    """
    Load JIRA issues following the specified filters.

    The aggregation is OR between the participation roles.

    :param installation_ids: JIRA installation ID and the enabled project IDs.
    :param time_from: Issues should not be resolved before this timestamp.
    :param time_to: Issues should be opened before this timestamp.
    :param exclude_inactive: Issues must be updated after `time_from`.
    :param labels: Issues must satisfy these label conditions.
    :param priorities: List of lower-case priorities.
    :param types: List of lower-case types.
    :param epics: List of required parent epic keys.
    :param reporters: List of lower-case issue reporters.
    :param assignees: List of lower-case issue assignees. None means unassigned.
    :param commenters: List of lower-case issue commenters.
    :param extra_columns: Additional `Issue` or `AthenianIssue` columns to fetch.
    """
    issues = await _fetch_issues(
        installation_ids, time_from, time_to, exclude_inactive, labels, priorities, types, epics,
        reporters, assignees, commenters, mdb, cache,
        extra_columns=extra_columns)
    pr_rows = await mdb.fetch_all(
        sql.select([NodePullRequestJiraIssues.node_id, NodePullRequestJiraIssues.jira_id])
        .where(sql.and_(NodePullRequestJiraIssues.jira_acc == installation_ids[0],
                        NodePullRequestJiraIssues.jira_id.in_(issues.index),
                        NodePullRequestJiraIssues.node_acc.in_(meta_ids))))
    pr_to_issue = {r[0]: r[1] for r in pr_rows}
    # TODO(vmarkovtsev): load the "fresh" released PRs
    released_prs = await _fetch_released_prs(pr_to_issue, default_branches, release_settings, pdb)
    unreleased_prs = pr_to_issue.keys() - released_prs.keys()
    issue_to_index = {iid: i for i, iid in enumerate(issues.index.values)}
    prs_count = np.full(len(issues.index), 0, int)
    nat = np.datetime64("nat")
    work_began = np.full(len(issues.index), nat, "datetime64[ns]")
    released = work_began.copy()

    @sentry_span
    async def released_flow():
        for issue, count in Counter(pr_to_issue.values()).items():
            prs_count[issue_to_index[issue]] = count
        ghdprf = GitHubDonePullRequestFacts
        for pr_node_id, row in released_prs.items():
            pr_created_at = row[ghdprf.pr_created_at.key]
            i = issue_to_index[pr_to_issue[pr_node_id]]
            dt = work_began[i]
            if dt != dt:
                work_began[i] = pr_created_at
            else:
                work_began[i] = min(dt, np.datetime64(pr_created_at))
            pr_released_at = row[ghdprf.pr_done_at.key]
            dt = released[i]
            if dt != dt:
                released[i] = pr_released_at
            else:
                released[i] = max(dt, np.datetime64(pr_released_at))

    if not unreleased_prs:
        await released_flow()
    else:
        pr_created_ats_and_repos, err = await gather(
            _fetch_pr_created_ats_and_repos(unreleased_prs, meta_ids, mdb), released_flow(),
            op="released and unreleased")
        prefix = PREFIXES["github"]
        for row in pr_created_ats_and_repos:
            pr_created_at = row[PullRequest.created_at.key]
            repo = row[PullRequest.repository_full_name.key]
            i = issue_to_index[pr_to_issue[row[PullRequest.node_id.key]]]
            dt = work_began[i]
            if dt != dt:
                work_began[i] = pr_created_at
            else:
                work_began[i] = min(dt, np.datetime64(pr_created_at))
            if (prefix + repo) not in release_settings:
                # deleted repository, consider the PR as force push dropped
                released[i] = work_began[i]
            else:
                released[i] = nat

    issues[ISSUE_PRS_BEGAN] = work_began
    issues[ISSUE_PRS_RELEASED] = released
    issues[ISSUE_PRS_COUNT] = prs_count
    return issues


@sentry_span
async def _fetch_pr_created_ats_and_repos(pr_node_ids: Iterable[str],
                                          meta_ids: Tuple[int, ...],
                                          mdb: databases.Database,
                                          ) -> List[Mapping[str, Union[str, datetime]]]:
    return await mdb.fetch_all(
        sql.select([PullRequest.node_id, PullRequest.created_at, PullRequest.repository_full_name])
        .where(sql.and_(PullRequest.node_id.in_(pr_node_ids),
                        PullRequest.acc_id.in_(meta_ids))))


@sentry_span
async def _fetch_released_prs(pr_node_ids: Iterable[str],
                              default_branches: Dict[str, str],
                              release_settings: Dict[str, ReleaseMatchSetting],
                              pdb: databases.Database,
                              ) -> Dict[str, Mapping[str, Any]]:
    ghdprf = GitHubDonePullRequestFacts
    released_rows = await pdb.fetch_all(
        sql.select([ghdprf.pr_node_id,
                    ghdprf.pr_created_at,
                    ghdprf.pr_done_at,
                    ghdprf.repository_full_name,
                    ghdprf.release_match])
        .where(ghdprf.pr_node_id.in_(pr_node_ids)))
    released_by_repo = defaultdict(lambda: defaultdict(dict))
    for r in released_rows:
        released_by_repo[
            r[ghdprf.repository_full_name.key]][
            r[ghdprf.release_match.key]][
            r[ghdprf.pr_node_id.key]] = r
    released_prs = {}
    ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
    prefix = PREFIXES["github"]
    for repo, matches in released_by_repo.items():
        for match, prs in matches.items():
            if (prefix + repo) not in release_settings:
                for node_id, row in prs.items():
                    try:
                        if released_prs[node_id][ghdprf.pr_done_at] < row[ghdprf.pr_done_at]:
                            released_prs[node_id] = row
                    except KeyError:
                        released_prs[node_id] = row
                continue
            dump = triage_by_release_match(repo, match, release_settings, default_branches,
                                           prefix, released_prs, ambiguous)
            if dump is None:
                continue
            dump.update(prs)
    released_prs.update(ambiguous[ReleaseMatch.tag.name])
    for node_id, row in ambiguous[ReleaseMatch.branch.name].items():
        if node_id not in released_prs:
            released_prs[node_id] = row
    return released_prs


@sentry_span
async def _fetch_issues(ids: Tuple[int, List[str]],
                        time_from: Optional[datetime],
                        time_to: Optional[datetime],
                        exclude_inactive: bool,
                        labels: LabelFilter,
                        priorities: Collection[str],
                        types: Collection[str],
                        epics: Collection[str],
                        reporters: Collection[str],
                        assignees: Collection[Optional[str]],
                        commenters: Collection[str],
                        mdb: databases.Database,
                        cache: Optional[aiomcache.Client],
                        extra_columns: Iterable[InstrumentedAttribute] = (),
                        ) -> pd.DataFrame:
    postgres = mdb.url.dialect in ("postgres", "postgresql")
    columns = [
        Issue.id,
        Issue.type,
        Issue.created,
        Issue.updated,
        AthenianIssue.work_began,
        AthenianIssue.resolved,
        Issue.priority_name,
        Issue.epic_id,
        Issue.status,
        Issue.labels,
    ]
    columns.extend(extra_columns)
    # this is backed with a DB index
    far_away_future = datetime(3000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    and_filters = [
        Issue.acc_id == ids[0],
        Issue.project_id.in_(ids[1]),
    ]
    if time_from is not None:
        and_filters.append(sql.func.coalesce(Issue.resolved, far_away_future) >= time_from)
    if time_to is not None:
        and_filters.append(Issue.created < time_to)
    if exclude_inactive and time_from is not None:
        and_filters.append(Issue.updated >= time_from)
        # FIXME(vmarkovtsev): replace with AthenianIssue.updated >= time_from from DEV-1654
    if len(priorities):
        and_filters.append(sql.func.lower(Issue.priority_name).in_(priorities))
    if len(types):
        and_filters.append(sql.func.lower(Issue.type).in_(types))
    if len(epics):
        and_filters.append(Epic.key.in_(epics))
    or_filters = []
    if labels:
        components = await _load_components(labels, ids[0], mdb, cache)
        _append_label_filters(
            labels, components, mdb.url.dialect in ("postgres", "postgresql"), and_filters)
    if reporters and (postgres or not commenters):
        or_filters.append(sql.func.lower(Issue.reporter_display_name).in_(reporters))
    if assignees and (postgres or not commenters):
        if None in assignees:
            # NULL IN (NULL) = false
            or_filters.append(Issue.assignee_display_name.is_(None))
        or_filters.append(sql.func.lower(Issue.assignee_display_name).in_(assignees))
    if commenters:
        if postgres:
            or_filters.append(Issue.commenters_display_names.overlap(commenters))
        else:
            if reporters and all(c.key != "reporter" for c in extra_columns):
                columns.append(sql.func.lower(Issue.reporter_display_name).label("reporter"))
            if assignees and all(c.key != "assignee" for c in extra_columns):
                columns.append(sql.func.lower(Issue.assignee_display_name).label("assignee"))
            if all(c.key != "commenters" for c in extra_columns):
                columns.append(Issue.commenters_display_names.label("commenters"))

    def query_start():
        seed = Issue
        if len(epics):
            seed = sql.join(Issue, Epic, sql.and_(Issue.epic_id == Epic.id,
                                                  Issue.acc_id == Epic.acc_id))
        return sql.select(columns).select_from(sql.outerjoin(
            seed, AthenianIssue, sql.and_(Issue.acc_id == AthenianIssue.acc_id,
                                          Issue.id == AthenianIssue.id)))

    if or_filters:
        if postgres:
            query = sql.union(*(query_start().where(sql.and_(or_filter, *and_filters))
                                for or_filter in or_filters))
        else:
            query = query_start().where(sql.and_(sql.or_(*or_filters), *and_filters))
    else:
        query = query_start().where(sql.and_(*and_filters))
    df = await read_sql_query(query, mdb, columns, index=Issue.id.key)
    df.sort_index(inplace=True)
    if postgres or not commenters:
        return df
    passed = np.full(len(df), False)
    if reporters:
        passed |= df["reporter"].isin(reporters).values
    if assignees:
        passed |= df["assignee"].isin(assignees).values
    # don't go hardcore vectorized here, we don't have to with SQLite
    for i, issue_commenters in enumerate(df["commenters"].values):
        if len(np.intersect1d(issue_commenters, commenters)):
            passed[i] = True
    return df.take(np.nonzero(passed)[0])


async def append_pr_jira_mapping(prs: Dict[str, PullRequestFacts],
                                 meta_ids: Tuple[int, ...],
                                 mdb: DatabaseLike) -> None:
    """Load and insert "jira_id" to the PR facts."""
    jira_map = await load_pr_jira_mapping(prs, meta_ids, mdb)
    for pr, facts in prs.items():
        facts.jira_id = jira_map.get(pr)


@sentry_span
async def load_pr_jira_mapping(prs: Iterable[str],
                               meta_ids: Tuple[int, ...],
                               mdb: DatabaseLike) -> Dict[str, str]:
    """Fetch the mapping from PR node IDs to JIRA issue IDs."""
    nprji = NodePullRequestJiraIssues
    rows = await mdb.fetch_all(sql.select([nprji.node_id, nprji.jira_id])
                               .where(sql.and_(nprji.node_id.in_(prs),
                                               nprji.node_acc.in_(meta_ids))))
    return {r[0]: r[1] for r in rows}


def resolve_work_began_and_resolved(issue_work_began: Optional[np.datetime64],
                                    prs_began: Optional[np.datetime64],
                                    issue_resolved: Optional[np.datetime64],
                                    prs_released: Optional[np.datetime64],
                                    ) -> Tuple[Optional[np.datetime64], Optional[np.datetime64]]:
    """Compute the final timestamps of when the work started on the issue, and when the issue \
    became fully resolved."""
    if issue_work_began != issue_work_began or issue_work_began is None:
        return None, None
    if prs_began != prs_began or prs_began is None:
        return issue_work_began, \
            issue_resolved \
            if (issue_resolved == issue_resolved and issue_resolved is not None) \
            else None
    work_began = min(prs_began, issue_work_began)
    if (prs_released != prs_released or prs_released is None) or \
            (issue_resolved != issue_resolved or issue_resolved is None):
        return work_began, None
    return work_began, max(issue_resolved, prs_released)
