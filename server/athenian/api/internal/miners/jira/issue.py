from collections import defaultdict
from datetime import datetime, timezone
from itertools import chain
import logging
import pickle
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import aiomcache
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import func, sql
from sqlalchemy.orm import aliased
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached, middle_term_exptime, short_term_exptime
from athenian.api.db import Database, DatabaseLike
from athenian.api.internal.jira import JIRAConfig
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.label import fetch_labels_to_filter
from athenian.api.internal.miners.github.logical import split_logical_prs
from athenian.api.internal.miners.github.precomputed_prs import triage_by_release_match
from athenian.api.internal.miners.types import PullRequestFactsMap
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseMatch, \
    ReleaseSettings
from athenian.api.models.metadata.github import NodePullRequest, NodePullRequestJiraIssues, \
    NodeRepository, PullRequest
from athenian.api.models.metadata.jira import AthenianIssue, Component, Epic, Issue, Status
from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts
from athenian.api.to_object_arrays import is_not_null
from athenian.api.tracing import sentry_span


async def generate_jira_prs_query(filters: List[ClauseElement],
                                  jira: JIRAFilter,
                                  meta_ids: Optional[Tuple[int, ...]],
                                  mdb: Database,
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
    meta_ids_cond = (on[1].in_(meta_ids),) if meta_ids is not None else ()
    if jira.unmapped:
        return sql.select(columns).select_from(sql.outerjoin(
            seed, _map, sql.and_(on[0] == _map.node_id, on[1] == _map.node_acc),
        )).where(sql.and_(*filters, *meta_ids_cond, _map.node_id.is_(None)))
    _issue = aliased(Issue, name="j")
    filters.extend((
        _issue.acc_id == jira.account,
        _issue.project_id.in_(jira.projects),
        _issue.is_deleted.is_(False),
        *meta_ids_cond,
    ))
    if jira.labels:
        components = await _load_components(jira.labels, jira.account, mdb, cache)
        _append_label_filters(
            jira.labels, components, mdb.url.dialect == "postgresql", filters, model=_issue)
    if jira.issue_types:
        filters.append(_issue.type.in_(jira.issue_types))
    if not jira.epics:
        return sql.select(columns).select_from(sql.join(
            seed, sql.join(_map, _issue, sql.and_(
                _map.jira_acc == _issue.acc_id,
                _map.jira_id == _issue.id,
            )),
            sql.and_(
                on[0] == _map.node_id,
                on[1] == _map.node_acc,
            ),
        )).where(sql.and_(*filters))
    _issue_epic = aliased(Issue, name="e")
    filters.append(_issue_epic.key.in_(jira.epics))
    return sql.select(columns).select_from(sql.join(
        seed, sql.join(
            _map, sql.join(_issue, _issue_epic, sql.and_(
                _issue.epic_id == _issue_epic.id,
                _issue.acc_id == _issue_epic.acc_id,
            )),
            sql.and_(
                _map.jira_id == _issue.id,
                _map.jira_acc == _issue.acc_id,
            )),
        sql.and_(
            on[0] == _map.node_id,
            on[1] == _map.node_acc,
        ),
    )).where(sql.and_(*filters))


@sentry_span
@cached(
    exptime=middle_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda labels, account, **_: (labels, account),
    refresh_on_access=True,
)
async def _load_components(labels: LabelFilter,
                           account: int,
                           mdb: Database,
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
ISSUE_PR_IDS = "pr_ids"


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda installation_ids, time_from, time_to, exclude_inactive, labels, priorities, types, epics, reporters, assignees, commenters, nested_assignees, release_settings, logical_settings, **kwargs: (  # noqa
        installation_ids[0],
        ",".join(installation_ids[1]),
        time_from.timestamp() if time_from else "-",
        time_to.timestamp() if time_to else "-",
        exclude_inactive,
        labels,
        ",".join(sorted(priorities)),
        ",".join(sorted(types)),
        ",".join(sorted(epics) if not isinstance(epics, bool) else ["<flying>"]),
        ",".join(sorted(reporters)),
        ",".join(sorted((ass if ass is not None else "<None>") for ass in assignees)),
        ",".join(sorted(commenters)),
        nested_assignees,
        ",".join(c.name for c in kwargs.get("extra_columns", ())),
        release_settings,
        logical_settings,
    ),
)
async def fetch_jira_issues(installation_ids: JIRAConfig,
                            time_from: Optional[datetime],
                            time_to: Optional[datetime],
                            exclude_inactive: bool,
                            labels: LabelFilter,
                            priorities: Collection[str],
                            types: Collection[str],
                            epics: Union[Collection[str], bool],
                            reporters: Collection[str],
                            assignees: Collection[Optional[str]],
                            commenters: Collection[str],
                            nested_assignees: bool,
                            default_branches: Dict[str, str],
                            release_settings: ReleaseSettings,
                            logical_settings: LogicalRepositorySettings,
                            account: int,
                            meta_ids: Tuple[int, ...],
                            mdb: Database,
                            pdb: Database,
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
    :param epics: List of required parent epic keys. If empty, disable filtering by epics. \
                  If false, return only those issues which are without an epic and are not epics \
                  themselves.
    :param reporters: List of lower-case issue reporters.
    :param assignees: List of lower-case issue assignees. None means unassigned.
    :param commenters: List of lower-case issue commenters.
    :param nested_assignees: If filter by assignee, include all the children's.
    :param extra_columns: Additional `Issue` or `AthenianIssue` columns to fetch.
    """
    log = logging.getLogger("%s.jira" % metadata.__package__)
    issues = await _fetch_issues(
        installation_ids, time_from, time_to, exclude_inactive, labels, priorities, types, epics,
        reporters, assignees, commenters, nested_assignees, mdb, cache,
        extra_columns=extra_columns)
    if not exclude_inactive:
        # DEV-1899: exclude and report issues with empty AthenianIssue
        if (missing_updated := issues[AthenianIssue.updated.name].isnull().values).any():
            log.error("JIRA issues are missing in jira.athenian_issue: %s",
                      ", ".join(issues[Issue.key.name].take(np.nonzero(missing_updated)[0])))
            issues = issues.take(np.nonzero(~missing_updated)[0])
    if len(issues.index) >= 20:
        jira_id_cond = NodePullRequestJiraIssues.jira_id.in_any_values(issues.index)
    else:
        jira_id_cond = NodePullRequestJiraIssues.jira_id.in_(issues.index)
    nullable_repository_id = NodePullRequest.repository_id
    nullable_repository_id = nullable_repository_id.label(nullable_repository_id.name)
    nullable_repository_id.nullable = True
    pr_cols = [
        NodePullRequestJiraIssues.node_id,
        NodePullRequestJiraIssues.jira_id,
        NodePullRequest.title,
        NodePullRequest.created_at,
        nullable_repository_id,
        NodeRepository.name_with_owner.label(PullRequest.repository_full_name.name),
    ]
    prs = await read_sql_query(
        sql.select(pr_cols)
        .select_from(
            sql.outerjoin(
                sql.outerjoin(NodePullRequestJiraIssues, NodePullRequest, sql.and_(
                    NodePullRequestJiraIssues.node_acc == NodePullRequest.acc_id,
                    NodePullRequestJiraIssues.node_id == NodePullRequest.graph_id,
                )),
                NodeRepository,
                sql.and_(
                    NodePullRequest.acc_id == NodeRepository.acc_id,
                    NodePullRequest.repository_id == NodeRepository.graph_id,
                )))
        .where(sql.and_(NodePullRequestJiraIssues.jira_acc == installation_ids[0],
                        NodePullRequestJiraIssues.node_acc.in_(meta_ids),
                        jira_id_cond)),
        mdb, pr_cols, index=NodePullRequestJiraIssues.node_id.name,
    )
    # TODO(vmarkovtsev): load the "fresh" released PRs
    existing_repos = np.flatnonzero(is_not_null(
        prs[PullRequest.repository_full_name.name].values))
    if len(existing_repos) < len(prs):
        log.error(
            "Repositories referenced by github.node_pullrequest do not exist in "
            "github.node_repository on GitHub account %s: %s",
            meta_ids, np.unique(prs[NodePullRequest.repository_id.name].values[np.setdiff1d(
                np.arange(len(prs)), existing_repos, assume_unique=True)]).tolist())
        prs = prs.take(existing_repos)
    unique_pr_node_ids = prs.index.unique()
    released_prs, labels = await gather(
        _fetch_released_prs(unique_pr_node_ids, default_branches, release_settings, account, pdb),
        fetch_labels_to_filter(unique_pr_node_ids, meta_ids, mdb),
    )
    prs = split_logical_prs(
        prs, labels,
        logical_settings.with_logical_prs(prs[PullRequest.repository_full_name.name].unique()),
        logical_settings)
    pr_to_issue = {
        key: ji for key, ji in zip(
            prs.index.values,
            prs[NodePullRequestJiraIssues.jira_id.name].values,
        )
    }
    issue_to_index = {iid: i for i, iid in enumerate(issues.index.values)}

    pr_node_ids = prs.index.get_level_values(0).values
    jira_ids = prs[NodePullRequestJiraIssues.jira_id.name].values
    unique_jira_ids, index_map, counts = np.unique(
        jira_ids, return_inverse=True, return_counts=True)
    split_pr_node_ids = np.split(pr_node_ids[np.argsort(index_map)], np.cumsum(counts[:-1]))
    issue_prs = [[]] * len(issues)  # yes, the references to the same list
    issue_indexes = []
    for issue, node_ids in zip(unique_jira_ids, split_pr_node_ids):
        issue_index = issue_to_index[issue]
        issue_indexes.append(issue_index)
        issue_prs[issue_index] = node_ids
    prs_count = np.zeros(len(issues), dtype=int)
    prs_count[issue_indexes] = counts

    nat = np.datetime64("nat")
    work_began = np.full(len(issues), nat, "datetime64[ns]")
    released = work_began.copy()

    for key, pr_created_at in zip(
        prs.index.values,
        prs[NodePullRequest.created_at.name].values,
    ):
        i = issue_to_index[pr_to_issue[key]]
        node_id, repo = key
        if pr_created_at is not None:
            work_began[i] = np.nanmin(np.array(
                [work_began[i], pr_created_at],
                dtype=np.datetime64))
        if (row := released_prs.get(key)) is not None:
            released[i] = np.nanmax(np.array(
                [released[i], row[GitHubDonePullRequestFacts.pr_done_at.name]],
                dtype=np.datetime64))
            continue
        if repo not in release_settings.native:
            # deleted repository, consider the PR as force push dropped
            released[i] = work_began[i]
        else:
            released[i] = nat

    issues[ISSUE_PRS_BEGAN] = work_began
    issues[ISSUE_PRS_RELEASED] = released
    issues[ISSUE_PRS_COUNT] = prs_count
    issues[ISSUE_PR_IDS] = issue_prs
    resolved_colname = AthenianIssue.resolved.name
    created_colname = Issue.created.name
    issues[resolved_colname] = issues[resolved_colname].astype(issues[created_colname].dtype)
    if (negative := issues[resolved_colname].values < issues[created_colname].values).any():
        log.error("JIRA issues have resolved < created: %s",
                  issues.index.values[negative].tolist())
        issues[resolved_colname].values[negative] = issues[created_colname].values[negative]
    return issues


@sentry_span
async def _fetch_released_prs(pr_node_ids: Iterable[int],
                              default_branches: Dict[str, str],
                              release_settings: ReleaseSettings,
                              account: int,
                              pdb: Database,
                              ) -> Dict[Tuple[int, str], Mapping[str, Any]]:
    ghdprf = GitHubDonePullRequestFacts
    released_rows = await pdb.fetch_all(
        sql.select([ghdprf.pr_node_id,
                    ghdprf.pr_created_at,
                    ghdprf.pr_done_at,
                    ghdprf.repository_full_name,
                    ghdprf.release_match])
        .where(sql.and_(ghdprf.pr_node_id.in_(pr_node_ids),
                        ghdprf.acc_id == account)))
    released_by_repo = defaultdict(lambda: defaultdict(dict))
    for r in released_rows:
        released_by_repo[
            r[ghdprf.repository_full_name.name]][
            r[ghdprf.release_match.name]][
            r[ghdprf.pr_node_id.name]] = r
    released_prs = {}
    ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
    for repo, matches in released_by_repo.items():
        for match, prs in matches.items():
            if repo not in release_settings.native:
                for node_id, row in prs.items():
                    key = (node_id, repo)
                    try:
                        if released_prs[key][ghdprf.pr_done_at] < row[ghdprf.pr_done_at]:
                            released_prs[key] = row
                    except KeyError:
                        released_prs[key] = row
                continue
            dump = triage_by_release_match(
                repo, match, release_settings, default_branches, released_prs, ambiguous)
            if dump is None:
                continue
            for node_id, row in prs.items():
                dump[(node_id, repo)] = row
    released_prs.update(ambiguous[ReleaseMatch.tag.name])
    for key, row in ambiguous[ReleaseMatch.branch.name].items():
        released_prs.setdefault(key, row)
    return released_prs


@sentry_span
async def _fetch_issues(ids: JIRAConfig,
                        time_from: Optional[datetime],
                        time_to: Optional[datetime],
                        exclude_inactive: bool,
                        labels: LabelFilter,
                        priorities: Collection[str],
                        types: Collection[str],
                        epics: Union[Collection[str], bool],
                        reporters: Collection[str],
                        assignees: Collection[Optional[str]],
                        commenters: Collection[str],
                        nested_assignees: bool,
                        mdb: Database,
                        cache: Optional[aiomcache.Client],
                        extra_columns: Iterable[InstrumentedAttribute] = (),
                        ) -> pd.DataFrame:
    postgres = mdb.url.dialect == "postgresql"
    columns = [
        Issue.id,
        Issue.type,
        Issue.created,
        AthenianIssue.updated,
        AthenianIssue.work_began,
        AthenianIssue.resolved,
        Issue.priority_name,
        Issue.epic_id,
        Issue.status,
        Status.category_name,
        Issue.labels,
    ]
    columns.extend(extra_columns)
    # this is backed with a DB index
    far_away_future = datetime(3000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    and_filters = [
        Issue.acc_id == ids.acc_id,
        Issue.project_id.in_(ids.projects),
        Issue.is_deleted.is_(False),
    ]
    filter_by_athenian_issue = False
    if time_from is not None:
        filter_by_athenian_issue = True
        and_filters.append(sql.func.coalesce(AthenianIssue.resolved, far_away_future) >= time_from)
    if time_to is not None:
        and_filters.append(Issue.created < time_to)
    if exclude_inactive and time_from is not None:
        filter_by_athenian_issue = True
        and_filters.append(AthenianIssue.acc_id == ids.acc_id)
        and_filters.append(AthenianIssue.updated >= time_from)
    if len(priorities):
        and_filters.append(sql.func.lower(Issue.priority_name).in_(priorities))
    if len(types):
        and_filters.append(sql.func.lower(Issue.type).in_(types))
    if isinstance(epics, bool):
        assert epics is False
        epics_major = aliased(Epic, name="epics_major")
        epics_parent = aliased(Epic, name="epics_parent")
        epics_self = aliased(Epic, name="epics_self")
        for alias in (epics_major, epics_parent, epics_self):
            and_filters.append(alias.name.is_(None))
    elif len(epics):
        and_filters.append(Epic.key.in_(epics))
    or_filters = []
    if labels:
        components = await _load_components(labels, ids[0], mdb, cache)
        _append_label_filters(
            labels, components, mdb.url.dialect == "postgresql", and_filters)
    if reporters and (postgres or not commenters):
        or_filters.append(sql.func.lower(Issue.reporter_display_name).in_(reporters))
    if assignees and (postgres or (not commenters and not nested_assignees)):
        if None in assignees:
            # NULL IN (NULL) = false
            or_filters.append(sql.func.lower(Issue.assignee_display_name).is_(None))
        if nested_assignees:
            filter_by_athenian_issue = True
            or_filters.append(AthenianIssue.nested_assignee_display_names.has_any(assignees))
        else:
            or_filters.append(sql.func.lower(Issue.assignee_display_name).in_(assignees))
    if commenters:
        if postgres:
            or_filters.append(Issue.commenters_display_names.overlap(commenters))
        else:
            if reporters:
                columns.append(sql.func.lower(Issue.reporter_display_name).label("_reporter"))
            if assignees:
                columns.append(sql.func.lower(Issue.assignee_display_name).label("_assignee"))
                if nested_assignees and all(
                        c.name != AthenianIssue.nested_assignee_display_names.name
                        for c in extra_columns):
                    columns.append(AthenianIssue.nested_assignee_display_names)
            if all(c.name != "commenters" for c in extra_columns):
                columns.append(Issue.commenters_display_names.label("commenters"))
    if assignees and not postgres:
        if nested_assignees and all(
                c.name != AthenianIssue.nested_assignee_display_names.name
                for c in columns):
            columns.append(AthenianIssue.nested_assignee_display_names)
        if None in assignees and all(c.name != "_assignee" for c in columns):
            columns.append(sql.func.lower(Issue.assignee_display_name).label("_assignee"))

    def query_starts():
        seeds = [seed := sql.join(Issue, Status, sql.and_(Issue.status_id == Status.id,
                                                          Issue.acc_id == Status.acc_id))]
        if epics is False:
            seeds = [
                sql.outerjoin(
                    sql.outerjoin(
                        sql.outerjoin(seed, epics_major,
                                      sql.and_(Issue.epic_id == epics_major.id,
                                               Issue.acc_id == epics_major.acc_id)),
                        epics_parent, sql.and_(Issue.parent_id == epics_parent.id,
                                               Issue.acc_id == epics_parent.acc_id),
                    ),
                    epics_self, sql.and_(Issue.id == epics_self.id,
                                         Issue.acc_id == epics_self.acc_id),
                ),
            ]
        elif len(epics):
            seeds = [
                sql.join(seed, Epic, sql.and_(Issue.epic_id == Epic.id,
                                              Issue.acc_id == Epic.acc_id)),
                sql.join(seed, Epic, sql.and_(Issue.parent_id == Epic.id,
                                              Issue.acc_id == Epic.acc_id)),
            ]
        return tuple(sql.select(columns).select_from(sql.outerjoin(
            seed, AthenianIssue, sql.and_(Issue.acc_id == AthenianIssue.acc_id,
                                          Issue.id == AthenianIssue.id)))
                     for seed in seeds)

    if or_filters:
        if postgres:
            query = [start.where(sql.and_(or_filter, *and_filters))
                     for or_filter in or_filters
                     for start in query_starts()]
        else:
            query = [start.where(sql.and_(sql.or_(*or_filters), *and_filters))
                     for start in query_starts()]
    else:
        query = [start.where(sql.and_(*and_filters)) for start in query_starts()]

    def hint(q):
        return q \
            .with_statement_hint("Leading(((athenian_issue issue) (s c)))") \
            .with_statement_hint("Rows(athenian_issue issue *1000)") \
            .with_statement_hint("Rows(s c *200)")

    if postgres:
        if len(query) == 1:
            query = query[0]
            if filter_by_athenian_issue:
                query = hint(query)
        elif not filter_by_athenian_issue:
            query = sql.union(*query)
        else:
            query = [hint(q) for q in query]
        if isinstance(query, list):
            df = await gather(*(read_sql_query(q, mdb, columns, index=Issue.id.name)
                                for q in query))
            df = pd.concat(df, copy=False)
            df.disable_consolidate()
            _, unique = np.unique(df.index.values, return_index=True)
            df = df.take(unique)
        else:
            df = await read_sql_query(query, mdb, columns, index=Issue.id.name)
    else:
        # SQLite does not allow to use parameters multiple times
        df = pd.concat(await gather(*(read_sql_query(q, mdb, columns, index=Issue.id.name)
                                      for q in query)))
    df = _validate_and_clean_issues(df, ids[0])
    df.sort_index(inplace=True)
    if postgres or (not commenters and (not nested_assignees or not assignees)):
        return df
    passed = np.full(len(df), False)
    if reporters:
        passed |= df["_reporter"].isin(reporters).values
    if assignees:
        if nested_assignees:
            assignees = set(assignees)
            passed |= df[AthenianIssue.nested_assignee_display_names.name].apply(
                lambda obj: bool(obj.keys() & assignees)).values
        else:
            passed |= df["_assignee"].isin(assignees).values
        if None in assignees:
            passed |= df["_assignee"].isnull().values
    if commenters:
        # don't go hardcore vectorized here, we don't have to with SQLite
        for i, issue_commenters in enumerate(df["commenters"].values):
            if len(np.intersect1d(issue_commenters, commenters)):
                passed[i] = True
    df.disable_consolidate()
    df = df.take(np.flatnonzero(passed))
    sentry_sdk.Hub.current.scope.span.description = str(len(df))
    return df


def _validate_and_clean_issues(df: pd.DataFrame, acc_id: int) -> pd.DataFrame:
    in_progress = df[Status.category_name.name].values == Status.CATEGORY_IN_PROGRESS
    done = df[Status.category_name.name].values == Status.CATEGORY_DONE
    no_work_began = df[AthenianIssue.work_began.name].isnull().values
    no_resolved = df[AthenianIssue.resolved.name].isnull().values
    in_progress_no_work_began = in_progress & no_work_began
    done_no_work_began = done & no_work_began
    done_no_resolved = done & no_resolved
    invalid = in_progress_no_work_began | done_no_work_began | done_no_resolved
    if not invalid.any():
        return df
    log = logging.getLogger(f"{metadata.__package__}.validate_and_clean_issues")
    issue_ids = df.index.values
    if in_progress_no_work_began.any():
        log.error("account %d has issues in progress but their `work_began` is null: %s",
                  acc_id, issue_ids[in_progress_no_work_began].tolist())
    if done_no_work_began.any():
        log.error("account %d has issues done but their `work_began` is null: %s",
                  acc_id, issue_ids[done_no_work_began].tolist())
    if done_no_resolved.any():
        log.error("account %d has issues done but their `resolved` is null: %s",
                  acc_id, issue_ids[done_no_resolved].tolist())
    old_len = len(df)
    df = df.take(np.flatnonzero(~invalid))
    log.warning("cleaned JIRA issues %d / %d", len(df), old_len)
    return df


class PullRequestJiraMapper:
    """Mapper of pull requests to JIRA tickets."""

    @classmethod
    async def append_pr_jira_mapping(cls,
                                     prs: PullRequestFactsMap,
                                     meta_ids: Tuple[int, ...],
                                     mdb: DatabaseLike) -> None:
        """Load and insert "jira_id" to the PR facts."""
        pr_node_ids = defaultdict(list)
        for node_id, repo in prs:
            pr_node_ids[node_id].append(repo)
        jira_map = await cls.load_pr_jira_mapping(pr_node_ids, meta_ids, mdb)
        for pr_node_id, jira in jira_map.items():
            for repo in pr_node_ids[pr_node_id]:
                try:
                    prs[(pr_node_id, repo)].jira_ids = jira
                except KeyError:
                    # we removed this PR in JIRA filter
                    continue

    @classmethod
    @sentry_span
    async def load_pr_jira_mapping(cls,
                                   prs: Collection[int],
                                   meta_ids: Tuple[int, ...],
                                   mdb: DatabaseLike,
                                   ) -> Dict[int, List[str]]:
        """Fetch the mapping from PR node IDs to JIRA issue IDs."""
        nprji = NodePullRequestJiraIssues
        if len(prs) >= 100:
            node_id_cond = nprji.node_id.in_any_values(prs)
        else:
            node_id_cond = nprji.node_id.in_(prs)
        rows = await mdb.fetch_all(
            sql.select([nprji.node_id, Issue.key])
            .select_from(sql.outerjoin(nprji, Issue, sql.and_(
                nprji.jira_acc == Issue.acc_id,
                nprji.jira_id == Issue.id,
            )))
            .where(sql.and_(node_id_cond,
                            nprji.node_acc.in_(meta_ids))))
        result = defaultdict(list)
        for r in rows:
            result[r[0]].append(r[1])
        return result


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


async def fetch_jira_issues_for_prs(pr_nodes: Collection[int],
                                    meta_ids: Tuple[int, ...],
                                    jira_ids: JIRAConfig,
                                    mdb: DatabaseLike,
                                    ) -> List[Mapping[str, Any]]:
    """Load brief information about JIRA issues mapped to the given PRs."""
    regiss = aliased(Issue, name="regular")
    epiciss = aliased(Epic, name="epic")
    prmap = aliased(NodePullRequestJiraIssues, name="m")
    return await mdb.fetch_all(
        sql.select([prmap.node_id.label("node_id"),
                    regiss.key.label("key"),
                    regiss.title.label("title"),
                    regiss.labels.label("labels"),
                    regiss.type.label("type"),
                    epiciss.key.label("epic")])
        .select_from(sql.outerjoin(
            sql.join(regiss, prmap, sql.and_(regiss.id == prmap.jira_id,
                                             regiss.acc_id == prmap.jira_acc)),
            epiciss, sql.and_(epiciss.id == regiss.epic_id,
                              epiciss.acc_id == regiss.acc_id)))
        .where(sql.and_(prmap.node_id.in_(pr_nodes),
                        prmap.node_acc.in_(meta_ids),
                        regiss.project_id.in_(jira_ids[1]),
                        regiss.is_deleted.is_(False))))


participant_columns = [
    func.lower(Issue.reporter_display_name).label("reporter"),
    func.lower(Issue.assignee_display_name).label("assignee"),
    Issue.commenters_display_names.label("commenters"),
]
