import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from itertools import chain
import logging
import pickle
from typing import (
    Any,
    Callable,
    Collection,
    Coroutine,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Type,
)

import aiomcache
import medvedi as md
from medvedi.accelerators import array_of_objects, is_not_null
import numpy as np
from numpy import typing as npt
import sentry_sdk
import sqlalchemy as sa
from sqlalchemy import BigInteger, func, sql
from sqlalchemy.orm import aliased
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import CancelCache, cached, middle_term_exptime, short_term_exptime
from athenian.api.db import Database, DatabaseLike, Row, least
from athenian.api.internal.jira import JIRAConfig
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.label import fetch_labels_to_filter
from athenian.api.internal.miners.github.logical import split_logical_prs
from athenian.api.internal.miners.github.precomputed_prs import triage_by_release_match
from athenian.api.internal.miners.types import (
    PR_JIRA_DETAILS_COLUMN_MAP,
    JIRAEntityToFetch,
    LoadedJIRADetails,
    PullRequestFactsMap,
)
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseMatch, ReleaseSettings
from athenian.api.models.metadata.github import (
    NodePullRequest,
    NodePullRequestJiraIssues,
    NodeRepository,
    PullRequest,
)
from athenian.api.models.metadata.jira import (
    AthenianIssue,
    Component,
    EmptyTextArray,
    Epic,
    Issue,
    Status,
)
from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts
from athenian.api.tracing import sentry_span


async def generate_jira_prs_query(
    filters: list[ClauseElement],
    jira: JIRAFilter,
    meta_ids: Optional[tuple[int, ...]],
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
    assert not isinstance(jira.epics, bool)  # not yet supported
    if columns is PullRequest:
        columns = [PullRequest]
    _map = aliased(NodePullRequestJiraIssues, name="m")
    filters = list(filters)
    if meta_ids is not None:
        filters.append(on[1].in_(meta_ids))
    if jira.unmapped:
        return (
            sql.select(*columns)
            .select_from(
                sql.outerjoin(
                    seed, _map, sql.and_(on[0] == _map.node_id, on[1] == _map.node_acc),
                ),
            )
            .where(_map.node_id.is_(None), *filters)
        )
    _issue = aliased(Issue, name="j")
    filters.append(_issue.is_deleted.is_(False))
    if jira.labels:
        components = await _load_components(jira.labels, jira.account, mdb, cache)
        _append_label_filters(
            jira.labels, components, mdb.url.dialect == "postgresql", filters, model=_issue,
        )
    if jira.issue_types:
        filters.append(_issue.type.in_(jira.issue_types))
    if jira.priorities:
        # priorities are normalized in JIRAFilter but not in the DB
        filters.append(sql.func.lower(_issue.priority_name).in_(jira.priorities))

    if not jira.epics:
        filters.extend([_issue.acc_id == jira.account, _issue.project_id.in_(jira.projects)])
        return (
            sql.select(*columns)
            .select_from(
                sql.join(
                    seed,
                    sql.join(
                        _map,
                        _issue,
                        sql.and_(
                            _map.jira_acc == _issue.acc_id,
                            _map.jira_id == _issue.id,
                        ),
                    ),
                    sql.and_(
                        on[0] == _map.node_id,
                        on[1] == _map.node_acc,
                    ),
                ),
            )
            .where(*filters)
        )

    _issue_epic = aliased(Issue, name="e")
    filters.extend(
        [
            _issue_epic.acc_id == jira.account,
            _issue_epic.key.in_(jira.epics),
            _issue_epic.project_id.in_(jira.projects),
        ],
    )

    return (
        sql.select(*columns)
        .select_from(
            sql.join(
                sql.join(
                    sql.join(
                        _issue_epic,
                        _issue,
                        sql.and_(
                            _issue.epic_id == _issue_epic.id,
                            _issue.acc_id == _issue_epic.acc_id,
                        ),
                    ),
                    _map,
                    sql.and_(
                        _map.jira_id == _issue.id,
                        _map.jira_acc == _issue.acc_id,
                    ),
                ),
                seed,
                sql.and_(
                    on[0] == _map.node_id,
                    on[1] == _map.node_acc,
                ),
            ),
        )
        .where(*filters)
    )


@sentry_span
@cached(
    exptime=middle_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda labels, account, **_: (labels, account),
    refresh_on_access=True,
)
async def _load_components(
    labels: LabelFilter,
    account: int,
    mdb: Database,
    cache: Optional[aiomcache.Client],
) -> dict[str, str]:
    all_labels = set()
    for label in chain(labels.include, labels.exclude):
        for part in label.split(","):
            all_labels.add(part.strip())
    rows = await mdb.fetch_all(
        sql.select(Component.id, Component.name).where(
            Component.acc_id == account, sql.func.lower(Component.name).in_(all_labels),
        ),
    )
    return {r[1].lower(): r[0] for r in rows}


def _append_label_filters(
    labels: LabelFilter,
    components: dict[str, str],
    postgres: bool,
    filters: list[ClauseElement],
    model=Issue,
):
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
                filters.append(
                    sql.not_(
                        model.components.overlap(
                            [components[s] for s in labels.exclude if s in components],
                        ),
                    ),
                )
    else:
        # neither 100% correct nor efficient, but enough for local development
        if labels.include:
            or_items = []
            singles, multiples = LabelFilter.split(labels.include)
            or_items.extend(model.labels.like("%%%s%%" % s) for s in singles)
            or_items.extend(
                sql.and_(*(model.labels.like("%%%s%%" % s) for s in g)) for g in multiples
            )
            if components:
                if singles:
                    or_items.extend(
                        model.components.like("%%%s%%" % components[s])
                        for s in singles
                        if s in components
                    )
                if multiples:
                    or_items.extend(
                        sql.and_(
                            *(
                                model.components.like("%%%s%%" % components[s])
                                for s in g
                                if s in components
                            ),
                        )
                        for g in multiples
                    )
            filters.append(sql.or_(*or_items))
        if labels.exclude:
            filters.append(
                sql.not_(sql.or_(*(model.labels.like("%%%s%%" % s) for s in labels.exclude))),
            )
            if components:
                filters.append(
                    sql.not_(
                        sql.or_(
                            *(
                                model.components.like("%%%s%%" % components[s])
                                for s in labels.exclude
                                if s in components
                            ),
                        ),
                    ),
                )


ISSUE_PRS_BEGAN = "prs_began"
ISSUE_PRS_RELEASED = "prs_released"
ISSUE_PRS_COUNT = "prs_count"
ISSUE_PR_IDS = "pr_ids"
ISSUE_REQUIRES_RELEASES = "requires_releases"


class _NoResult:
    @classmethod
    async def stub(cls) -> Type["_NoResult"]:
        return cls


async def fetch_jira_issues(
    time_from: Optional[datetime],
    time_to: Optional[datetime],
    jira_filter: JIRAFilter,
    exclude_inactive: bool,
    reporters: Collection[str],
    assignees: Collection[Optional[str]],
    commenters: Collection[str],
    nested_assignees: bool,
    default_branches: Optional[dict[str, str]],
    release_settings: Optional[ReleaseSettings],
    logical_settings: Optional[LogicalRepositorySettings],
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
    extra_columns: Iterable[InstrumentedAttribute] = (),
    adjust_timestamps_using_prs=True,
    on_raw_fetch_complete: Optional[Callable[[md.DataFrame], Coroutine]] = None,
) -> md.DataFrame | tuple[md.DataFrame, Any]:
    """
    Load JIRA issues following the specified filters.

    The aggregation is OR between the participation roles.

    :param time_from: Issues should not be resolved before this timestamp.
    :param time_to: Issues should be opened before this timestamp.
    :param exclude_inactive: Issues must be updated after `time_from`.
    :param reporters: List of lower-case issue reporters.
    :param assignees: List of lower-case issue assignees. None means unassigned.
    :param commenters: List of lower-case issue commenters.
    :param nested_assignees: If filter by assignee, include all the children's.
    :param extra_columns: Additional `Issue` or `AthenianIssue` columns to fetch.
    :param adjust_timestamps_using_prs: Value indicating whether we must populate the columns \
                                        which depend on the mapped pull requests.
    :param on_raw_fetch_complete: Execute arbitrary code upon fetching the raw list of issues.
    """
    result = await _fetch_jira_issues(
        time_from,
        time_to,
        jira_filter,
        exclude_inactive,
        reporters,
        assignees,
        commenters,
        nested_assignees,
        default_branches,
        release_settings,
        logical_settings,
        account,
        meta_ids,
        mdb,
        pdb,
        cache,
        extra_columns,
        adjust_timestamps_using_prs,
        on_raw_fetch_complete,
    )
    if result[1] is _NoResult:
        return result[0]
    return result[:-1]


def _postprocess_fetch_jira_issues(
    result: tuple[md.DataFrame, Any, bool],
    adjust_timestamps_using_prs=True,
    **_,
) -> tuple[md.DataFrame, Any, bool]:
    if adjust_timestamps_using_prs and not result[-1]:
        raise CancelCache()
    return result


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, jira_filter, exclude_inactive, reporters, assignees, commenters, nested_assignees, release_settings, logical_settings, **kwargs: (  # noqa
        time_from.timestamp() if time_from else "-",
        time_to.timestamp() if time_to else "-",
        jira_filter,
        exclude_inactive,
        ",".join(sorted(reporters)),
        ",".join(sorted((ass if ass is not None else "<None>") for ass in assignees)),
        ",".join(sorted(commenters)),
        nested_assignees,
        ",".join(c.name for c in kwargs.get("extra_columns", ())),
        release_settings,
        logical_settings,
    ),
    postprocess=_postprocess_fetch_jira_issues,
)
async def _fetch_jira_issues(
    time_from: Optional[datetime],
    time_to: Optional[datetime],
    jira_filter: JIRAFilter,
    exclude_inactive: bool,
    reporters: Collection[str],
    assignees: Collection[Optional[str]],
    commenters: Collection[str],
    nested_assignees: bool,
    default_branches: Optional[dict[str, str]],
    release_settings: Optional[ReleaseSettings],
    logical_settings: Optional[LogicalRepositorySettings],
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
    extra_columns: Iterable[InstrumentedAttribute],
    adjust_timestamps_using_prs: bool,
    on_raw_fetch_complete: Optional[Callable[[md.DataFrame], Coroutine]],
) -> tuple[md.DataFrame, Any, bool]:
    assert jira_filter.account > 0
    log = logging.getLogger("%s.jira" % metadata.__package__)

    async def query_raw(requires_releases: bool, **kwargs) -> md.DataFrame:
        df = await query_jira_raw(
            [
                Issue.id,
                Issue.created,
                AthenianIssue.updated,
                AthenianIssue.work_began,
                AthenianIssue.resolved,
                Status.category_name,
                *extra_columns,
            ],
            **dict(
                time_to=time_to,
                jira_filter=jira_filter,
                exclude_inactive=exclude_inactive,
                reporters=reporters,
                assignees=assignees,
                commenters=commenters,
                nested_assignees=nested_assignees,
                mdb=mdb,
                cache=cache,
                **kwargs,
            ),
        )
        df[ISSUE_REQUIRES_RELEASES] = requires_releases
        return df

    fetch_tasks = []
    if time_from and time_to:
        # `query_jira_raw` applies `time_from` considering only jira resolution time, we need to
        # extend this by considering also done time of any linked PRs.
        # In order to do so we first retrieve node ids of PRs that probably released in the time
        # interval, then we execute query_jira_raw() again with no `time_from` but filtering
        # the issues resolved before `time_from` and mapped to those PRs
        # We don't need the exact match because we will reliably disambiguate in subsequent code.
        async def _fetch_by_pr_release() -> md.DataFrame | None:
            pr_ids = await _fetch_potentially_released_pr_ids(time_from, time_to, account, pdb)
            if len(pr_ids) == 0:
                return None
            return await query_raw(
                True,
                time_from=None,
                resolved_before=time_from,
                mapped_to_prs=pr_ids,
                meta_ids=meta_ids,
            )

        fetch_tasks.append(_fetch_by_pr_release())
    fetch_tasks.append(query_raw(False, time_from=time_from))

    results = await gather(*fetch_tasks, op="_fetch_jira_issues_fetches")
    results = [r for r in results if r is not None]
    issues = md.concat(*results, copy=False)

    if not exclude_inactive:
        # DEV-1899: exclude and report issues with empty AthenianIssue
        if (missing_updated := issues.isnull(AthenianIssue.updated.name)).any():
            log.error(
                "JIRA issues are missing in jira.athenian_issue: %s",
                ", ".join(map(str, issues.index.values[missing_updated])),
            )
            issues = issues.take(~missing_updated)
    if on_raw_fetch_complete is not None:
        on_raw_fetch_complete_task = asyncio.create_task(
            on_raw_fetch_complete(issues), name="fetch_jira_issues/on_raw_fetch_complete",
        )
    else:
        on_raw_fetch_complete_task = _NoResult.stub()
    if issues.empty or not adjust_timestamps_using_prs:
        _fill_issues_with_empty_prs_info(issues)
        if issues[ISSUE_REQUIRES_RELEASES].any():
            assert default_branches is not None
            requires_releases_mask = issues[ISSUE_REQUIRES_RELEASES]
            requires_releases_issues = issues.take(requires_releases_mask)
            await _fill_issues_with_mapped_prs_info(
                requires_releases_issues,
                default_branches,
                release_settings,
                logical_settings,
                account,
                meta_ids,
                jira_filter.account,
                pdb,
                mdb,
            )
            issues[ISSUE_PRS_RELEASED][requires_releases_mask] = requires_releases_issues[
                ISSUE_PRS_RELEASED
            ]
    else:
        assert default_branches is not None
        await _fill_issues_with_mapped_prs_info(
            issues,
            default_branches,
            release_settings,
            logical_settings,
            account,
            meta_ids,
            jira_filter.account,
            pdb,
            mdb,
        )

    requires_releases_mask = ~(issues[ISSUE_REQUIRES_RELEASES] & issues.isnull(ISSUE_PRS_RELEASED))
    if not requires_releases_mask.all():
        issues.take(requires_releases_mask, inplace=True)
    del issues[ISSUE_REQUIRES_RELEASES]

    return issues, await on_raw_fetch_complete_task, adjust_timestamps_using_prs


@sentry_span
async def fetch_jira_issues_by_keys(
    keys: Sequence[str],
    jira_config: JIRAConfig,
    default_branches: dict[str, str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
    extra_columns: Iterable[InstrumentedAttribute] = (),
) -> md.DataFrame:
    """Load JIRA issues based on their key. Result dataframe order is undetermined."""
    columns = [
        Issue.id,
        Issue.created,
        AthenianIssue.updated,
        AthenianIssue.work_began,
        AthenianIssue.resolved,
        Status.category_name,
        *extra_columns,
    ]
    issues = await _query_jira_raw_by_keys(keys, columns, jira_config, mdb, cache)
    if issues.empty:
        _fill_issues_with_empty_prs_info(issues)
    else:
        await _fill_issues_with_mapped_prs_info(
            issues,
            default_branches,
            release_settings,
            logical_settings,
            account,
            meta_ids,
            jira_config.acc_id,
            pdb,
            mdb,
        )
    return issues


@sentry_span
async def _fill_issues_with_mapped_prs_info(
    issues: md.DataFrame,
    default_branches: dict[str, str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    account: int,
    meta_ids: tuple[int, ...],
    jira_account: int,
    pdb: DatabaseLike,
    mdb: DatabaseLike,
) -> None:
    log = logging.getLogger("%s.jira" % metadata.__package__)

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
    selects = [
        _mapped_prs_select(issues.index.values, pr_cols, jira_account, meta_id)
        for meta_id in meta_ids
    ]
    stmt = sa.union_all(*selects)
    if len(issues) > 100:
        estimated_rows = len(issues) // (2 * len(meta_ids))
        for hint in (
            f"Leading((({NodePullRequestJiraIssues.__tablename__} *VALUES*) "
            f"{NodePullRequest.__tablename__}))",
            f"Rows({NodePullRequestJiraIssues.__tablename__} *VALUES* #{estimated_rows})"
            f"Rows({NodePullRequestJiraIssues.__tablename__} *VALUES* "
            f"{NodeRepository.__tablename__} #{estimated_rows})",
            f"Rows({NodePullRequestJiraIssues.__tablename__} *VALUES* "
            f"{NodePullRequest.__tablename__} #{estimated_rows})",
            f"Rows({NodePullRequestJiraIssues.__tablename__} *VALUES* "
            f"{NodePullRequest.__tablename__} {NodeRepository.__tablename__} #{estimated_rows})",
        ):
            stmt = stmt.with_statement_hint(hint)

    prs = await read_sql_query(stmt, mdb, pr_cols, index=NodePullRequestJiraIssues.node_id.name)

    # TODO(vmarkovtsev): load the "fresh" released PRs
    existing_repos = np.flatnonzero(prs.notnull(PullRequest.repository_full_name.name))
    if len(existing_repos) < len(prs):
        log.error(
            "Repositories referenced by github.node_pullrequest do not exist in "
            "github.node_repository on GitHub account %s: %s",
            meta_ids,
            np.unique(
                prs[NodePullRequest.repository_id.name][
                    np.setdiff1d(np.arange(len(prs)), existing_repos, assume_unique=True)
                ],
            ).tolist(),
        )
        prs = prs.take(existing_repos)
    unique_pr_node_ids = prs.unique(prs.index.name, unordered=True)
    released_prs, labels = await gather(
        _fetch_released_prs(unique_pr_node_ids, default_branches, release_settings, account, pdb),
        fetch_labels_to_filter(unique_pr_node_ids, meta_ids, mdb)
        if logical_settings.has_prs_by_label()
        else None,
    )
    prs = split_logical_prs(
        prs,
        labels,
        logical_settings.with_logical_prs(
            prs.unique(PullRequest.repository_full_name.name, unordered=True),
        ),
        logical_settings,
    )
    pr_to_issue = {
        (node_id, repo): ji
        for node_id, repo, ji in prs.iterrows(
            *prs.index.names,
            NodePullRequestJiraIssues.jira_id.name,
        )
    }
    issue_to_index = {iid: i for i, iid in enumerate(issues.index.values)}

    pr_node_ids = prs.index.get_level_values(0)
    jira_ids = prs[NodePullRequestJiraIssues.jira_id.name]
    unique_jira_ids, index_map, counts = np.unique(
        jira_ids, return_inverse=True, return_counts=True,
    )
    split_pr_node_ids = np.split(pr_node_ids[np.argsort(index_map)], np.cumsum(counts[:-1]))
    issue_prs = array_of_objects(len(issues), [])  # yes, the references to the same list
    issue_indexes = []
    for issue, node_ids in zip(unique_jira_ids, split_pr_node_ids):
        issue_index = issue_to_index[issue]
        issue_indexes.append(issue_index)
        issue_prs[issue_index] = node_ids
    prs_count = np.zeros(len(issues), dtype=int)
    prs_count[issue_indexes] = counts

    nat = np.datetime64("nat")
    work_began = np.full(len(issues), nat, "datetime64[us]")
    released = work_began.copy()

    for node_id, repo, pr_created_at in prs.iterrows(
        *prs.index.names,
        NodePullRequest.created_at.name,
    ):
        key = node_id, repo
        i = issue_to_index[pr_to_issue[key]]
        if pr_created_at is not None:
            if pr_created_at == pr_created_at:
                key_work_began = work_began[i]
                work_began[i] = key_work_began if key_work_began < pr_created_at else pr_created_at
        if (pr_done_at := released_prs.get(key)) is not None:
            if pr_done_at == pr_done_at:
                key_released = released[i]
                released[i] = key_released if key_released > pr_done_at else pr_done_at
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
    ISSUE_RESOLVED = AthenianIssue.resolved.name
    ISSUE_CREATED = Issue.created.name

    if (negative := issues[ISSUE_RESOLVED] < issues[ISSUE_CREATED]).any():
        log.error(
            "JIRA issues have resolved < created: %s",
            issues.index.values[negative].tolist(),
        )
        issues[ISSUE_RESOLVED][negative] = issues[ISSUE_CREATED][negative]


def _mapped_prs_select(issue_ids: Iterable[str], cols, jira_id: int, meta_id: int) -> sql.Select:
    """Return the select statement to select mapped PRs belonging to a single meta account."""
    return (
        sql.select(*cols)
        .select_from(
            sql.outerjoin(
                sql.outerjoin(
                    NodePullRequestJiraIssues,
                    NodePullRequest,
                    sql.and_(
                        NodePullRequestJiraIssues.node_acc == NodePullRequest.acc_id,
                        NodePullRequestJiraIssues.node_id == NodePullRequest.graph_id,
                    ),
                ),
                NodeRepository,
                sql.and_(
                    NodePullRequest.acc_id == NodeRepository.acc_id,
                    NodePullRequest.repository_id == NodeRepository.graph_id,
                ),
            ),
        )
        .where(
            NodePullRequestJiraIssues.jira_acc == jira_id,
            NodePullRequestJiraIssues.node_acc == meta_id,
            NodePullRequestJiraIssues.jira_id.progressive_in(issue_ids),
        )
    )


def _fill_issues_with_empty_prs_info(issues: md.DataFrame) -> None:
    for col in (ISSUE_PRS_BEGAN, ISSUE_PRS_RELEASED, AthenianIssue.resolved.name):
        issues[col] = np.full(len(issues), np.datetime64("nat"), "datetime64[s]")

    issues[ISSUE_PRS_COUNT] = np.zeros(len(issues), float)
    issues[ISSUE_PR_IDS] = np.full(len(issues), None, object)


@sentry_span
async def _fetch_released_prs(
    pr_node_ids: Iterable[int],
    default_branches: dict[str, str],
    release_settings: ReleaseSettings,
    account: int,
    pdb: Database,
) -> dict[tuple[int, str], np.datetime64]:
    ghdprf = GitHubDonePullRequestFacts
    selected = [
        ghdprf.pr_node_id,
        ghdprf.pr_done_at,
        ghdprf.repository_full_name,
        ghdprf.release_match,
    ]
    released_df = await read_sql_query(
        sql.select(*selected).where(ghdprf.acc_id == account, ghdprf.pr_node_id.in_(pr_node_ids)),
        pdb,
        selected,
    )
    pr_node_ids_col = released_df[ghdprf.pr_node_id.name]
    pr_done_at_col = released_df[ghdprf.pr_done_at.name]
    released_prs = {}
    ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}

    for repo, match, indexes in _iter_released_prs_by_match(released_df):
        if repo not in release_settings.native:
            for node_id, pr_done_at in zip(pr_node_ids_col[indexes], pr_done_at_col[indexes]):
                key = (node_id, repo)
                try:
                    if released_prs[key] < pr_done_at:
                        released_prs[key] = pr_done_at
                except KeyError:
                    released_prs[key] = pr_done_at
            continue
        dump = triage_by_release_match(
            repo, match, release_settings, default_branches, released_prs, ambiguous,
        )
        if dump is None:
            continue
        for node_id, pr_done_at in zip(pr_node_ids_col[indexes], pr_done_at_col[indexes]):
            dump[(node_id, repo)] = pr_done_at
    released_prs.update(ambiguous[ReleaseMatch.tag.name])
    for key, pr_done_at in ambiguous[ReleaseMatch.branch.name].items():
        released_prs.setdefault(key, pr_done_at)
    return released_prs


@sentry_span
async def _fetch_potentially_released_pr_ids(
    time_from: datetime,
    time_to: datetime,
    account: int,
    pdb: Database,
) -> Sequence[str]:
    """
    Fetch the node ids of the PRs *potentially* released in the interval.

    There is a guarantee that if a PR released on the time interval, it is included in the result.
    """
    ghdprf = GitHubDonePullRequestFacts
    format_version = ghdprf.__table__.columns[ghdprf.format_version.key].default.arg
    where = [
        ghdprf.acc_id == account,
        ghdprf.release_match.like("%|%"),
        ghdprf.format_version == format_version,
        ghdprf.pr_done_at.between(time_from, time_to),
    ]
    df = await read_sql_query(
        sa.select(sa.distinct(ghdprf.pr_node_id)).where(*where), pdb, [ghdprf.pr_node_id],
    )
    return df[ghdprf.pr_node_id.name]


def _iter_released_prs_by_match(df: md.DataFrame) -> Iterator[tuple[npt.NDArray[int], str, str]]:
    """Iterate over the released PRs dataframe grouped by release match.

    Emits a tuple for every group of PRs with the same release match and repo; each tuple has
    - the string of the release match
    - the repository
    - indexes of `df` composing the group

    """
    ghdprf = GitHubDonePullRequestFacts
    repos_col = df[ghdprf.repository_full_name.name]
    match_col = df[ghdprf.release_match.name]
    for indexes in df.groupby(ghdprf.repository_full_name.name, ghdprf.release_match.name):
        yield repos_col[indexes[0]], match_col[indexes[0]], indexes


@sentry_span
async def query_jira_raw(
    columns: list[InstrumentedAttribute],
    time_from: Optional[datetime],
    time_to: Optional[datetime],
    jira_filter: JIRAFilter,
    exclude_inactive: bool,
    reporters: Collection[str],
    assignees: Collection[Optional[str]],
    commenters: Collection[str],
    nested_assignees: bool,
    mdb: Database,
    cache: Optional[aiomcache.Client],
    distinct: bool = False,
    resolved_before: datetime | None = None,
    mapped_to_prs: Sequence[int] | None = None,
    meta_ids: tuple[int, ...] | None = None,
) -> md.DataFrame:
    """
    Fetch arbitrary columns from Issue or any joined tables according to the filters.

    :param distinct: Generate a counting "GROUP BY" instead of a plain SELECT.
    """
    assert columns
    _PR_ISSUES_THRESHOLD = 20

    postgres = mdb.url.dialect == "postgresql"
    # this is backed with a DB index
    far_away_future = datetime(3000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    and_filters = [
        Issue.acc_id == jira_filter.account,
        Issue.project_id.in_(jira_filter.projects),
        Issue.is_deleted.is_(False),
    ]
    filter_by_athenian_issue = False
    if time_from is not None:
        filter_by_athenian_issue = True
        resolved_coalesced = sql.func.coalesce(AthenianIssue.resolved, far_away_future)
        if exclude_inactive:
            and_filters.append(
                (await least(mdb))(resolved_coalesced, AthenianIssue.updated) >= time_from,
            )
        else:
            and_filters.append(resolved_coalesced >= time_from)
    if resolved_before is not None:
        filter_by_athenian_issue = True
        and_filters.append(AthenianIssue.resolved < resolved_before)
    if time_to is not None:
        and_filters.append(Issue.created < time_to)
    if len(jira_filter.priorities):
        and_filters.append(sql.func.lower(Issue.priority_name).in_(jira_filter.priorities))
    if len(jira_filter.issue_types):
        and_filters.append(sql.func.lower(Issue.type).in_(jira_filter.issue_types))
    if isinstance(jira_filter.epics, bool):
        assert jira_filter.epics is False
        epics_major = aliased(Epic, name="epics_major")
        epics_self = aliased(Epic, name="epics_self")
        for alias in (epics_major, epics_self):
            and_filters.append(alias.name.is_(None))
    elif len(jira_filter.epics):
        epic = aliased(Issue, name="epic")
        and_filters.append(epic.key.in_(jira_filter.epics))
    if jira_filter.status_categories:
        and_filters.append(Status.category_name.in_(jira_filter.status_categories))

    if mapped_to_prs is not None:
        assert meta_ids
        PRIssues = NodePullRequestJiraIssues
        subselect = sa.exists().where(
            PRIssues.jira_acc == jira_filter.account,
            PRIssues.node_acc.in_(meta_ids),
            PRIssues.jira_id == Issue.id,
            PRIssues.node_id.progressive_in(mapped_to_prs, threshold=_PR_ISSUES_THRESHOLD),
        )
        and_filters.append(subselect)

    or_filters = []
    if jira_filter.labels:
        components = await _load_components(jira_filter.labels, jira_filter.account, mdb, cache)
        _append_label_filters(
            jira_filter.labels, components, mdb.url.dialect == "postgresql", and_filters,
        )
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
            assert not distinct, "not supported in SQLite"
            if reporters:
                columns.append(sql.func.lower(Issue.reporter_display_name).label("_reporter"))
            if assignees:
                columns.append(sql.func.lower(Issue.assignee_display_name).label("_assignee"))
                if nested_assignees and all(
                    c.name != AthenianIssue.nested_assignee_display_names.name for c in columns
                ):
                    columns.append(AthenianIssue.nested_assignee_display_names)
            if all(c.name != "commenters" for c in columns):
                columns.append(Issue.commenters_display_names.label("commenters"))
    if assignees and not postgres:
        if nested_assignees and all(
            c.name != AthenianIssue.nested_assignee_display_names.name for c in columns
        ):
            assert not distinct, "not supported in SQLite"
            columns.append(AthenianIssue.nested_assignee_display_names)
        if None in assignees and all(c.name != "_assignee" for c in columns):
            assert not distinct, "not supported in SQLite"
            columns.append(sql.func.lower(Issue.assignee_display_name).label("_assignee"))
    if distinct:
        columns.append(count_col := sql.literal_column("COUNT(*)", BigInteger).label("count"))
        count_col.nullable = False

    def query_start():
        seed = sql.join(
            Issue,
            Status,
            sql.and_(
                Issue.status_id == Status.id,
                Issue.acc_id == Status.acc_id,
            ),
        )
        if jira_filter.epics is False:
            seed = sql.outerjoin(
                sql.outerjoin(
                    seed,
                    epics_major,
                    sql.and_(Issue.epic_id == epics_major.id, Issue.acc_id == epics_major.acc_id),
                ),
                epics_self,
                sql.and_(Issue.id == epics_self.id, Issue.acc_id == epics_self.acc_id),
            )
        elif len(jira_filter.epics):
            seed = sql.join(
                seed,
                epic,
                sql.and_(Issue.epic_id == epic.id, Issue.acc_id == epic.acc_id),
            )
        return sql.select(*columns).select_from(
            sql.join(
                seed,
                AthenianIssue,
                sql.and_(Issue.acc_id == AthenianIssue.acc_id, Issue.id == AthenianIssue.id),
                # this is important, otherwise we fail to push down the filters after hints
                isouter=not filter_by_athenian_issue,
            ),
        )

    def query_finish(query):
        if not distinct:
            return query
        return query.group_by(*columns[:-1])

    if or_filters:
        if postgres:
            query = [
                query_finish(query_start().where(or_filter, *and_filters))
                for or_filter in or_filters
            ]
        else:
            query = [query_finish(query_start().where(sql.or_(*or_filters), *and_filters))]
    else:
        query = [query_finish(query_start().where(*and_filters))]

    AthenianIssueT = AthenianIssue.__tablename__
    IssueT = Issue.__tablename__

    def hint_athenian_issue(q):
        # "s" and "c" are table aliases used in the api_statuses view
        hints = [
            f"Rows({AthenianIssueT} {IssueT} *1000)",
            # we should not blindly enforce HashJoin, e.g. acc. 135 (jira: 26) severely degrades
            # f"HashJoin({AthenianIssueT} {IssueT})",
            "Rows(s c *200)",
        ]
        if mapped_to_prs is not None:
            PRIssuesT = NodePullRequestJiraIssues.__tablename__
            if len(mapped_to_prs) > _PR_ISSUES_THRESHOLD:
                hints.append(
                    f"Leading((((({PRIssuesT} *VALUES*) {AthenianIssueT}) {IssueT}) (s c)))",
                )
                hints.append(f"HashJoin({PRIssuesT} *VALUES*)")
                hints.append(f"Rows({PRIssuesT} *VALUES* #{len(mapped_to_prs)}")
            else:
                hints.append(f"Leading(((({PRIssues} {AthenianIssueT}) {IssueT}) (s c)))")
        else:
            hints.append(f"Leading((({AthenianIssueT} {IssueT}) (s c)))")

        for hint in hints:
            q = q.with_statement_hint(hint)
        return q

    def hint_epics(q):
        exp_rows = len(jira_filter.epics) * 2
        return (
            q.with_statement_hint(f"Leading(((((epic {IssueT}) s) c) {AthenianIssueT}))")
            .with_statement_hint(f"Rows(epic {IssueT} s c {AthenianIssueT} #{exp_rows})")
            .with_statement_hint(f"Rows(epic {IssueT} #{exp_rows})")
            .with_statement_hint(f"Rows(epic {IssueT} s #{exp_rows})")
            .with_statement_hint(f"Rows(epic {IssueT} s c #{exp_rows})")
        )

    if postgres:
        if len(query) == 1:
            query = query[0]
            if filter_by_athenian_issue:
                query = hint_athenian_issue(query)
            elif len(jira_filter.epics):
                query = hint_epics(query)
        elif filter_by_athenian_issue:
            query = [hint_athenian_issue(q) for q in query]
        elif len(jira_filter.epics):
            query = [hint_epics(q) for q in query]
        else:
            query = sql.union(*query)

        if isinstance(query, list):
            df = await gather(
                *(
                    read_sql_query(q, mdb, columns, index=Issue.id.name if not distinct else None)
                    for q in query
                ),
            )
            df = md.concat(*df, copy=False)
            if not distinct:
                df.drop_duplicates(df.index.names, inplace=True)
        else:
            df = await read_sql_query(
                query, mdb, columns, index=Issue.id.name if not distinct else None,
            )
    else:
        # SQLite does not allow to use parameters multiple times
        df = md.concat(
            *(
                await gather(
                    *(
                        read_sql_query(
                            q, mdb, columns, index=Issue.id.name if not distinct else None,
                        )
                        for q in query
                    ),
                )
            ),
        )

    if not distinct:
        df = _validate_and_clean_issues(df, jira_filter.account)
    if sentry_sdk.Hub.current.scope.span is not None:
        sentry_sdk.Hub.current.scope.span.description = str(len(df))
    if postgres or (not commenters and (not nested_assignees or not assignees)):
        return df
    passed = np.full(len(df), False)
    if reporters:
        passed |= df.isin("_reporter", reporters)
    if assignees:
        if nested_assignees:
            assignees = set(assignees)
            passed |= np.fromiter(
                (
                    bool(obj.keys() & assignees)
                    for obj in df[AthenianIssue.nested_assignee_display_names.name]
                ),
                bool,
                len(df),
            )
        else:
            passed |= df.isin("_assignee", assignees)
        if None in assignees:
            passed |= df.isnull("_assignee")
    if commenters:
        # don't go hardcore vectorized here, we don't have to with SQLite
        for i, issue_commenters in enumerate(df["commenters"]):
            if len(np.intersect1d(issue_commenters, commenters)):
                passed[i] = True
    return df.take(passed, inplace=True)


@sentry_span
async def _query_jira_raw_by_keys(
    keys: Sequence[str],
    columns: list[InstrumentedAttribute],
    jira_config: JIRAConfig,
    mdb: Database,
    cache: Optional[aiomcache.Client],
) -> md.DataFrame:
    where = [
        Issue.acc_id == jira_config.acc_id,
        Issue.project_id.in_(jira_config.projects),
        Issue.is_deleted.is_(False),
        Issue.key.progressive_in(keys),
    ]
    issue_status_join = sa.outerjoin(
        Issue,
        Status,
        sql.and_(Issue.status_id == Status.id, Issue.acc_id == Status.acc_id),
    )
    issue_athenian_issue_join = sa.outerjoin(
        issue_status_join,
        AthenianIssue,
        sql.and_(Issue.acc_id == AthenianIssue.acc_id, Issue.id == AthenianIssue.id),
    )
    stmt = sa.select(*columns).select_from(issue_athenian_issue_join).where(*where)
    if len(keys) > 100:
        stmt = stmt.with_statement_hint(f"Leading({Issue.__tablename__} *VALUES*)")
    issues = await read_sql_query(stmt, mdb, columns, index=Issue.id.name)
    return _validate_and_clean_issues(issues, jira_config.acc_id)


def _validate_and_clean_issues(df: md.DataFrame, acc_id: int) -> md.DataFrame:
    in_progress = df[Status.category_name.name] == Status.CATEGORY_IN_PROGRESS
    done = df[Status.category_name.name] == Status.CATEGORY_DONE
    no_work_began = df.isnull(AthenianIssue.work_began.name)
    no_resolved = df.isnull(AthenianIssue.resolved.name)
    in_progress_no_work_began = in_progress & no_work_began
    done_no_work_began = done & no_work_began
    done_no_resolved = done & no_resolved
    invalid = in_progress_no_work_began | done_no_work_began | done_no_resolved
    if not invalid.any():
        df.sort_index(inplace=True)
        return df
    log = logging.getLogger(f"{metadata.__package__}.validate_and_clean_issues")
    issue_ids = df.index.values
    if in_progress_no_work_began.any():
        log.error(
            "account %d has issues in progress but their `work_began` is null: %s",
            acc_id,
            [iid.decode() for iid in issue_ids[in_progress_no_work_began]],
        )
    if done_no_work_began.any():
        log.error(
            "account %d has issues done but their `work_began` is null: %s",
            acc_id,
            [iid.decode() for iid in issue_ids[done_no_work_began]],
        )
    if done_no_resolved.any():
        log.error(
            "account %d has issues done but their `resolved` is null: %s",
            acc_id,
            [iid.decode() for iid in issue_ids[done_no_resolved]],
        )
    old_len = len(df)
    df = df.take(np.flatnonzero(~invalid))
    df.sort_index(inplace=True)
    log.warning("cleaned JIRA issues %d / %d", len(df), old_len)
    return df


class PullRequestJiraMapper:
    """Mapper of pull requests to JIRA tickets."""

    @classmethod
    @sentry_span
    async def append(
        cls,
        prs: PullRequestFactsMap,
        entities: JIRAEntityToFetch | int,
        meta_ids: tuple[int, ...],
        mdb: DatabaseLike,
    ) -> None:
        """Load and insert "jira_id" to the PR facts."""
        pr_node_ids = defaultdict(list)
        for node_id, repo in prs:
            pr_node_ids[node_id].append(repo)
        jira_map = await cls.load(pr_node_ids, entities, meta_ids, mdb)
        for pr_node_id, jira in jira_map.items():
            for repo in pr_node_ids[pr_node_id]:
                try:
                    prs[(pr_node_id, repo)].jira = jira
                except KeyError:
                    # we removed this PR in JIRA filter
                    continue

    @classmethod
    @sentry_span
    async def load(
        cls,
        prs: Collection[int],
        entities: JIRAEntityToFetch | int,
        meta_ids: tuple[int, ...],
        mdb: DatabaseLike,
    ) -> dict[int, LoadedJIRADetails]:
        """Fetch the mapping from PR node IDs to JIRA issue IDs."""
        nprji = NodePullRequestJiraIssues
        columns = [nprji.node_id, *JIRAEntityToFetch.to_columns(entities)]
        if Issue.labels in columns:
            # import_components_as_labels needs acc_id and components
            columns.extend([Issue.acc_id, Issue.components])

        stmt = (
            sql.select(*columns)
            .select_from(
                sql.outerjoin(
                    nprji,
                    Issue,
                    sql.and_(nprji.jira_acc == Issue.acc_id, nprji.jira_id == Issue.id),
                ),
            )
            .where(nprji.node_id.progressive_in(prs), nprji.node_acc.in_(meta_ids))
            .with_statement_hint(f"Rows({nprji.__tablename__} *VALUES* {len(prs) // 2})")
            .with_statement_hint(
                f"Leading((({nprji.__tablename__} *VALUES*) {Issue.__tablename__}))",
            )
        )
        df = await read_sql_query(stmt, mdb, columns, index=nprji.node_id.name)
        if Issue.labels in columns:
            await import_components_as_labels(df, mdb)
        res: dict[int, LoadedJIRADetails] = {}
        cls.append_from_df(res, df)
        return res

    @classmethod
    @sentry_span
    def append_from_df(
        cls,
        existing: dict[int, LoadedJIRADetails],
        df: md.DataFrame,
    ) -> None:
        """Add the JIRA details in `df` to `existing` mapping from PR node IDs to JIRA."""
        pr_node_ids = df.index.get_level_values(0)
        empty_cols = {}
        payload_columns = []
        for col in JIRAEntityToFetch.to_columns(JIRAEntityToFetch.EVERYTHING()):
            df_name, dtype = PR_JIRA_DETAILS_COLUMN_MAP[col]
            if col.name not in df:
                empty_cols[df_name] = np.array([], dtype=dtype)
            else:
                payload_columns.append(col)
        for indexes in df.groupby(df.index.names[0]):
            # we can deduplicate. shall we? must benchmark the profit.
            existing[pr_node_ids[indexes[0]]] = LoadedJIRADetails(
                # labels are handled differently since they are already an array for each issue
                **{
                    PR_JIRA_DETAILS_COLUMN_MAP[c][0]: np.concatenate(
                        df[c.name][indexes], dtype="U", casting="unsafe",
                    )
                    if c is Issue.labels
                    else df[c.name][indexes]
                    for c in payload_columns
                },
                **empty_cols,
            )

    @classmethod
    def apply_to_pr_facts(
        self,
        facts: PullRequestFactsMap,
        jira: dict[int, LoadedJIRADetails],
    ) -> None:
        """Apply the jira mappings to the facts in PullRequestFactsMap, in place."""
        empty = LoadedJIRADetails.empty()
        for (pr_id, _), pr_facts in facts.items():
            try:
                pr_facts.jira = jira[pr_id]
            except KeyError:
                pr_facts.jira = empty

    @classmethod
    async def load_and_apply_to_pr_facts(
        cls,
        facts: PullRequestFactsMap,
        entities: JIRAEntityToFetch | int,
        meta_ids: tuple[int, ...],
        mdb: DatabaseLike,
    ) -> None:
        """Load the jira mappings and apply it to the facts in PullRequestFactsMap, in place."""
        pr_node_ids = np.fromiter((pr_node_id for pr_node_id, _ in facts), int, len(facts))
        jira_map = await cls.load(pr_node_ids, entities, meta_ids, mdb)
        cls.apply_to_pr_facts(facts, jira_map)

    @classmethod
    def apply_empty_to_pr_facts(self, facts: PullRequestFactsMap) -> None:
        """Apply an empty jira mappings to the facts in PullRequestFactsMap, in place."""
        empty_jira = LoadedJIRADetails.empty()
        for f in facts.values():
            f.jira = empty_jira


def resolve_work_began(work_began: np.ndarray, prs_began: np.ndarray) -> np.ndarray:
    """Compute the final timestamp when the work started on the issues.

    `work_began` are the `AthenianIssue.work_began` values
    `prs_began` are the ISSUE_PRS_BEGAN computed fields of the issues.
    """
    res = np.full(len(work_began), np.datetime64("NaT"), dtype=work_began.dtype)
    has_work_began = work_began == work_began
    res[has_work_began] = np.fmin(work_began[has_work_began], prs_began[has_work_began])
    return res


def resolve_resolved(
    issue_resolved: np.ndarray,
    prs_began: np.ndarray,
    prs_released: np.ndarray,
) -> np.ndarray:
    """Compute the final timestamp when the issue became fully resolved.

    `issue_resolved` are the `AthenianIssue.resolved` values
    `prs_began` are the ISSUE_PRS_BEGAN computed fields of the issues.
    `prs_released` are the ISSUE_PRS_RELEASED fields of the issues.

    """
    resolved = issue_resolved.copy()
    have_prs_mask = prs_began == prs_began
    resolved[have_prs_mask] = np.maximum(prs_released[have_prs_mask], resolved[have_prs_mask])
    return resolved


def resolve_acknowledge_time(
    created: npt.NDArray[np.datetime64],
    work_began: npt.NDArray[np.datetime64],
    now: np.datetime64,
) -> npt.NDArray[np.timedelta64]:
    """Compute the acknowledge_time for the issues.

    `created` are the `Issue.created.name` values
    `work_began` are the final values computed by resolve_work_began()

    The acknowledge_time is `work_began` - `created`.
    For the issues not yet started (so no work_began) it is `now()` - `created`.
    `acknowledge_time` can never be less than 0.
    """
    res = np.empty(len(created), dtype="timedelta64[us]")
    res = work_began - created
    no_work_began = res != res
    res[no_work_began] = now - created[no_work_began]
    res[res < np.timedelta64(0, "us")] = np.timedelta64(0, "us")
    return res


@sentry_span
async def fetch_jira_issues_rows_by_keys(
    keys: Collection[str],
    jira_ids: JIRAConfig,
    mdb: DatabaseLike,
) -> list[Row]:
    """Load brief information about JIRA issues mapped to the given issue keys."""
    regiss = aliased(Issue, name="regular")
    epiciss = aliased(Issue, name="epic")
    query = (
        sql.select(
            regiss.key.label("id"),
            regiss.title.label("title"),
            regiss.labels.label("labels"),
            regiss.type.label("type"),
            epiciss.key.label("epic"),
        )
        .select_from(
            sql.outerjoin(
                regiss,
                epiciss,
                sql.and_(epiciss.id == regiss.epic_id, epiciss.acc_id == regiss.acc_id),
            ),
        )
        .where(
            regiss.acc_id == jira_ids[0],
            regiss.key.progressive_in(keys),
            regiss.project_id.in_(jira_ids[1]),
            regiss.is_deleted.is_(False),
        )
    )
    if len(keys) > 100:
        query = (
            query.with_statement_hint("Leading(((regular *VALUES*) epic))")
            .with_statement_hint(f"Rows(*VALUES* regular #{len(keys)})")
            .with_statement_hint(f"Rows(*VALUES* regular epic #{len(keys)})")
        )
    return await mdb.fetch_all(query)


@sentry_span
async def fetch_jira_issues_by_prs(
    pr_nodes: Collection[int],
    jira_ids: JIRAConfig,
    meta_ids: tuple[int, ...],
    mdb: DatabaseLike,
) -> md.DataFrame:
    """Load brief information about JIRA issues mapped to the given PRs."""
    assert jira_ids is not None
    regiss = aliased(Issue, name="regular")
    epiciss = aliased(Issue, name="epic")
    prmap = aliased(NodePullRequestJiraIssues, name="m")
    in_any_values = len(pr_nodes) > 100
    selected = [prmap.node_id.label("node_id"), regiss.key.label("jira_id")]
    query = (
        sql.select(*selected)
        .select_from(
            sql.outerjoin(
                sql.join(
                    regiss,
                    prmap,
                    sql.and_(
                        regiss.id == prmap.jira_id,
                        regiss.acc_id == prmap.jira_acc,
                    ),
                ),
                epiciss,
                sql.and_(
                    epiciss.id == regiss.epic_id,
                    epiciss.acc_id == regiss.acc_id,
                ),
            ),
        )
        .where(
            prmap.node_acc.in_(meta_ids),
            prmap.jira_acc == jira_ids[0],
            prmap.node_id.in_any_values(pr_nodes)
            if in_any_values
            else prmap.node_id.in_(pr_nodes),
            regiss.project_id.in_(jira_ids[1]),
            regiss.is_deleted.is_(False),
        )
        .order_by(prmap.node_id)
    )
    if in_any_values:
        (
            query.with_statement_hint("Leading((((m *VALUES*) regular) epic))")
            .with_statement_hint(f"Rows(m *VALUES* #{len(pr_nodes)})")
            .with_statement_hint(f"Rows(m *VALUES* regular #{len(pr_nodes)})")
        )
    return await read_sql_query(query, mdb, selected)


@sentry_span
async def import_components_as_labels(issues: md.DataFrame, mdb: DatabaseLike) -> None:
    """Import the components names as labels in the issues dataframe.

    The `issues` dataframe must have `acc_id` and `components` (with components ids) columns.

    """
    if issues.empty:
        return
    conditions = []
    accounts_col = issues[Issue.acc_id.name]
    components_col = issues[Issue.components.name]
    for indexes in issues.groupby(Issue.acc_id.name):
        acc = accounts_col[indexes[0]]
        components = components_col[indexes]
        vals = set(chain.from_iterable(components[is_not_null(components)]))
        conditions.append(sql.and_(Component.id.in_(vals), Component.acc_id == int(acc)))
    df = await read_sql_query(
        sql.select(Component.acc_id, Component.id, func.lower(Component.name)).where(
            sql.or_(*conditions),
        ),
        mdb,
        [Component.acc_id, Component.id, Component.name],
    )
    cmap: dict[int, dict[str, str]] = {}
    for indexes in df.groupby(Component.acc_id.name):
        cmap.setdefault(df[Component.acc_id.name][indexes[0]], {}).update(
            zip(df[Component.id.name][indexes], df[Component.name.name][indexes]),
        )
    labels_col = issues[Issue.labels.name]
    for i, (acc_id, row_labels, row_components) in enumerate(
        issues.iterrows(Issue.acc_id.name, Issue.labels.name, Issue.components.name),
    ):
        if row_labels is None:
            labels_col[i] = row_labels = []
        else:
            for j, s in enumerate(row_labels):
                row_labels[j] = s.lower()
        if row_components is not None:
            row_labels.extend(cmap[acc_id][c] for c in row_components)


participant_columns = (
    func.lower(Issue.reporter_display_name).label("reporter"),
    func.lower(Issue.assignee_display_name).label("assignee"),
    func.coalesce(Issue.commenters_display_names, EmptyTextArray()).label("commenters"),
)
