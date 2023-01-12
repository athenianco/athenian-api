from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from itertools import chain
from typing import Collection, FrozenSet, Iterable, List, Optional, Set, Tuple, Type, Union

import aiomcache
import morcilla
import numpy as np
import pandas as pd
from sqlalchemy import and_, exists, func, not_, select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.async_utils import gather, read_sql_query
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.release_load import ReleaseLoader
from athenian.api.internal.miners.jira.issue import generate_jira_prs_query
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings
from athenian.api.models.metadata.github import (
    NodePullRequest,
    PullRequestComment,
    PullRequestLabel,
    PullRequestReview,
    PullRequestReviewComment,
    PushCommit,
    Release,
    Repository,
    User,
)
from athenian.api.tracing import sentry_span


class DeveloperTopic(Enum):
    """Possible developer statistics kinds."""

    commits_pushed = "dev-commits-pushed"
    lines_changed = "dev-lines-changed"
    prs_created = "dev-prs-created"
    prs_reviewed = "dev-prs-reviewed"
    prs_merged = "dev-prs-merged"
    releases = "dev-releases"
    reviews = "dev-reviews"
    review_approvals = "dev-review-approvals"
    review_rejections = "dev-review-rejections"
    review_neutrals = "dev-review-neutrals"
    pr_comments = "dev-pr-comments"
    regular_pr_comments = "dev-regular-pr-comments"
    review_pr_comments = "dev-review-pr-comments"
    active = "dev-active"
    active0 = "dev-active0"
    worked = "dev-worked"

    def __lt__(self, other: "DeveloperTopic") -> bool:
        """Support sorting."""
        return self.value < other.value


@dataclass(slots=True, frozen=True)
class DeveloperStats:
    """Calculated statistics about developer activities."""

    commits_pushed: int = 0
    lines_changed: int = 0
    prs_created: int = 0
    prs_reviewed: int = 0
    prs_merged: int = 0
    releases: int = 0
    reviews: int = 0
    review_approvals: int = 0
    review_rejections: int = 0
    review_neutrals: int = 0
    pr_comments: int = 0
    regular_pr_comments: int = 0
    review_pr_comments: int = 0
    active: int = 0


def _filter_by_labels(
    other_model_acc_id: InstrumentedAttribute,
    other_model_pull_request_node_id: InstrumentedAttribute,
    labels: LabelFilter,
    filters: list,
) -> bool:
    singles, multiples = LabelFilter.split(labels.include)
    embedded_labels_query = not multiples
    if all_in_labels := (set(singles + list(chain.from_iterable(multiples)))):
        filters.append(
            exists().where(
                PullRequestLabel.acc_id == other_model_acc_id,
                PullRequestLabel.pull_request_node_id == other_model_pull_request_node_id,
                func.lower(PullRequestLabel.name).in_(all_in_labels),
            ),
        )
    if labels.exclude:
        filters.append(
            not_(
                exists().where(
                    PullRequestLabel.acc_id == other_model_acc_id,
                    PullRequestLabel.pull_request_node_id == other_model_pull_request_node_id,
                    func.lower(PullRequestLabel.name).in_(labels.exclude),
                ),
            ),
        )
    assert embedded_labels_query, "TODO: we don't support label combinations yet"
    return embedded_labels_query


@sentry_span
async def _mine_commits(
    repo_ids: np.ndarray,
    repo_names: np.ndarray,
    dev_ids: np.ndarray,
    dev_names: np.ndarray,
    time_from: datetime,
    time_to: datetime,
    topics: Set[DeveloperTopic],
    labels: LabelFilter,
    jira: JIRAFilter,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    mdb: morcilla.Database,
    pdb: morcilla.Database,
    rdb: morcilla.Database,
    cache: Optional[aiomcache.Client],
) -> pd.DataFrame:
    if labels or jira:
        # TODO(vmarkovtsev): filter PRs, take merge commits, load the DAGs, find the PR commits
        # We cannot rely on PullRequestCommit because there can be duplicates after force pushes;
        # on the other hand, "rebase/squash and merge" erases information about individual commits
        return pd.DataFrame(
            columns=[
                developer_identity_column,
                developer_repository_column,
                PushCommit.committed_date.name,
            ],
        )
    columns = [
        PushCommit.author_user_id.label(developer_identity_column),
        PushCommit.repository_node_id.label(developer_repository_column),
        PushCommit.committed_date,
    ]
    if DeveloperTopic.lines_changed in topics:
        columns.append(
            (PushCommit.additions + PushCommit.deletions).label(developer_changed_lines_column),
        )
    query = (
        select(columns)
        .where(
            and_(
                PushCommit.committed_date.between(time_from, time_to),
                PushCommit.author_user_id.in_any_values(dev_ids),
                PushCommit.repository_node_id.in_(repo_ids),
                PushCommit.acc_id.in_(meta_ids),
            ),
        )
        .with_statement_hint("IndexOnlyScan(cmm github_node_commit_check_runs)")
        .with_statement_hint("Leading(((((cmm *VALUES*) repo) ath) cath))")
        .with_statement_hint("Rows(cmm *VALUES* *1000)")
        .with_statement_hint("Rows(cmm *VALUES* repo *10000)")
        .with_statement_hint("HashJoin(cmm *VALUES* repo)")
    )
    return await read_sql_query(query, mdb, columns)


async def _mine_prs(
    attr_user: InstrumentedAttribute,
    attr_filter: InstrumentedAttribute,
    repo_ids: np.ndarray,
    repo_names: np.ndarray,
    dev_ids: np.ndarray,
    dev_names: np.ndarray,
    time_from: datetime,
    time_to: datetime,
    topics: Set[DeveloperTopic],
    labels: LabelFilter,
    jira: JIRAFilter,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    mdb: morcilla.Database,
    pdb: morcilla.Database,
    rdb: morcilla.Database,
    cache: Optional[aiomcache.Client],
    hints: Iterable[str] = (),
) -> pd.DataFrame:
    selected = [
        attr_user.label(developer_identity_column),
        NodePullRequest.repository_id.label(developer_repository_column),
        attr_filter,
    ]
    filters = [
        attr_filter.between(time_from, time_to),
        attr_user.in_(dev_ids),
        NodePullRequest.repository_id.in_(repo_ids),
        NodePullRequest.acc_id.in_(meta_ids),
    ]
    if labels:
        _filter_by_labels(NodePullRequest.acc_id, NodePullRequest.node_id, labels, filters)
        if jira:
            query = await generate_jira_prs_query(
                filters,
                jira,
                None,
                mdb,
                cache,
                columns=selected,
                seed=NodePullRequest,
                on=(NodePullRequest.node_id, NodePullRequest.acc_id),
            )
        else:
            query = select(selected).where(and_(*filters))
    elif jira:
        query = await generate_jira_prs_query(
            filters,
            jira,
            None,
            mdb,
            cache,
            columns=selected,
            seed=NodePullRequest,
            on=(NodePullRequest.node_id, NodePullRequest.acc_id),
        )
    else:
        query = select(selected).where(and_(*filters))
        for hint in hints:
            query = query.with_statement_hint(hint)
    return await read_sql_query(query, mdb, [c.name for c in selected])


@sentry_span
async def _mine_prs_created(*args, **kwargs) -> pd.DataFrame:
    return await _mine_prs(
        NodePullRequest.user_node_id,
        NodePullRequest.created_at,
        *args,
        **kwargs,
        hints=[
            f"IndexOnlyScan({NodePullRequest.__tablename__} "
            "github_node_pull_request_author_created)",
        ],
    )


@sentry_span
async def _mine_prs_merged(*args, **kwargs) -> pd.DataFrame:
    return await _mine_prs(
        NodePullRequest.merged_by_id,
        NodePullRequest.merged_at,
        *args,
        **kwargs,
        hints=[
            f"IndexOnlyScan({NodePullRequest.__tablename__} "
            "github_node_pull_request_author_merge_cover)",
        ],
    )


@sentry_span
async def _mine_releases(
    repo_ids: np.ndarray,
    repo_names: np.ndarray,
    dev_ids: np.ndarray,
    dev_names: np.ndarray,
    time_from: datetime,
    time_to: datetime,
    topics: Set[DeveloperTopic],
    labels: LabelFilter,
    jira: JIRAFilter,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    mdb: morcilla.Database,
    pdb: morcilla.Database,
    rdb: morcilla.Database,
    cache: Optional[aiomcache.Client],
) -> pd.DataFrame:
    if labels or jira:
        return pd.DataFrame(
            columns=[
                developer_identity_column + _dereferenced_suffix,
                developer_repository_column + _dereferenced_suffix,
                Release.published_at.name,
            ],
        )
    branches, default_branches = await BranchMiner.load_branches(
        repo_names, prefixer, account, meta_ids, mdb, pdb, cache,
    )
    releases, _ = await ReleaseLoader.load_releases(
        repo_names,
        branches,
        default_branches,
        time_from,
        time_to,
        release_settings,
        logical_settings,
        prefixer,
        account,
        meta_ids,
        mdb,
        pdb,
        rdb,
        cache,
    )
    release_authors = releases[Release.author.name].values.astype("U")
    matched_devs_mask = np.in1d(release_authors, dev_names)
    return pd.DataFrame(
        {
            developer_identity_column + _dereferenced_suffix: release_authors[matched_devs_mask],
            developer_repository_column
            + _dereferenced_suffix: releases[Release.repository_full_name.name].values[
                matched_devs_mask
            ],
            Release.published_at.name: releases[Release.published_at.name].values[
                matched_devs_mask
            ],
        },
    )


@sentry_span
async def _mine_reviews(
    repo_ids: np.ndarray,
    repo_names: np.ndarray,
    dev_ids: np.ndarray,
    dev_names: np.ndarray,
    time_from: datetime,
    time_to: datetime,
    topics: Set[DeveloperTopic],
    labels: LabelFilter,
    jira: JIRAFilter,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    mdb: morcilla.Database,
    pdb: morcilla.Database,
    rdb: morcilla.Database,
    cache: Optional[aiomcache.Client],
) -> pd.DataFrame:
    selected = [
        PullRequestReview.user_node_id.label(developer_identity_column),
        PullRequestReview.repository_node_id.label(developer_repository_column),
        PullRequestReview.pull_request_node_id,
        PullRequestReview.submitted_at,
        PullRequestReview.state,
    ]
    filters = [
        PullRequestReview.acc_id.in_(meta_ids),
        PullRequestReview.submitted_at.between(time_from, time_to),
        PullRequestReview.user_node_id.in_(dev_ids),
        PullRequestReview.repository_node_id.in_(repo_ids),
    ]
    if labels:
        _filter_by_labels(
            PullRequestReview.acc_id, PullRequestReview.pull_request_node_id, labels, filters,
        )
        if jira:
            query = await generate_jira_prs_query(
                filters,
                jira,
                None,
                mdb,
                cache,
                columns=selected,
                seed=PullRequestReview,
                on=(PullRequestReview.pull_request_node_id, PullRequestReview.acc_id),
            )
        else:
            query = select(selected).where(and_(*filters))
    elif jira:
        query = await generate_jira_prs_query(
            filters,
            jira,
            None,
            mdb,
            cache,
            columns=selected,
            seed=PullRequestReview,
            on=(PullRequestReview.pull_request_node_id, PullRequestReview.acc_id),
        )
    else:
        query = select(selected).where(and_(*filters))
    return await read_sql_query(query, mdb, selected)


async def _mine_pr_comments(
    model: Union[Type[PullRequestComment], Type[PullRequestReviewComment]],
    repo_ids: np.ndarray,
    repo_names: np.ndarray,
    dev_ids: np.ndarray,
    dev_names: np.ndarray,
    time_from: datetime,
    time_to: datetime,
    topics: Set[DeveloperTopic],
    labels: LabelFilter,
    jira: JIRAFilter,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    mdb: morcilla.Database,
    pdb: morcilla.Database,
    rdb: morcilla.Database,
    cache: Optional[aiomcache.Client],
) -> pd.DataFrame:
    selected = [
        model.user_node_id.label(developer_identity_column),
        model.repository_node_id.label(developer_repository_column),
        model.created_at,
    ]
    filters = [
        model.acc_id.in_(meta_ids),
        model.created_at.between(time_from, time_to),
        model.user_node_id.in_(dev_ids),
        model.repository_node_id.in_(repo_ids),
    ]
    if labels:
        _filter_by_labels(model.acc_id, model.pull_request_node_id, labels, filters)
        if jira:
            query = await generate_jira_prs_query(
                filters,
                jira,
                None,
                mdb,
                cache,
                columns=selected,
                seed=model,
                on=(model.pull_request_node_id, model.acc_id),
            )
        else:
            query = select(selected).where(and_(*filters))
    elif jira:
        query = await generate_jira_prs_query(
            filters,
            jira,
            None,
            mdb,
            cache,
            columns=selected,
            seed=model,
            on=(model.pull_request_node_id, model.acc_id),
        )
    else:
        query = select(selected).where(and_(*filters))
    return await read_sql_query(query, mdb, [c.name for c in selected])


@sentry_span
async def _mine_pr_comments_regular(*args, **kwargs) -> pd.DataFrame:
    return await _mine_pr_comments(PullRequestComment, *args, **kwargs)


@sentry_span
async def _mine_pr_comments_review(*args, **kwargs) -> pd.DataFrame:
    return await _mine_pr_comments(PullRequestReviewComment, *args, **kwargs)


developer_repository_column = "repository"
developer_identity_column = "developer"
developer_changed_lines_column = "lines"
_dereferenced_suffix = "_dereferenced"

processors = [
    (
        frozenset(
            (
                DeveloperTopic.commits_pushed,
                DeveloperTopic.lines_changed,
                DeveloperTopic.active,
                DeveloperTopic.active0,
                DeveloperTopic.worked,
            ),
        ),
        _mine_commits,
    ),
    (frozenset((DeveloperTopic.prs_created, DeveloperTopic.worked)), _mine_prs_created),
    (frozenset((DeveloperTopic.prs_merged, DeveloperTopic.worked)), _mine_prs_merged),
    (frozenset((DeveloperTopic.releases, DeveloperTopic.worked)), _mine_releases),
    (
        frozenset(
            (
                DeveloperTopic.reviews,
                DeveloperTopic.review_approvals,
                DeveloperTopic.review_neutrals,
                DeveloperTopic.review_rejections,
                DeveloperTopic.prs_reviewed,
                DeveloperTopic.worked,
            ),
        ),
        _mine_reviews,
    ),
    (
        frozenset(
            (
                DeveloperTopic.pr_comments,
                DeveloperTopic.regular_pr_comments,
                DeveloperTopic.worked,
            ),
        ),
        _mine_pr_comments_regular,
    ),
    (
        frozenset((DeveloperTopic.pr_comments, DeveloperTopic.review_pr_comments)),
        _mine_pr_comments_review,
    ),
]


@sentry_span
async def mine_developer_activities(
    devs: Collection[str],
    repos: Collection[str],
    time_from: datetime,
    time_to: datetime,
    topics: Set[DeveloperTopic],
    labels: LabelFilter,
    jira: JIRAFilter,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: Tuple[int, ...],
    mdb: morcilla.Database,
    pdb: morcilla.Database,
    rdb: morcilla.Database,
    cache: Optional[aiomcache.Client],
) -> List[Tuple[FrozenSet[DeveloperTopic], pd.DataFrame]]:
    """Extract pandas DataFrame-s for each topic relationship group."""
    zerotd = timedelta(0)
    assert (
        isinstance(time_from, datetime)
        and time_from.tzinfo is not None
        and time_from.tzinfo.utcoffset(time_from) == zerotd
    )
    assert (
        isinstance(time_to, datetime)
        and time_to.tzinfo is not None
        and time_to.tzinfo.utcoffset(time_to) == zerotd
    )
    repo_ids, repo_names, dev_ids, dev_names = await _fetch_node_ids(devs, repos, meta_ids, mdb)
    tasks = {}
    for key, miner in processors:
        if key.intersection(topics):
            tasks[key] = miner(
                repo_ids,
                repo_names,
                dev_ids,
                dev_names,
                time_from,
                time_to,
                topics,
                labels,
                jira,
                release_settings,
                logical_settings,
                prefixer,
                account,
                meta_ids,
                mdb,
                pdb,
                rdb,
                cache,
            )
    df_by_topic = dict(zip(tasks.keys(), await gather(*tasks.values())))
    # convert node IDs to user logins and repository names
    for key, df in df_by_topic.items():
        try:
            df[developer_identity_column] = dev_names[
                np.searchsorted(dev_ids, df[developer_identity_column].values)
            ]
            df[developer_repository_column] = repo_names[
                np.searchsorted(repo_ids, df[developer_repository_column].values)
            ]
        except IndexError as e:
            raise AssertionError(str(key)) from e
        except KeyError:
            df.rename(
                columns={
                    developer_identity_column + _dereferenced_suffix: developer_identity_column,
                    developer_repository_column
                    + _dereferenced_suffix: developer_repository_column,
                },
                inplace=True,
                errors="raise",
            )
    new_df_by_topic = {}
    worked_dfs = []
    for key, val in df_by_topic.items():
        if DeveloperTopic.worked in key:
            worked_dfs.append(val)
            new_df_by_topic[key - {DeveloperTopic.worked}] = val
        else:
            new_df_by_topic[key] = val
    df_by_topic = new_df_by_topic
    if DeveloperTopic.worked in topics:
        df_by_topic[frozenset((DeveloperTopic.worked,))] = pd.concat(worked_dfs)
    if DeveloperTopic.pr_comments in topics:
        regular_pr_comments = df_by_topic[
            (key := frozenset((DeveloperTopic.pr_comments, DeveloperTopic.regular_pr_comments)))
        ]
        del df_by_topic[key]
        if DeveloperTopic.regular_pr_comments in topics:
            df_by_topic[frozenset((DeveloperTopic.regular_pr_comments,))] = regular_pr_comments
        review_pr_comments = df_by_topic[
            (key := frozenset((DeveloperTopic.pr_comments, DeveloperTopic.review_pr_comments)))
        ]
        del df_by_topic[key]
        if DeveloperTopic.review_pr_comments in topics:
            df_by_topic[frozenset((DeveloperTopic.review_pr_comments,))] = review_pr_comments
        df_by_topic[frozenset((DeveloperTopic.pr_comments,))] = pd.concat(
            [regular_pr_comments, review_pr_comments],
        )
    effective_df_by_topic = {}
    for key, df in df_by_topic.items():
        if not (new_key := key.intersection(topics)):
            continue
        effective_df_by_topic[new_key] = df
    return list(effective_df_by_topic.items())


@sentry_span
async def _fetch_node_ids(
    devs: Collection[str],
    repos: Collection[str],
    meta_ids: Tuple[int, ...],
    mdb: morcilla.Database,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tasks = [
        mdb.fetch_all(
            select([Repository.node_id, Repository.full_name])
            .where(and_(Repository.full_name.in_(repos), Repository.acc_id.in_(meta_ids)))
            .order_by(Repository.node_id),
        ),
        mdb.fetch_all(
            select([User.node_id, User.login])
            .where(and_(User.login.in_(devs), User.acc_id.in_(meta_ids)))
            .order_by(User.node_id),
        ),
    ]
    repo_id_rows, dev_id_rows = await gather(*tasks)
    repo_ids = np.fromiter((r[0] for r in repo_id_rows), int, len(repo_id_rows))
    repo_names = np.array([r[1] for r in repo_id_rows], dtype="U")
    dev_ids = np.fromiter((r[0] for r in dev_id_rows), int, len(dev_id_rows))
    dev_names = np.array([r[1] for r in dev_id_rows], dtype="U")
    return repo_ids, repo_names, dev_ids, dev_names
