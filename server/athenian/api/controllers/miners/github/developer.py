from datetime import datetime, timedelta
from enum import Enum
from itertools import chain
from typing import Collection, FrozenSet, List, Optional, Set, Tuple, Type, Union

import aiomcache
import databases
import numpy as np
import pandas as pd
from sqlalchemy import and_, exists, func, not_, select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.async_utils import gather, read_sql_query
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.release_load import ReleaseLoader
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.controllers.settings import LogicalRepositorySettings, ReleaseSettings
from athenian.api.models.metadata.github import PullRequest, PullRequestComment, \
    PullRequestLabel, PullRequestReview, PullRequestReviewComment, PushCommit, Release, \
    Repository, User
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import dataclass


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


def _filter_by_labels(other_model_acc_id: InstrumentedAttribute,
                      other_model_pull_request_node_id: InstrumentedAttribute,
                      labels: LabelFilter,
                      filters: list) -> bool:
    singles, multiples = LabelFilter.split(labels.include)
    embedded_labels_query = not multiples
    if all_in_labels := (set(singles + list(chain.from_iterable(multiples)))):
        filters.append(
            exists().where(and_(
                PullRequestLabel.acc_id == other_model_acc_id,
                PullRequestLabel.pull_request_node_id == other_model_pull_request_node_id,
                func.lower(PullRequestLabel.name).in_(all_in_labels),
            )))
    if labels.exclude:
        filters.append(
            not_(exists().where(and_(
                PullRequestLabel.acc_id == other_model_acc_id,
                PullRequestLabel.pull_request_node_id == other_model_pull_request_node_id,
                func.lower(PullRequestLabel.name).in_(labels.exclude),
            ))))
    assert embedded_labels_query, "TODO: we don't support label combinations yet"
    return embedded_labels_query


async def _mine_commits(repo_ids: np.ndarray,
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
                        mdb: databases.Database,
                        pdb: databases.Database,
                        rdb: databases.Database,
                        cache: Optional[aiomcache.Client],
                        ) -> pd.DataFrame:
    columns = [PushCommit.author_user_id.label(developer_identity_column),
               PushCommit.repository_node_id.label(developer_repository_column),
               PushCommit.committed_date]
    if DeveloperTopic.lines_changed in topics:
        columns.append(
            (PushCommit.additions + PushCommit.deletions).label(developer_changed_lines_column))
    query = select(columns).where(and_(
        PushCommit.committed_date.between(time_from, time_to),
        PushCommit.author_user_id.in_(dev_ids),
        PushCommit.repository_node_id.in_(repo_ids),
        PushCommit.acc_id.in_(meta_ids),
    ))
    return await read_sql_query(query, mdb, columns)


async def _mine_prs(attr_user: InstrumentedAttribute,
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
                    mdb: databases.Database,
                    pdb: databases.Database,
                    rdb: databases.Database,
                    cache: Optional[aiomcache.Client],
                    ) -> pd.DataFrame:
    selected = [attr_user.label(developer_identity_column),
                PullRequest.repository_node_id.label(developer_repository_column),
                attr_filter]
    filters = [
        attr_filter.between(time_from, time_to),
        attr_user.in_(dev_ids),
        PullRequest.repository_node_id.in_(repo_ids),
        PullRequest.acc_id.in_(meta_ids),
    ]
    if labels:
        _filter_by_labels(PullRequest.acc_id, PullRequest.node_id, labels, filters)
        if jira:
            query = await generate_jira_prs_query(
                filters, jira, mdb, cache, columns=selected, seed=PullRequest)
        else:
            query = select(selected).where(and_(*filters))
    elif jira:
        query = await generate_jira_prs_query(
            filters, jira, mdb, cache, columns=selected)
    else:
        query = select(selected).where(and_(*filters))
    return await read_sql_query(query, mdb, [c.name for c in selected])


async def _mine_prs_created(*args, **kwargs) -> pd.DataFrame:
    return await _mine_prs(PullRequest.user_node_id, PullRequest.created_at, *args, **kwargs)


async def _mine_prs_merged(*args, **kwargs) -> pd.DataFrame:
    return await _mine_prs(PullRequest.merged_by_id, PullRequest.merged_at, *args, **kwargs)


async def _mine_releases(repo_ids: np.ndarray,
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
                         mdb: databases.Database,
                         pdb: databases.Database,
                         rdb: databases.Database,
                         cache: Optional[aiomcache.Client],
                         ) -> pd.DataFrame:
    branches, default_branches = await BranchMiner.extract_branches(
        repo_names, prefixer, meta_ids, mdb, cache)
    releases, _ = await ReleaseLoader.load_releases(
        repo_names, branches, default_branches, time_from, time_to,
        release_settings, logical_settings, prefixer, account, meta_ids, mdb, pdb, rdb, cache)
    release_authors = releases[Release.author.name].values.astype("U")
    matched_devs_mask = np.in1d(release_authors, dev_names)
    return pd.DataFrame({
        developer_identity_column + _dereferenced_suffix: release_authors[matched_devs_mask],
        developer_repository_column + _dereferenced_suffix:
            releases[Release.repository_full_name.name].values[matched_devs_mask],
        Release.published_at.name: releases[Release.published_at.name].values[matched_devs_mask],
    })


async def _mine_reviews(repo_ids: np.ndarray,
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
                        mdb: databases.Database,
                        pdb: databases.Database,
                        rdb: databases.Database,
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
        _filter_by_labels(PullRequestReview.acc_id, PullRequestReview.pull_request_node_id,
                          labels, filters)
        if jira:
            query = await generate_jira_prs_query(
                filters, jira, mdb, cache, columns=selected, seed=PullRequestReview,
                on=(PullRequestReview.pull_request_node_id, PullRequestReview.acc_id))
        else:
            query = select(selected).where(and_(*filters))
    elif jira:
        query = await generate_jira_prs_query(
            filters, jira, mdb, cache, columns=selected, seed=PullRequestReview,
            on=(PullRequestReview.pull_request_node_id, PullRequestReview.acc_id))
    else:
        query = select(selected).where(and_(*filters))
    return await read_sql_query(query, mdb, selected)


async def _mine_pr_comments(model: Union[Type[PullRequestComment], Type[PullRequestReviewComment]],
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
                            mdb: databases.Database,
                            pdb: databases.Database,
                            rdb: databases.Database,
                            cache: Optional[aiomcache.Client],
                            ) -> pd.DataFrame:
    selected = [model.user_node_id.label(developer_identity_column),
                model.repository_node_id.label(developer_repository_column),
                model.created_at]
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
                filters, jira, mdb, cache, columns=selected, seed=model,
                on=(model.pull_request_node_id, model.acc_id))
        else:
            query = select(selected).where(and_(*filters))
    elif jira:
        query = await generate_jira_prs_query(
            filters, jira, mdb, cache, columns=selected, seed=model,
            on=(model.pull_request_node_id, model.acc_id))
    else:
        query = select(selected).where(and_(*filters))
    return await read_sql_query(query, mdb, [c.name for c in selected])


async def _mine_pr_comments_regular(*args, **kwargs) -> pd.DataFrame:
    return await _mine_pr_comments(PullRequestComment, *args, **kwargs)


async def _mine_pr_comments_review(*args, **kwargs) -> pd.DataFrame:
    return await _mine_pr_comments(PullRequestReviewComment, *args, **kwargs)


developer_repository_column = "repository"
developer_identity_column = "developer"
developer_changed_lines_column = "lines"
_dereferenced_suffix = "_dereferenced"

processors = [
    (frozenset((DeveloperTopic.commits_pushed,
                DeveloperTopic.lines_changed,
                DeveloperTopic.active)),
     _mine_commits),
    (frozenset((DeveloperTopic.prs_created,)), _mine_prs_created),
    (frozenset((DeveloperTopic.prs_merged,)), _mine_prs_merged),
    (frozenset((DeveloperTopic.releases,)), _mine_releases),
    (frozenset((DeveloperTopic.reviews,
                DeveloperTopic.review_approvals,
                DeveloperTopic.review_neutrals,
                DeveloperTopic.review_rejections,
                DeveloperTopic.prs_reviewed)),
     _mine_reviews),
    (frozenset((DeveloperTopic.pr_comments, DeveloperTopic.regular_pr_comments)),
     _mine_pr_comments_regular),
    (frozenset((DeveloperTopic.pr_comments, DeveloperTopic.review_pr_comments)),
     _mine_pr_comments_review),
]


async def mine_developer_activities(devs: Collection[str],
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
                                    mdb: databases.Database,
                                    pdb: databases.Database,
                                    rdb: databases.Database,
                                    cache: Optional[aiomcache.Client],
                                    ) -> List[Tuple[FrozenSet[DeveloperTopic], pd.DataFrame]]:
    """Extract pandas DataFrame-s for each topic relationship group."""
    zerotd = timedelta(0)
    assert isinstance(time_from, datetime) and time_from.tzinfo is not None and \
        time_from.tzinfo.utcoffset(time_from) == zerotd
    assert isinstance(time_to, datetime) and time_to.tzinfo is not None and \
        time_to.tzinfo.utcoffset(time_to) == zerotd
    repo_ids, repo_names, dev_ids, dev_names = await _fetch_node_ids(devs, repos, meta_ids, mdb)
    tasks = {}
    for key, miner in processors:
        if key.intersection(topics):
            tasks[key] = miner(
                repo_ids, repo_names, dev_ids, dev_names,
                time_from, time_to, topics, labels, jira, release_settings, logical_settings,
                prefixer, account, meta_ids, mdb, pdb, rdb, cache)
    df_by_topic = dict(zip(tasks.keys(), await gather(*tasks.values())))
    if DeveloperTopic.pr_comments in topics:
        regular_pr_comments = df_by_topic[(
            key := frozenset((DeveloperTopic.pr_comments, DeveloperTopic.regular_pr_comments))
        )]
        del df_by_topic[key]
        if DeveloperTopic.regular_pr_comments in topics:
            df_by_topic[frozenset((DeveloperTopic.regular_pr_comments,))] = regular_pr_comments
        review_pr_comments = df_by_topic[(
            key := frozenset((DeveloperTopic.pr_comments, DeveloperTopic.review_pr_comments))
        )]
        del df_by_topic[key]
        if DeveloperTopic.review_pr_comments in topics:
            df_by_topic[frozenset((DeveloperTopic.review_pr_comments,))] = review_pr_comments
        df_by_topic[frozenset((DeveloperTopic.pr_comments,))] = pd.concat([
            regular_pr_comments, review_pr_comments])
    # convert node IDs to user logins and repository names
    effective_df_by_topic = {}
    for key, df in df_by_topic.items():
        effective_df_by_topic[key.intersection(topics)] = df
        try:
            df[developer_identity_column] = dev_names[np.searchsorted(
                dev_ids, df[developer_identity_column].values)]
            df[developer_repository_column] = repo_names[np.searchsorted(
                repo_ids, df[developer_repository_column].values)]
        except KeyError:
            df.rename(columns={
                developer_identity_column + _dereferenced_suffix: developer_identity_column,
                developer_repository_column + _dereferenced_suffix: developer_repository_column,
            }, inplace=True, errors="raise")
    return list(effective_df_by_topic.items())


@sentry_span
async def _fetch_node_ids(devs: Collection[str],
                          repos: Collection[str],
                          meta_ids: Tuple[int, ...],
                          mdb: databases.Database,
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tasks = [
        mdb.fetch_all(select([Repository.node_id, Repository.full_name])
                      .where(and_(Repository.full_name.in_(repos),
                                  Repository.acc_id.in_(meta_ids)))
                      .order_by(Repository.node_id)),
        mdb.fetch_all(select([User.node_id, User.login])
                      .where(and_(User.login.in_(devs),
                                  User.acc_id.in_(meta_ids)))
                      .order_by(User.node_id)),
    ]
    repo_id_rows, dev_id_rows = await gather(*tasks)
    repo_ids = np.fromiter((r[0] for r in repo_id_rows), int, len(repo_id_rows))
    repo_names = np.array([r[1] for r in repo_id_rows], dtype="U")
    dev_ids = np.fromiter((r[0] for r in dev_id_rows), int, len(dev_id_rows))
    dev_names = np.array([r[1] for r in dev_id_rows], dtype="U")
    return repo_ids, repo_names, dev_ids, dev_names
