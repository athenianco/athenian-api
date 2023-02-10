import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import chain, groupby, product, repeat
import json
import logging
from operator import attrgetter
from typing import Any, Collection, Iterator, KeysView, Mapping, NamedTuple, Optional, Sequence

import aiomcache
import numpy as np
from numpy import typing as npt
import pandas as pd
import sentry_sdk
import sqlalchemy as sa
from sqlalchemy import (
    and_,
    desc,
    distinct,
    exists,
    func,
    join,
    literal_column,
    not_,
    or_,
    select,
    union_all,
)
from sqlalchemy.orm import aliased
from sqlalchemy.sql import Executable, Select
from sqlalchemy.sql.elements import UnaryExpression

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import CancelCache, cached, short_term_exptime
from athenian.api.db import (
    Database,
    DatabaseLike,
    add_pdb_hits,
    add_pdb_misses,
    ensure_db_datetime_tz,
    insert_or_ignore,
)
from athenian.api.defer import defer
from athenian.api.internal.jira import JIRAConfig
from athenian.api.internal.logical_repos import coerce_logical_repos, drop_logical_repo
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.commit import (
    COMMIT_FETCH_COMMITS_COLUMNS,
    DAG,
    fetch_dags_with_commits,
    fetch_repository_commits,
)
from athenian.api.internal.miners.github.dag_accelerated import (
    extract_independent_ownership,
    extract_pr_commits,
    mark_dag_access,
    mark_dag_parents,
    searchsorted_inrange,
)
from athenian.api.internal.miners.github.deployment_accelerated import split_prs_to_jira_ids
from athenian.api.internal.miners.github.deployment_light import (
    fetch_components_and_prune_unresolved,
    fetch_deployment_candidates,
    fetch_labels,
)
from athenian.api.internal.miners.github.label import (
    fetch_labels_to_filter,
    find_left_prs_by_labels,
)
from athenian.api.internal.miners.github.logical import (
    split_logical_deployed_components,
    split_logical_prs,
)
from athenian.api.internal.miners.github.precomputed_releases import (
    compose_release_match,
    reverse_release_settings,
)
from athenian.api.internal.miners.github.rebased_pr import match_rebased_prs
from athenian.api.internal.miners.github.release_load import (
    ReleaseLoader,
    dummy_releases_df,
    set_matched_by_from_release_match,
    unfresh_releases_lag,
)
from athenian.api.internal.miners.github.release_mine import (
    group_hashes_by_ownership,
    mine_releases_by_ids,
)
from athenian.api.internal.miners.jira.issue import (
    fetch_jira_issues_by_keys,
    fetch_jira_issues_by_prs,
    generate_jira_prs_query,
)
from athenian.api.internal.miners.participation import (
    ReleaseParticipants,
    ReleaseParticipationKind,
)
from athenian.api.internal.miners.types import (
    DeploymentConclusion,
    DeploymentFacts,
    JIRAEntityToFetch,
    PullRequestJIRAIssueItem,
    ReleaseFacts,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import (
    LogicalDeploymentSettings,
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseMatchSetting,
    ReleaseSettings,
)
from athenian.api.models.metadata.github import (
    NodeCommit,
    NodePullRequest,
    NodeRepository,
    PullRequest,
    PullRequestLabel,
    PushCommit,
    Release,
    Repository,
)
from athenian.api.models.persistentdata.models import (
    DeployedComponent,
    DeployedLabel,
    DeploymentNotification,
    HealthMetric,
    ReleaseNotification,
)
from athenian.api.models.precomputed.models import (
    GitHubCommitDeployment,
    GitHubDeploymentFacts,
    GitHubPullRequestDeployment,
    GitHubRebasedPullRequest,
    GitHubRelease as PrecomputedRelease,
    GitHubReleaseDeployment,
)
from athenian.api.object_arrays import array_of_objects, is_not_null, nested_lengths
from athenian.api.pandas_io import deserialize_args, serialize_args
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import df_from_structs
from athenian.api.unordered_unique import in1d_str, map_array_values, unordered_unique


@dataclass(slots=True)
class MineDeploymentsMetrics:
    """Deployment mining error statistics."""

    count: int
    unresolved: int
    broken: int

    @classmethod
    def empty(cls) -> "MineDeploymentsMetrics":
        """Initialize a new MineDeploymentsMetrics instance filled with zeros."""
        return MineDeploymentsMetrics(0, 0, 0)

    def as_db(self) -> Iterator[HealthMetric]:
        """Generate HealthMetric-s from this instance."""
        yield HealthMetric(name="deployments_count", value=self.count)
        yield HealthMetric(name="deployments_unresolved", value=self.unresolved)
        yield HealthMetric(name="deployments_broken", value=self.unresolved)


async def mine_deployments(
    repositories: Collection[str],
    participants: ReleaseParticipants,
    time_from: datetime,
    time_to: datetime,
    environments: Collection[str],
    conclusions: Collection[DeploymentConclusion],
    with_labels: Mapping[str, Any],
    without_labels: Mapping[str, Any],
    pr_labels: LabelFilter,
    jira: JIRAFilter,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    branches: pd.DataFrame,
    default_branches: dict[str, str],
    prefixer: Prefixer,
    account: int,
    jira_ids: Optional[JIRAConfig],
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
    with_extended_prs: bool = False,
    with_jira: bool = False,
    eager_filter_repositories: bool = True,
    metrics: Optional[MineDeploymentsMetrics] = None,
) -> pd.DataFrame:
    """Gather facts about deployments that satisfy the specified filters.

    This is a join()-ing wrapper around _mine_deployments(). The reason why we split the code
    is that serialization of independent DataFrame-s is much more efficient than the final
    DataFrame with nested DataFrame-s.

    :param eager_filter_repositories: Value indicating whether we should exclude deployments \
    where all specified repositories having 0 deployed commits, even if they own corresponding \
    components.

    :return: Deployment stats with deployed components and releases sub-dataframes.
    """
    (
        notifications,
        components,
        facts,
        labels,
        releases,
        extended_prs,
        jira_df,
        *_,
    ) = await _mine_deployments(
        repositories,
        participants,
        time_from,
        time_to,
        environments,
        conclusions,
        with_labels,
        without_labels,
        pr_labels,
        jira,
        release_settings,
        logical_settings,
        branches,
        default_branches,
        prefixer,
        account,
        jira_ids,
        meta_ids,
        mdb,
        pdb,
        rdb,
        cache,
        with_extended_prs,
        with_jira,
        eager_filter_repositories,
        metrics,
    )
    if notifications.empty:
        return pd.DataFrame()
    components = _group_components(components)
    labels = _group_labels(labels)
    releases = [_group_releases(releases)] if not releases.empty else []
    if jira_df.empty:
        empty_stub = array_of_objects(len(facts), [])
        jira_df = pd.DataFrame(
            {
                DeploymentFacts.f.jira_ids: empty_stub,
                DeploymentFacts.f.jira_offsets: empty_stub,
                DeploymentFacts.f.prs_jira_ids: empty_stub,
                DeploymentFacts.f.prs_jira_offsets: empty_stub,
            },
        )
        jira_df.index = facts.index
    # fmt: off
    joined = (
        notifications
        .join([components, facts], how="inner")  # critical info
        .join([labels, extended_prs, jira_df] + releases)
    )
    # fmt: on
    joined = _adjust_empty_df(joined, "releases")
    joined["labels"] = joined["labels"].astype(object, copy=False)
    no_labels = joined["labels"].isnull().values  # can be NaN-s
    subst = np.empty(no_labels.sum(), dtype=object)
    subst.fill(pd.DataFrame())
    joined["labels"].values[no_labels] = subst

    if metrics is not None:
        metrics.count = len(joined)

    return joined


def _postprocess_deployments(
    result: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        bool,
        bool,
    ],
    with_extended_prs: bool = False,
    with_jira: bool = False,
    **_,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    bool,
    bool,
]:
    *dfs, cached_with_extended_prs, cached_with_jira = result
    if with_extended_prs and not cached_with_extended_prs:
        raise CancelCache()
    if with_jira and not cached_with_jira:
        raise CancelCache()
    return (*dfs, with_extended_prs, with_jira)


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=serialize_args,
    deserialize=deserialize_args,
    key=lambda repositories, participants, time_from, time_to, environments, conclusions, with_labels, without_labels, pr_labels, jira, release_settings, logical_settings, default_branches, eager_filter_repositories, **_: (  # noqa
        ",".join(sorted(repositories)),
        ",".join(
            "%s: %s" % (k, "+".join(str(p) for p in sorted(v)))
            for k, v in sorted(participants.items())
        ),
        time_from.timestamp(),
        time_to.timestamp(),
        ",".join(sorted(environments)),
        ",".join(sorted(str(c) for c in conclusions)),
        ",".join(f"{k}:{v}" for k, v in sorted(with_labels.items())),
        ",".join(f"{k}:{v}" for k, v in sorted(without_labels.items())),
        pr_labels,
        jira,
        release_settings,
        logical_settings,
        ",".join("%s: %s" % p for p in sorted(default_branches.items())),
        "1" if eager_filter_repositories else "0",
    ),
    postprocess=_postprocess_deployments,
)
async def _mine_deployments(
    repositories: Collection[str],
    participants: ReleaseParticipants,
    time_from: datetime,
    time_to: datetime,
    environments: Collection[str],
    conclusions: Collection[DeploymentConclusion],
    with_labels: Mapping[str, Any],
    without_labels: Mapping[str, Any],
    pr_labels: LabelFilter,
    jira: JIRAFilter,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    branches: pd.DataFrame,
    default_branches: dict[str, str],
    prefixer: Prefixer,
    account: int,
    jira_ids: Optional[JIRAConfig],
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
    with_extended_prs: bool,
    with_jira: bool,
    eager_filter_repositories: bool,
    metrics: Optional[MineDeploymentsMetrics],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    bool,
    bool,
]:
    log = logging.getLogger(f"{metadata.__package__}.mine_deployments")
    if not isinstance(repositories, (set, frozenset, KeysView)):
        repositories = set(repositories)
    if repositories:
        repo_name_to_node = prefixer.repo_name_to_node.get
        repo_node_ids = sorted(repo_name_to_node(r, 0) for r in coerce_logical_repos(repositories))
        if repo_node_ids[0] == 0:
            repo_node_ids = repo_node_ids[1:]
        if not repo_node_ids:
            return (*repeat(pd.DataFrame(), 7), with_extended_prs, with_jira)
    else:
        repo_node_ids = []
    notifications = await fetch_deployment_candidates(
        repo_node_ids,
        time_from,
        time_to,
        environments,
        conclusions,
        with_labels,
        without_labels,
        account,
        rdb,
        cache,
    )
    notifications_count_before_pruning = len(notifications)
    (notifications, components), labels = await gather(
        fetch_components_and_prune_unresolved(notifications, prefixer, account, rdb),
        fetch_labels(notifications.index.values, account, rdb),
    )
    if unresolved := notifications_count_before_pruning - len(notifications):
        log.warning(
            "removed %d / %d unresolved notifications",
            unresolved,
            notifications_count_before_pruning,
        )
        if metrics is not None:
            metrics.unresolved = unresolved
    if notifications.empty:
        return (*repeat(pd.DataFrame(), 7), with_extended_prs, with_jira)

    # we must load all logical repositories at once to unambiguously process the residuals
    # (the root repository minus all the logicals)
    components = split_logical_deployed_components(
        notifications,
        labels,
        components,
        logical_settings.with_logical_deployments(repositories),
        logical_settings,
    )
    repo_names, release_settings = await _finalize_release_settings(
        notifications,
        time_from,
        time_to,
        release_settings,
        logical_settings,
        branches,
        default_branches,
        prefixer,
        account,
        meta_ids,
        mdb,
        pdb,
        rdb,
        cache,
    )
    releases = asyncio.create_task(
        _fetch_precomputed_deployed_releases(
            notifications,
            repo_names,
            release_settings,
            logical_settings,
            branches,
            default_branches,
            prefixer,
            account,
            meta_ids,
            mdb,
            pdb,
            rdb,
            cache,
        ),
        name=f"_fetch_precomputed_deployed_releases({len(notifications)})",
    )
    facts = await _fetch_precomputed_deployment_facts(
        notifications.index.values, default_branches, release_settings, account, pdb,
    )
    # facts = facts.iloc[:0]  # uncomment to disable pdb
    if with_extended_prs:
        extended_prs_task = asyncio.create_task(
            _fetch_extended_prs_for_facts(facts, meta_ids, mdb),
            name="mine_deployments/_fetch_extended_prs_for_facts",
        )
    else:
        extended_prs_task = None
    if with_jira:
        jira_task = asyncio.create_task(
            _fetch_deployed_jira(facts, jira_ids, meta_ids, mdb),
            name="mine_deployments/_fetch_deployed_jira",
        )
    else:
        jira_task = None
    hits = len(facts)
    misses = len(notifications) - hits
    if misses > 0:
        if (
            set(components[DeployedComponent.repository_node_id.name].unique())
            - set(repo_node_ids)
            or conclusions
            or with_labels
            or without_labels
        ):
            # we have to look broader so that we compute the commit ownership correctly
            full_notifications = await fetch_deployment_candidates(
                repo_node_ids, time_from, time_to, environments, [], {}, {}, account, rdb, cache,
            )
            (full_notifications, full_components), full_labels = await gather(
                fetch_components_and_prune_unresolved(full_notifications, prefixer, account, rdb),
                fetch_labels(full_notifications.index.values, account, rdb),
            )
            full_components = split_logical_deployed_components(
                full_notifications, full_labels, full_components, repositories, logical_settings,
            )
            full_facts = await _fetch_precomputed_deployment_facts(
                full_notifications.index.values, default_branches, release_settings, account, pdb,
            )
        else:
            full_notifications, full_components, full_facts = notifications, components, facts
        missed_mask = np.in1d(
            full_notifications.index.values.astype("U"),
            full_facts.index.values.astype("U"),
            assume_unique=True,
            invert=True,
        )

        invalidated = await _invalidate_precomputed_on_out_of_order_notifications(
            full_notifications, missed_mask, account, pdb,
        )
        if len(invalidated):
            facts = facts.take(
                np.flatnonzero(
                    ~in1d_str(facts[DeploymentFacts.f.name].values.astype("U"), invalidated),
                ),
            )
            missed_mask |= in1d_str(notifications.index.values.astype("U"), invalidated)
            hits = len(facts)
            misses = len(notifications) - hits
        add_pdb_hits(pdb, "deployments", hits)
        add_pdb_misses(pdb, "deployments", misses)

        full_notifications = _reduce_to_missed_notifications_if_possible(
            full_notifications, missed_mask,
        )
        (
            missed_facts,
            missed_releases,
            missed_extended_prs,
            missed_jira_df,
        ) = await _compute_deployment_facts(
            full_notifications,
            full_components,
            with_extended_prs,
            with_jira,
            release_settings,
            logical_settings,
            branches,
            default_branches,
            prefixer,
            account,
            jira_ids,
            meta_ids,
            mdb,
            pdb,
            rdb,
            cache,
        )
        if not missed_facts.empty:
            facts = pd.concat([facts, missed_facts])
    else:
        invalidated = np.array([], dtype="U")
        missed_releases = missed_extended_prs = missed_jira_df = pd.DataFrame()
        add_pdb_hits(pdb, "deployments", hits)
        add_pdb_misses(pdb, "deployments", misses)

    facts = await _filter_by_participants(facts, participants)
    if eager_filter_repositories:
        # DEV-5482: remove deployments which have 0 deployed commits of the given repositories
        facts = await _filter_by_repositories(facts, repositories)
    if pr_labels or jira:
        facts = await _filter_by_prs(facts, pr_labels, jira, meta_ids, mdb, cache)

    releases, extended_prs, jira_df = await gather(releases, extended_prs_task, jira_task)
    # releases = releases.iloc[:0]  # uncomment to disable pdb
    # invalidate release facts with previously computed  invalid deploy names
    if invalidated.size and not releases.empty:
        releases = releases.take(
            np.flatnonzero(
                ~in1d_str(
                    releases[GitHubReleaseDeployment.deployment_name.name].values.astype("U"),
                    invalidated,
                ),
            ),
        )
    if not missed_releases.empty:
        if not releases.empty:
            # there is a minuscule chance that some releases are duplicated here, we ignore it
            releases = pd.concat([releases, missed_releases])
        else:
            releases = missed_releases
    if with_extended_prs:
        if not missed_extended_prs.empty:
            if not extended_prs.empty:
                extended_prs = pd.concat([extended_prs, missed_extended_prs])
            else:
                extended_prs = missed_extended_prs
        extended_prs.columns = [f"prs_{name}" for name in extended_prs.columns]
    else:
        extended_prs = pd.DataFrame()
    if with_jira:
        if not missed_jira_df.empty:
            if not jira_df.empty:
                jira_df = pd.concat([jira_df, missed_jira_df])
            else:
                jira_df = missed_jira_df
    else:
        jira_df = pd.DataFrame()
    return (
        notifications,
        components,
        facts,
        labels,
        releases,
        extended_prs,
        jira_df,
        with_extended_prs,
        with_jira,
    )


async def _invalidate_precomputed_on_out_of_order_notifications(
    notifications: pd.DataFrame,
    missed_mask: npt.NDArray[bool],
    account: int,
    pdb: DatabaseLike,
) -> npt.NDArray[str]:
    """Invalidate precomputed facts of deployments newer than deploys not precomputed yet.

    We clear the related pdb table records (e.g., deployed PRs) for each such invalidated
    deployment.

    `missed_mask` is a bitmask relative to `notifications` stating which deployments are
    not yet precomputed.

    A unicode string array with the names of affected precomputed deployments is returned.
    """
    if len(to_invalidate := _find_invalid_precomputed_deploys(notifications, missed_mask)):
        log = logging.getLogger(f"{metadata.__package__}.mine_deployments")
        await _delete_precomputed_deployments(to_invalidate, account, pdb)
        log.warning("invalidated out-of-order deployments: %s", to_invalidate.tolist())
    return to_invalidate


async def _delete_precomputed_deployments(
    names: Collection[str],
    account: int,
    pdb: Database,
) -> None:
    stmts = []
    tables = (
        GitHubCommitDeployment,
        GitHubPullRequestDeployment,
        GitHubReleaseDeployment,
        GitHubDeploymentFacts,
    )
    for Table in tables:
        stmts.append(
            sa.delete(Table).where(Table.acc_id == account, Table.deployment_name.in_(names)),
        )
    await gather(
        *(pdb.execute(stmt) for stmt in stmts),
        op="_invalidate_precomputed_on_out_of_order_notifications/sql",
    )


def _find_invalid_precomputed_deploys(
    notifications: pd.DataFrame,
    missed_mask: npt.NDArray[bool],
) -> npt.NDArray[str]:
    chrono_order = np.argsort(notifications[DeploymentNotification.finished_at.name].values)
    envs_col = notifications[DeploymentNotification.environment.name].values[chrono_order]
    # stable sort so that we don't lose the chronological order
    env_order = np.argsort(envs_col, kind="stable")
    final_order = chrono_order[env_order]

    missed = missed_mask.take(final_order)
    detect = np.diff(missed, prepend=[0])  # -1 for out of order deployments

    # count deployments in each environment
    unique_envs, env_counts = np.unique(envs_col[env_order], return_counts=True)
    env_borders = np.cumsum(env_counts)
    detect[env_borders[:-1]] = 0  # remove -1 on environment borders

    # distinguish the environments
    detect *= np.repeat(np.arange(1, len(unique_envs) + 1), env_counts)
    deltas, first_positions = np.unique(detect, return_index=True)
    first_positions = first_positions[deltas < 0][::-1]

    # find the nearest greater border index to each first occurrence position
    last_positions = env_borders[np.searchsorted(env_borders, first_positions, side="right")]

    # https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
    invalid_counts = last_positions - first_positions
    ranges = np.repeat(last_positions - invalid_counts.cumsum(), invalid_counts) + np.arange(
        invalid_counts.sum(),
    )

    return notifications.index.values[final_order[ranges]].astype("U")


@sentry_span
def _reduce_to_missed_notifications_if_possible(
    notifications: pd.DataFrame,
    missed_mask: np.ndarray,
) -> pd.DataFrame:
    _, env_indexes, env_counts = np.unique(
        notifications[DeploymentNotification.environment.name].values,
        return_counts=True,
        return_inverse=True,
    )
    order = np.argsort(env_indexes, kind="stable")
    effective_missed_mask = np.zeros(len(notifications), dtype=bool)
    pos = 0
    missed_mask = missed_mask.astype(int)
    for size in env_counts:
        next_pos = pos + size
        indexes = order[pos:next_pos]
        pos = next_pos
        env_missed_mask = missed_mask[indexes]
        # we order notifications by finished_at descending
        # if there are no gaps and the missing indexes are strictly at the beginning,
        # compute only them; otherwise, compute everything
        if (np.diff(env_missed_mask) <= 0).all():
            effective_missed_mask[indexes] = env_missed_mask
        else:
            effective_missed_mask[indexes] = True
    if effective_missed_mask.all():
        return notifications
    return notifications.take(np.flatnonzero(effective_missed_mask))


@sentry_span
def deployment_facts_extract_mentioned_people(df: pd.DataFrame) -> np.ndarray:
    """Return all the people ever mentioned anywhere in the deployment facts df."""
    if df.empty:
        return np.array([], dtype=int)
    everybody = np.concatenate(
        [
            *df[DeploymentFacts.f.pr_authors].values,
            *df[DeploymentFacts.f.commit_authors].values,
            *(
                (rdf[Release.author_node_id.name].values if not rdf.empty else [])
                for rdf in df["releases"].values
            ),
        ],
    )
    return np.unique(everybody[is_not_null(everybody)].astype(int, copy=False))


@sentry_span
async def _finalize_release_settings(
    notifications: pd.DataFrame,
    time_from: datetime,
    time_to: datetime,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    branches: pd.DataFrame,
    default_branches: dict[str, str],
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> tuple[set[str], ReleaseSettings]:
    assert not notifications.empty
    deployed_repos_df = await read_sql_query(
        select(distinct(DeployedComponent.repository_node_id)).where(
            DeployedComponent.account_id == account,
            DeployedComponent.deployment_name.in_any_values(notifications.index.values),
        ),
        rdb,
        [DeployedComponent.repository_node_id],
    )
    repos = logical_settings.with_logical_deployments(
        prefixer.repo_node_to_name.get(r)
        for r in deployed_repos_df[DeployedComponent.repository_node_id.name].values
    )
    del deployed_repos_df
    need_disambiguate = []
    log = logging.getLogger(f"{metadata.__package__}._finalize_release_settings")
    for repo in repos:
        try:
            repo_setting = release_settings.native[repo]
        except KeyError:
            # not yet added to the reposet
            log.warning("missing from the global release settings: %s", repo)
            repo_setting = release_settings.native[repo] = release_settings.prefixed[
                prefixer.repo_name_to_prefixed_name[repo]
            ] = ReleaseMatchSetting.default()
        if repo_setting.match == ReleaseMatch.tag_or_branch:
            need_disambiguate.append(repo)
    if not need_disambiguate:
        return repos, release_settings
    _, matched_bys = await ReleaseLoader.load_releases(
        need_disambiguate,
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
        only_applied_matches=True,
    )
    return repos, ReleaseLoader.disambiguate_release_settings(release_settings, matched_bys)


@sentry_span
async def _fetch_precomputed_deployed_releases(
    notifications: pd.DataFrame,
    repo_names: Collection[str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    branches: pd.DataFrame,
    default_branches: dict[str, str],
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> pd.DataFrame:
    assert repo_names
    reverse_settings = reverse_release_settings(repo_names, default_branches, release_settings)
    batches = []
    queries = []
    many_notifications = len(notifications) > 100

    def append_batch(i: int) -> None:
        another_batch_size = len(queries)
        query = union_all(*queries)
        queries.clear()
        for j in range(i - another_batch_size + 1, i + 1):
            if many_notifications:
                query = (
                    query.with_statement_hint(f"Leading(((ghrd{j} *VALUES*) prel{j}))")
                    .with_statement_hint(f"Rows(ghrd{j} *VALUES* *{len(notifications)})")
                    .with_statement_hint(f"Rows(ghrd{j} *VALUES* prel{j} *{len(notifications)})")
                )
            else:
                query = query.with_statement_hint(f"Rows(ghrd{j} prel{j} *{len(notifications)})")
        batches.append(
            read_sql_query(
                query,
                pdb,
                [GitHubReleaseDeployment.deployment_name, *PrecomputedRelease.__table__.columns],
            ),
        )

    batch_size = 10
    for i, ((m, v), repos) in enumerate(reverse_settings.items(), start=1):
        ghrd = aliased(GitHubReleaseDeployment, name=f"ghrd{i}")
        prel = aliased(PrecomputedRelease, name=f"prel{i}")
        queries.append(
            select(ghrd.deployment_name, prel)
            .select_from(
                join(
                    ghrd,
                    prel,
                    and_(
                        ghrd.acc_id == prel.acc_id,
                        ghrd.release_id == prel.node_id,
                        ghrd.release_match == prel.release_match,
                        ghrd.repository_full_name == prel.repository_full_name,
                    ),
                ),
            )
            .where(
                ghrd.acc_id == account,
                ghrd.repository_full_name.in_(repos),
                prel.release_match == compose_release_match(m, v),
                ghrd.deployment_name.in_any_values(notifications.index.values)
                if many_notifications
                else ghrd.deployment_name.in_(notifications.index.values),
            ),
        )
        if len(queries) == batch_size:
            append_batch(i)
    if queries:
        append_batch(len(reverse_settings))
    if not batches:
        return pd.DataFrame()
    releases = await gather(*batches, op="_fetch_precomputed_deployed_releases/batches")
    if len(releases) == 1:
        releases = releases[0]
    else:
        releases = pd.concat(releases, ignore_index=True)
    releases = set_matched_by_from_release_match(releases, False)
    del releases[PrecomputedRelease.acc_id.name]
    return await _postprocess_deployed_releases(
        releases,
        branches,
        default_branches,
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


async def _postprocess_deployed_releases(
    releases: pd.DataFrame,
    branches: pd.DataFrame,
    default_branches: dict[str, str],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> pd.DataFrame:
    if releases.empty:
        return pd.DataFrame()
    assert isinstance(releases.index, pd.Int64Index)
    releases.sort_values(
        Release.published_at.name, ascending=False, ignore_index=True, inplace=True,
    )
    if not isinstance(releases[Release.published_at.name].dtype, pd.DatetimeTZDtype):
        releases[Release.published_at.name] = releases[Release.published_at.name].dt.tz_localize(
            timezone.utc,
        )
    user_node_to_login_get = prefixer.user_node_to_login.get
    releases[Release.author.name] = [
        user_node_to_login_get(u) for u in releases[Release.author_node_id.name].values
    ]
    release_facts_df = await mine_releases_by_ids(
        releases,
        branches,
        default_branches,
        release_settings,
        logical_settings,
        prefixer,
        account,
        meta_ids,
        mdb,
        pdb,
        rdb,
        cache,
        with_avatars=False,
        with_extended_pr_details=False,
        with_jira=JIRAEntityToFetch.NOTHING,
    )
    releases.set_index([Release.node_id.name, Release.repository_full_name.name], inplace=True)
    releases_node_ids = releases.index.get_level_values(0).unique()
    computed_node_ids = (
        release_facts_df[ReleaseFacts.f.node_id].unique()
        if not release_facts_df.empty
        else np.array([], dtype=int)
    )
    if len(diff := np.setdiff1d(releases_node_ids, computed_node_ids, assume_unique=True)):
        log = logging.getLogger(f"{metadata.__package__}.mine_releases_by_ids/deployed")
        log.warning("failed to compute release facts of %s", diff.tolist())
        # this can happen when somebody removes a release while we are here
        # also, on deleted releases
    if len(release_facts_df) == 0:
        return pd.DataFrame()
    release_facts_df.set_index(
        [ReleaseFacts.f.node_id, ReleaseFacts.f.repository_full_name], inplace=True,
    )
    assert release_facts_df.index.is_unique
    for col in (
        ReleaseFacts.f.publisher,
        ReleaseFacts.f.published,
        ReleaseFacts.f.matched_by,
        ReleaseFacts.f.sha,
        ReleaseFacts.f.name,
        ReleaseFacts.f.url,
    ):
        del release_facts_df[col]
    releases = release_facts_df.join(releases)
    return releases


def _group_releases(df: pd.DataFrame) -> pd.DataFrame:
    groups = list(df.groupby("deployment_name", sort=False))
    grouped_releases = pd.DataFrame(
        {
            "deployment_name": [g[0] for g in groups],
            "releases": [g[1] for g in groups],
        },
    )
    for df in grouped_releases["releases"].values:
        del df["deployment_name"]
    grouped_releases.set_index("deployment_name", drop=True, inplace=True)
    return grouped_releases


def _group_components(df: pd.DataFrame) -> pd.DataFrame:
    groups = list(df.groupby(DeployedComponent.deployment_name.name, sort=False))
    grouped_components = pd.DataFrame(
        {
            "deployment_name": [g[0] for g in groups],
            "components": [g[1] for g in groups],
        },
    )
    for df in grouped_components["components"].values:
        df.reset_index(drop=True, inplace=True)
    grouped_components.set_index("deployment_name", drop=True, inplace=True)
    return grouped_components


@sentry_span
async def _filter_by_participants(
    df: pd.DataFrame,
    participants: ReleaseParticipants,
) -> pd.DataFrame:
    if df.empty or not participants:
        return df
    mask = np.zeros(len(df), dtype=bool)
    for pkind, col in zip(
        ReleaseParticipationKind,
        [
            DeploymentFacts.f.pr_authors,
            DeploymentFacts.f.commit_authors,
            DeploymentFacts.f.release_authors,
        ],
    ):
        if pkind not in participants:
            continue
        people = np.array(participants[pkind])
        values = df[col].values
        offsets = np.zeros(len(values) + 1, dtype=int)
        lengths = nested_lengths(values)
        np.cumsum(lengths, out=offsets[1:])
        values = np.concatenate([np.concatenate(values), [-1]])
        passing = np.bitwise_or.reduceat(np.in1d(values, people), offsets)[:-1]
        passing[lengths == 0] = False
        mask[passing] = True
    df.disable_consolidate()
    return df.take(np.flatnonzero(mask))


@sentry_span
async def _filter_by_repositories(
    df: pd.DataFrame,
    repos: set[str] | frozenset[str] | KeysView[str],
) -> pd.DataFrame:
    if df.empty or not repos:
        return df
    df_repos = np.concatenate(df[DeploymentFacts.f.repositories].values)
    df_deployed_commits = np.concatenate(df[DeploymentFacts.f.commits_overall].values)
    df_repos[df_deployed_commits == 0] = ""
    passed = np.in1d(df_repos, list(repos))
    lengths = nested_lengths(df[DeploymentFacts.f.repositories].values)
    offsets = np.empty(len(lengths), dtype=int)
    np.cumsum(lengths[:-1], out=offsets[1:])
    offsets[0] = 0
    indexes = np.flatnonzero(np.logical_or.reduceat(passed, offsets))
    if len(indexes) < len(df):
        return df.take(indexes)
    return df


@sentry_span
async def _filter_by_prs(
    df: pd.DataFrame,
    labels: LabelFilter,
    jira: JIRAFilter,
    meta_ids: tuple[int, ...],
    mdb: Database,
    cache: Optional[aiomcache.Client],
) -> pd.DataFrame:
    pr_node_ids = np.concatenate(df[DeploymentFacts.f.prs].values, dtype=int, casting="unsafe")
    unique_pr_node_ids = np.unique(pr_node_ids)
    lengths = np.empty(len(df) + 1, dtype=int)
    lengths[-1] = 0
    nested_lengths(df[DeploymentFacts.f.prs].values, lengths)
    offsets = np.zeros(len(lengths), dtype=int)
    np.cumsum(lengths[:-1], out=offsets[1:])
    filters = [
        PullRequest.acc_id.in_(meta_ids),
        PullRequest.node_id.in_any_values(unique_pr_node_ids),
    ]
    tasks = []
    if labels:
        singles, multiples = LabelFilter.split(labels.include)
        if not (embedded_labels_query := not multiples):
            label_columns = [
                PullRequestLabel.pull_request_node_id,
                func.lower(PullRequestLabel.name),
            ]
            tasks.append(
                read_sql_query(
                    select(label_columns).where(
                        PullRequestLabel.acc_id.in_(meta_ids),
                        PullRequestLabel.pull_request_node_id.in_any_values(unique_pr_node_ids),
                    ),
                    mdb,
                    label_columns,
                    index=PullRequestLabel.pull_request_node_id.name,
                ),
            )
        if all_in_labels := (set(singles + list(chain.from_iterable(multiples)))):
            filters.append(
                exists().where(
                    PullRequestLabel.acc_id == PullRequest.acc_id,
                    PullRequestLabel.pull_request_node_id == PullRequest.node_id,
                    func.lower(PullRequestLabel.name).in_(all_in_labels),
                ),
            )
        if labels.exclude:
            filters.append(
                not_(
                    exists().where(
                        PullRequestLabel.acc_id == PullRequest.acc_id,
                        PullRequestLabel.pull_request_node_id == PullRequest.node_id,
                        func.lower(PullRequestLabel.name).in_(labels.exclude),
                    ),
                ),
            )
    if jira:
        query = await generate_jira_prs_query(
            filters, jira, meta_ids, mdb, cache, columns=[PullRequest.node_id],
        )
    else:
        query = select(PullRequest.node_id).where(*filters)
    query = query.with_statement_hint(f"Rows(pr repo #{len(unique_pr_node_ids)})")
    tasks.insert(0, read_sql_query(query, mdb, [PullRequest.node_id]))
    prs_df, *label_df = await gather(*tasks, op="_filter_by_prs/sql")
    prs = prs_df[PullRequest.node_id.name].values
    if labels and not embedded_labels_query:
        label_df = label_df[0]
        left = find_left_prs_by_labels(
            pd.Index(prs),  # there are `multiples` so we don't care
            label_df.index,
            label_df[PullRequestLabel.name.name].values,
            labels,
        )
        prs = left.values
    passed = np.in1d(
        np.concatenate([pr_node_ids, [0]]),
        prs,
        assume_unique=len(pr_node_ids) == len(unique_pr_node_ids),
    )
    if labels.include or jira:
        passed = np.bitwise_or.reduceat(passed, offsets)
    else:
        passed = np.bitwise_and.reduceat(passed, offsets)
    passed[lengths == 0] = False
    return df.take(np.flatnonzero(passed))


@sentry_span
async def _compute_deployment_facts(
    notifications: pd.DataFrame,
    components: pd.DataFrame,
    with_extended_prs: bool,
    with_jira: bool,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    branches: pd.DataFrame,
    default_branches: dict[str, str],
    prefixer: Prefixer,
    account: int,
    jira_ids: Optional[JIRAConfig],
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    components = components.take(
        np.flatnonzero(
            np.in1d(components.index.values.astype("U"), notifications.index.values.astype("U")),
        ),
    )
    (commit_relationship, dags, deployed_commits_df) = await _resolve_commit_relationship(
        notifications,
        components,
        logical_settings,
        prefixer,
        account,
        meta_ids,
        mdb,
        pdb,
        rdb,
        cache,
    )
    if notifications.empty:
        return (pd.DataFrame(),) * 4
    with sentry_sdk.start_span(op=f"_extract_deployed_commits({len(components)})"):
        deployed_commits_per_repo_per_env, all_mentioned_hashes = await _extract_deployed_commits(
            notifications, components, deployed_commits_df, commit_relationship, dags,
        )
    await defer(
        _submit_deployed_commits(deployed_commits_per_repo_per_env, account, meta_ids, mdb, pdb),
        "_submit_deployed_commits",
    )
    max_release_time_to = notifications[DeploymentNotification.finished_at.name].max()
    (commit_stats, *jira_map_tasks), releases = await gather(
        _fetch_commit_stats(
            all_mentioned_hashes,
            components[DeployedComponent.repository_node_id.name].unique(),
            dags,
            with_extended_prs,
            with_jira,
            prefixer,
            logical_settings,
            account,
            jira_ids,
            meta_ids,
            mdb,
            pdb,
            rdb,
        ),
        _map_releases_to_deployments(
            deployed_commits_per_repo_per_env,
            all_mentioned_hashes,
            max_release_time_to,
            prefixer,
            release_settings,
            logical_settings,
            branches,
            default_branches,
            account,
            meta_ids,
            mdb,
            pdb,
            rdb,
            cache,
        ),
    )
    (facts, extended_prs), *jira_maps = await gather(
        _generate_deployment_facts(
            notifications,
            deployed_commits_per_repo_per_env,
            all_mentioned_hashes,
            commit_stats,
            releases,
            account,
            pdb,
        ),
        *jira_map_tasks,
    )
    if with_jira and jira_ids is not None:
        if jira_maps[1] is not None:
            df_jira_map = pd.concat(jira_maps, ignore_index=True)
        else:
            df_jira_map = jira_maps[0]
        jira = _apply_jira_to_deployment_facts(facts, df_jira_map)
    else:
        jira = pd.DataFrame()
    await defer(
        _submit_deployment_facts(
            facts, components, default_branches, release_settings, account, pdb,
        ),
        "_submit_deployment_facts",
    )
    return facts, releases, extended_prs, jira


def _adjust_empty_df(joined: pd.DataFrame, name: str) -> pd.DataFrame:
    try:
        no_df = joined[name].isnull().values  # can be NaN-s
    except KeyError:
        no_df = np.ones(len(joined), bool)
    col = np.full(no_df.sum(), None, object)
    col.fill(pd.DataFrame())
    joined.loc[no_df, name] = col
    return joined


async def _submit_deployment_facts(
    facts: pd.DataFrame,
    components: pd.DataFrame,
    default_branches: dict[str, str],
    settings: ReleaseSettings,
    account: int,
    pdb: Database,
) -> None:
    joined = _adjust_empty_df(facts.join(components), "components")
    values = [
        GitHubDeploymentFacts(
            acc_id=account,
            deployment_name=name,
            release_matches=json.dumps(
                {
                    r: settings.native[r].as_db(default_branches[drop_logical_repo(r)])
                    for r in components[DeployedComponent.repository_full_name].values
                },
            )
            if not components.empty
            else "{}",
            data=DeploymentFacts.from_fields(
                pr_authors=pr_authors,
                commit_authors=commit_authors,
                release_authors=release_authors,
                repositories=repos,
                lines_prs=lines_prs,
                lines_overall=lines_overall,
                commits_prs=commits_prs,
                commits_overall=commits_overall,
                prs=prs,
                prs_offsets=prs_offsets,
            ).data,
        )
        .create_defaults()
        .explode(with_primary_keys=True)
        for name, pr_authors, commit_authors, release_authors, repos, lines_prs, lines_overall, commits_prs, commits_overall, prs, prs_offsets, components in zip(  # noqa
            joined.index.values,
            joined[DeploymentFacts.f.pr_authors].values,
            joined[DeploymentFacts.f.commit_authors].values,
            joined[DeploymentFacts.f.release_authors].values,
            joined[DeploymentFacts.f.repositories].values,
            joined[DeploymentFacts.f.lines_prs].values,
            joined[DeploymentFacts.f.lines_overall].values,
            joined[DeploymentFacts.f.commits_prs].values,
            joined[DeploymentFacts.f.commits_overall].values,
            joined[DeploymentFacts.f.prs].values,
            joined[DeploymentFacts.f.prs_offsets].values,
            joined["components"].values,
        )
    ]
    await insert_or_ignore(GitHubDeploymentFacts, values, "_submit_deployment_facts", pdb)


class _CommitRelationship(NamedTuple):
    parent_node_ids: npt.NDArray[int]
    parent_shas: npt.NDArray[bytes]  # S40
    # indicates whether we *don't* deduplicate the corresponding parent
    externals: npt.NDArray[bool]


class _DeployedCommitDetails(NamedTuple):
    shas: npt.NDArray[bytes]
    deployments: npt.NDArray[str]


class _DeployedCommitStats(NamedTuple):
    commit_authors: npt.NDArray[int]
    lines: npt.NDArray[int]
    pull_requests: npt.NDArray[int]
    merge_shas: npt.NDArray[bytes]
    pr_lines: npt.NDArray[int]
    pr_authors: npt.NDArray[int]
    pr_commit_counts: npt.NDArray[int]
    pr_repository_full_names: npt.NDArray[str]
    ambiguous_prs: dict[str, list[int]]
    already_deployed_rebased_by_env_by_repo: dict[str, dict[str, npt.NDArray[int]]]
    extended_prs: dict[str, np.ndarray]


class _RepositoryDeploymentFacts(NamedTuple):
    pr_authors: npt.NDArray[int]
    commit_authors: npt.NDArray[int]
    release_authors: set[int]
    prs: npt.NDArray[int]
    lines_prs: int
    lines_overall: int
    commits_prs: int
    commits_overall: int
    extended_prs: dict[str, np.ndarray]


async def _generate_deployment_facts(
    notifications: pd.DataFrame,
    deployed_commits_per_repo_per_env: dict[str, dict[str, _DeployedCommitDetails]],
    all_mentioned_hashes: np.ndarray,
    commit_stats: _DeployedCommitStats,
    releases: pd.DataFrame,
    account: int,
    pdb: Database,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    name_to_finished = dict(
        zip(
            notifications.index.values,
            notifications[DeploymentNotification.finished_at.name].values,
        ),
    )
    pr_inserts = {}
    all_releases_authors = defaultdict(set)
    if not releases.empty:
        try:
            unique_release_deps, release_index, release_group_counts = np.unique(
                releases[GitHubReleaseDeployment.deployment_name.name].values,
                return_inverse=True,
                return_counts=True,
            )
        except TypeError as e:
            log = logging.getLogger(f"{metadata.__package__}._generate_deployment_facts")
            for i, v in enumerate(releases[GitHubReleaseDeployment.deployment_name.name].values):
                if not isinstance(v, str):
                    log.error(
                        "non-string release name: %s %s",
                        v,
                        releases[GitHubReleaseDeployment.deployment_name.name].iloc[i],
                    )
            raise e from None
        release_order = np.argsort(release_index)
        release_pos = 0
        for name, group_size in zip(unique_release_deps, release_group_counts):
            all_releases_authors[name].update(
                releases[Release.author_node_id.name].values[
                    release_order[release_pos : release_pos + group_size]
                ],
            )
            release_pos += group_size
    facts_per_repo_per_deployment = defaultdict(dict)
    repo_order = np.argsort(commit_stats.pr_repository_full_names)
    unique_repos, repo_commit_counts = np.unique(
        commit_stats.pr_repository_full_names[repo_order], return_counts=True,
    )
    repo_order_offsets = np.cumsum(repo_commit_counts)
    for env, repos in deployed_commits_per_repo_per_env.items():
        repo_indexes = np.searchsorted(unique_repos, list(repos))
        for repo_index, (repo_name, details) in zip(repo_indexes, repos.items()):
            has_merged_commits = len(commit_stats.merge_shas) and repo_index < len(unique_repos)
            try:
                # fmt: off
                already_deployed_rebased = (
                    commit_stats.already_deployed_rebased_by_env_by_repo[env][repo_name]
                )
                # fmt: on
            except KeyError:
                already_deployed_rebased = np.array([], dtype=int)
            if has_merged_commits:
                matched_repo_offset_end = repo_order_offsets[repo_index]
                matched_repo_offset_beg = matched_repo_offset_end - repo_commit_counts[repo_index]
                # sort to restore the original order by commit sha
                matched_repo_indexes = np.sort(
                    repo_order[matched_repo_offset_beg:matched_repo_offset_end],
                )
            ambiguous_prs = np.array(commit_stats.ambiguous_prs[repo_name])
            has_ambiguous_prs = bool(len(ambiguous_prs))
            ambiguous_pr_deployments = defaultdict(list)

            def calc_deployment_facts(
                deployment_name: str,
                deployed_shas: np.ndarray,
                pr_blocklist: Sequence[int],
            ) -> None:
                finished = pd.Timestamp(name_to_finished[deployment_name], tzinfo=timezone.utc)
                indexes = np.searchsorted(all_mentioned_hashes, deployed_shas)
                if has_merged_commits:
                    pr_indexes = searchsorted_inrange(
                        commit_stats.merge_shas[matched_repo_indexes], deployed_shas,
                    )
                    pr_indexes = matched_repo_indexes[
                        pr_indexes[
                            commit_stats.merge_shas[matched_repo_indexes[pr_indexes]]
                            == deployed_shas
                        ]
                    ]
                else:
                    pr_indexes = []
                if len(pr_blocklist):
                    passed_mask = np.ones(len(pr_indexes), dtype=bool)
                    passed_mask[pr_blocklist] = False
                    pr_indexes = pr_indexes[passed_mask]
                prs = commit_stats.pull_requests[pr_indexes]
                if len(already_deployed_rebased):
                    prs_fresh_mask = np.in1d(prs, already_deployed_rebased, invert=True)
                    prs = prs[prs_fresh_mask]
                    pr_indexes = pr_indexes[prs_fresh_mask]
                if has_ambiguous_prs:
                    for i in np.flatnonzero(np.in1d(prs, ambiguous_prs, assume_unique=True)):
                        ambiguous_pr_deployments[prs[i]].append(
                            (finished, deployment_name, deployed_shas, i),
                        )
                deployed_lines = commit_stats.lines[indexes]
                pr_deployed_lines = commit_stats.pr_lines[pr_indexes]
                commit_authors = np.unique(commit_stats.commit_authors[indexes])
                pr_authors = commit_stats.pr_authors[pr_indexes]
                if extended_prs := {
                    k: v[pr_indexes] for k, v in commit_stats.extended_prs.items()
                }:
                    extended_prs[PullRequest.user_node_id.name] = pr_authors
                facts_per_repo_per_deployment[deployment_name][
                    repo_name
                ] = _RepositoryDeploymentFacts(
                    pr_authors=pr_authors[pr_authors != 0],
                    commit_authors=commit_authors[commit_authors != 0],
                    release_authors=all_releases_authors.get(deployment_name, []),
                    lines_prs=pr_deployed_lines.sum(),
                    lines_overall=deployed_lines.sum(),
                    commits_prs=commit_stats.pr_commit_counts[pr_indexes].sum(),
                    commits_overall=len(indexes),
                    prs=prs,
                    extended_prs=extended_prs,
                )
                pr_inserts[(deployment_name, repo_name)] = [
                    (deployment_name, finished, repo_name, pr) for pr in prs
                ]

            for deployment_name, deployed_shas in zip(details.deployments, details.shas):
                calc_deployment_facts(deployment_name, deployed_shas, [])

            if ambiguous_pr_deployments:
                has_ambiguous_prs = False  # don't write them twice
                # erase everything after the oldest deployment
                removed_by_deployment = defaultdict(list)
                deployed_shas_by_name = {}
                for addrs in ambiguous_pr_deployments.values():
                    if len(addrs) == 1:
                        continue
                    addrs.sort()
                    for _, deployment_name, deployed_shas, i in addrs[1:]:
                        deployed_shas_by_name[deployment_name] = deployed_shas
                        removed_by_deployment[deployment_name].append(i)
                for deployment_name, removed in removed_by_deployment.items():
                    calc_deployment_facts(
                        deployment_name, deployed_shas_by_name[deployment_name], removed,
                    )
    pr_inserts = list(chain.from_iterable(pr_inserts.values()))
    facts = []
    extended_prs = {
        k.name: []
        for k in (
            PullRequest.number,
            PullRequest.title,
            PullRequest.created_at,
            PullRequest.additions,
            PullRequest.deletions,
            PullRequest.user_node_id,
        )
    }
    extended_prs["deployment_name"] = []
    for deployment_name, repos in facts_per_repo_per_deployment.items():
        repo_index = []
        pr_authors = []
        commit_authors = []
        release_authors = set()
        lines_prs = []
        lines_overall = []
        commits_prs = []
        commits_overall = []
        prs = []
        local_extended_prs = {k: [] for k in extended_prs}
        for repo, rf in sorted(repos.items()):
            repo_index.append(repo)
            pr_authors.append(rf.pr_authors)
            commit_authors.append(rf.commit_authors)
            release_authors.update(rf.release_authors)
            lines_prs.append(rf.lines_prs)
            lines_overall.append(rf.lines_overall)
            commits_prs.append(rf.commits_prs)
            commits_overall.append(rf.commits_overall)
            prs.append(rf.prs)
            for k, v in rf.extended_prs.items():
                local_extended_prs[k].append(v)
        if len(local_extended_prs[PullRequest.number.name]):
            for k, v in local_extended_prs.items():
                if k != "deployment_name":
                    extended_prs[k].append(np.concatenate(v))
                else:
                    extended_prs[k].append(deployment_name)
        prs_offsets = np.cumsum(nested_lengths(prs), dtype=np.uint32)[:-1]
        pr_authors = np.unique(np.concatenate(pr_authors))
        commit_authors = np.unique(np.concatenate(commit_authors))
        release_authors = np.array(list(release_authors - {None}), dtype=int)
        facts.append(
            DeploymentFacts.from_fields(
                pr_authors=pr_authors,
                commit_authors=commit_authors,
                release_authors=release_authors,
                repositories=repo_index,
                lines_prs=lines_prs,
                lines_overall=lines_overall,
                commits_prs=commits_prs,
                commits_overall=commits_overall,
                prs=np.concatenate(prs) if prs else [],
                prs_offsets=prs_offsets,
                name=deployment_name,
            ),
        )
    await defer(_submit_deployed_prs(pr_inserts, account, pdb), "_submit_deployed_prs")
    facts = df_from_structs(facts)
    facts.index = facts[DeploymentNotification.name.name].values
    extended_prs = pd.DataFrame(extended_prs)
    extended_prs.set_index("deployment_name", inplace=True)
    return facts, extended_prs


@sentry_span
async def _map_releases_to_deployments(
    deployed_commits_per_repo_per_env: dict[str, dict[str, _DeployedCommitDetails]],
    all_mentioned_hashes: np.ndarray,
    max_release_time_to: datetime,
    prefixer: Prefixer,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    branches: pd.DataFrame,
    default_branches: dict[str, str],
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> pd.DataFrame:
    repo_names = {r for repos in deployed_commits_per_repo_per_env.values() for r in repos}
    reverse_settings = reverse_release_settings(repo_names, default_branches, release_settings)
    reverse_settings_ptr = {}
    for key, repos in reverse_settings.items():
        for repo in repos:
            reverse_settings_ptr[repo] = key
    commits_by_reverse_key = {}
    all_commit_sha_repos = []
    all_deployment_names = []
    for repos in deployed_commits_per_repo_per_env.values():
        for repo, details in repos.items():
            all_repo_shas = np.concatenate(details.shas)
            commits_by_reverse_key.setdefault(reverse_settings_ptr[repo], []).append(all_repo_shas)
            all_commit_sha_repos.append(np.char.add(all_repo_shas, repo.encode()))
            all_deployment_names.append(
                np.repeat(details.deployments, [len(shas) for shas in details.shas]),
            )
    all_commit_sha_repos = np.concatenate(all_commit_sha_repos)
    all_deployment_names = np.concatenate(all_deployment_names)
    order = np.argsort(all_commit_sha_repos)
    all_commit_sha_repos = all_commit_sha_repos[order]
    all_deployment_names = all_deployment_names[order]
    assert all_commit_sha_repos.shape == all_deployment_names.shape
    _, unique_commit_sha_counts = np.unique(all_commit_sha_repos, return_counts=True)
    offsets = np.zeros(len(unique_commit_sha_counts) + 1, dtype=int)
    np.cumsum(unique_commit_sha_counts, out=offsets[1:])
    event_hashes = []
    for key, val in commits_by_reverse_key.items():
        if key[0] == ReleaseMatch.event:
            event_hashes.extend(val)
        else:
            commits_by_reverse_key[key] = np.unique(np.concatenate(val))
    if event_hashes:
        event_hashes = np.concatenate(event_hashes)
    commits_by_reverse_key = {
        k: v for k, v in commits_by_reverse_key.items() if k[0] != ReleaseMatch.event
    }

    async def dummy():
        df = dummy_releases_df()
        df[PrecomputedRelease.release_match.name] = []
        return df

    releases, event_releases = await gather(
        read_sql_query(
            union_all(
                *(
                    select(PrecomputedRelease).where(
                        PrecomputedRelease.acc_id == account,
                        PrecomputedRelease.release_match
                        == compose_release_match(match_id, match_value),
                        PrecomputedRelease.published_at < max_release_time_to,
                        PrecomputedRelease.sha.in_any_values(commit_shas),
                    )
                    for (match_id, match_value), commit_shas in commits_by_reverse_key.items()
                ),
            ),
            pdb,
            PrecomputedRelease,
        )
        if commits_by_reverse_key
        else dummy(),
        read_sql_query(
            select(ReleaseNotification).where(
                ReleaseNotification.account_id == account,
                ReleaseNotification.published_at < max_release_time_to,
                ReleaseNotification.resolved_commit_hash.in_any_values(event_hashes),
            ),
            rdb,
            ReleaseNotification,
        ),
    )
    if not event_releases.empty:
        event_releases.disable_consolidate()
        for rename_from, rename_to in [
            (ReleaseNotification.account_id, PrecomputedRelease.acc_id),
            (ReleaseNotification.resolved_commit_hash, PrecomputedRelease.sha),
            (ReleaseNotification.resolved_commit_node_id, PrecomputedRelease.commit_id),
            (ReleaseNotification.cloned, None),
            (ReleaseNotification.commit_hash_prefix, None),
            (ReleaseNotification.created_at, None),
            (ReleaseNotification.updated_at, None),
        ]:
            if rename_to is not None:
                event_releases[rename_to.name] = event_releases[rename_from.name]
            del event_releases[rename_from.name]
        repo_node_to_name = prefixer.repo_node_to_name.get
        event_releases[PrecomputedRelease.repository_full_name.name] = [
            repo_node_to_name(r)
            for r in event_releases[ReleaseNotification.repository_node_id.name].values
        ]
        event_releases[PrecomputedRelease.release_match.name] = ReleaseMatch.event.name
        event_releases[PrecomputedRelease.tag.name] = None
        event_releases[PrecomputedRelease.node_id.name] = event_releases[
            PrecomputedRelease.commit_id.name
        ] = event_releases[PrecomputedRelease.commit_id.name].astype(int)
        if releases.empty:
            releases = event_releases
        else:
            releases.disable_consolidate()
            releases = pd.concat([releases, event_releases], ignore_index=True, copy=False)
    releases = set_matched_by_from_release_match(releases, remove_ambiguous_tag_or_branch=False)
    if releases.empty:
        time_from = await mdb.fetch_val(
            select(func.min(NodeCommit.committed_date)).where(
                NodeCommit.acc_id.in_(meta_ids),
                NodeCommit.sha.in_any_values(all_mentioned_hashes),
            ),
        )
        if time_from is None:
            time_from = max_release_time_to - timedelta(days=10 * 365)
        elif mdb.url.dialect == "sqlite":
            time_from = time_from.replace(tzinfo=timezone.utc)
    else:
        time_from = releases[Release.published_at.name].max() + timedelta(seconds=1)
    if time_from < max_release_time_to:
        extra_releases, _ = await ReleaseLoader.load_releases(
            repo_names,
            branches,
            default_branches,
            time_from,
            max_release_time_to,
            release_settings,
            logical_settings,
            prefixer,
            account,
            meta_ids,
            mdb,
            pdb,
            rdb,
            cache,
            force_fresh=max_release_time_to > datetime.now(timezone.utc) - unfresh_releases_lag,
        )
        if not extra_releases.empty:
            releases = pd.concat([releases, extra_releases], copy=False, ignore_index=True)

    if releases.empty:
        release_commit_shas = np.ndarray([], dtype="S")
    else:
        release_commit_shas = np.char.add(
            releases[Release.sha.name].values,
            releases[Release.repository_full_name.name].values.astype("S"),
        )
    positions = searchsorted_inrange(all_commit_sha_repos, release_commit_shas)
    if len(all_commit_sha_repos):
        positions[all_commit_sha_repos[positions] != release_commit_shas] = len(
            all_commit_sha_repos,
        )
    else:
        positions[:] = len(all_commit_sha_repos)
    unique_commit_sha_counts = np.concatenate([unique_commit_sha_counts, [0]])
    lengths = unique_commit_sha_counts[np.searchsorted(offsets, positions)]
    deployment_names = all_deployment_names[
        np.repeat(positions + lengths - lengths.cumsum(), lengths) + np.arange(lengths.sum())
    ]
    col_vals = {
        c: np.repeat(releases[c].values, lengths)
        for c in releases.columns
        if c != PrecomputedRelease.acc_id.name
    }
    releases = pd.DataFrame({"deployment_name": deployment_names, **col_vals})
    result = await _postprocess_deployed_releases(
        releases,
        branches,
        default_branches,
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
    await defer(
        _submit_deployed_releases(releases, account, default_branches, release_settings, pdb),
        "_submit_deployed_releases",
    )
    return result


@sentry_span
async def _submit_deployed_commits(
    deployed_commits_per_repo_per_env: dict[str, dict[str, _DeployedCommitDetails]],
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
) -> None:
    all_shas = []
    for repos in deployed_commits_per_repo_per_env.values():
        for details in repos.values():
            all_shas.extend(details.shas)
    all_shas = unordered_unique(np.concatenate(all_shas))
    sha_id_df = await read_sql_query(
        select(NodeCommit.graph_id, NodeCommit.sha)
        .where(NodeCommit.acc_id.in_(meta_ids), NodeCommit.sha.in_any_values(all_shas))
        .order_by(NodeCommit.sha),
        mdb,
        [NodeCommit.graph_id, NodeCommit.sha],
    )
    sha_map_shas = sha_id_df[NodeCommit.sha.name].values
    sha_map_ids = sha_id_df[NodeCommit.id.name].values
    values = [
        GitHubCommitDeployment(
            acc_id=account,
            deployment_name=deployment_name,
            commit_id=commit_id,
            repository_full_name=repo,
        ).explode(with_primary_keys=True)
        for repos in deployed_commits_per_repo_per_env.values()
        for repo, details in repos.items()
        for deployment_name, shas in zip(details.deployments, details.shas)
        for sha, commit_id in zip(shas, sha_map_ids[np.searchsorted(sha_map_shas, shas)])
    ]
    await insert_or_ignore(GitHubCommitDeployment, values, "_submit_deployed_commits", pdb)


@sentry_span
async def _submit_deployed_prs(
    values: list[tuple[str, datetime, str, int]],
    account: int,
    pdb: Database,
) -> None:
    values = [
        GitHubPullRequestDeployment(
            acc_id=account,
            deployment_name=deployment_name,
            finished_at=finished_at,
            pull_request_id=pr,
            repository_full_name=repo,
        ).explode(with_primary_keys=True)
        for (deployment_name, finished_at, repo, pr) in values
    ]
    await insert_or_ignore(GitHubPullRequestDeployment, values, "_submit_deployed_prs", pdb)


@sentry_span
async def _submit_deployed_releases(
    releases: pd.DataFrame,
    account: int,
    default_branches: dict[str, str],
    settings: ReleaseSettings,
    pdb: Database,
) -> None:
    if releases.empty:
        return
    values = [
        GitHubReleaseDeployment(
            acc_id=account,
            deployment_name=deployment_name,
            repository_full_name=repo,
            release_id=release_id,
            release_match=settings.native[repo].as_db(default_branches[drop_logical_repo(repo)]),
        ).explode(with_primary_keys=True)
        for deployment_name, release_id, repo in zip(
            releases["deployment_name"].values,
            releases.index.get_level_values(0).values,
            releases.index.get_level_values(1).values,
        )
    ]
    await insert_or_ignore(GitHubReleaseDeployment, values, "_submit_deployed_releases", pdb)


@sentry_span
async def _fetch_commit_stats(
    all_mentioned_hashes: np.ndarray,
    repo_ids: Collection[int],
    dags: dict[str, tuple[bool, DAG]],
    with_extended_prs: bool,
    with_jira: bool,
    prefixer: Prefixer,
    logical_settings: LogicalRepositorySettings,
    account: int,
    jira_ids: Optional[JIRAConfig],
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
) -> tuple[_DeployedCommitStats, Optional[asyncio.Task], Optional[asyncio.Task]]:
    has_logical = logical_settings.has_logical_deployments()
    pull_request_id_col = NodePullRequest.id.label("pull_request_id")
    pull_request_id_col.info = {"reset_nulls": True}
    selected = [
        NodeCommit.id,
        NodeCommit.sha,
        pull_request_id_col,
        NodeCommit.author_user_id,
        NodeRepository.name_with_owner.label(PullRequest.repository_full_name.name),
        (NodeCommit.additions + NodeCommit.deletions).label("lines"),
        NodePullRequest.author_id.label("pr_author"),
    ]
    if with_extended_prs:
        extended_int_cols = [
            NodePullRequest.additions,
            NodePullRequest.deletions,
            NodePullRequest.number,
        ]
        for c in extended_int_cols:
            c.info = {"reset_nulls": True}
        selected.extend(
            [
                NodePullRequest.title,
                NodePullRequest.created_at,
                *extended_int_cols,
            ],
        )
    else:
        pr_lines_col = (NodePullRequest.additions + NodePullRequest.deletions).label("pr_lines")
        pr_lines_col.info = {"reset_nulls": True}
        selected.append(pr_lines_col)
        if has_logical:
            selected.append(NodePullRequest.title)
    commits_df, rebased_prs = await gather(
        read_sql_query(
            select(selected)
            .select_from(
                join(
                    join(
                        NodeCommit,
                        NodePullRequest,
                        and_(
                            NodeCommit.acc_id == NodePullRequest.acc_id,
                            NodeCommit.id == NodePullRequest.merge_commit_id,
                        ),
                        isouter=True,
                    ),
                    NodeRepository,
                    and_(
                        NodeCommit.acc_id == NodeRepository.acc_id,
                        NodeCommit.repository_id == NodeRepository.id,
                    ),
                    isouter=True,
                ),
            )
            .where(
                NodeCommit.acc_id.in_(meta_ids),
                NodeCommit.sha.in_any_values(all_mentioned_hashes),
            )
            .order_by(func.coalesce(NodePullRequest.merged_at, NodeCommit.committed_date)),
            mdb,
            selected,
        ),
        match_rebased_prs(repo_ids, account, meta_ids, mdb, pdb, commit_shas=all_mentioned_hashes),
    )
    if not rebased_prs.empty:
        rebased_map = dict(
            zip(
                rebased_prs[GitHubRebasedPullRequest.pr_node_id.name].values,
                rebased_prs[GitHubRebasedPullRequest.matched_merge_commit_sha.name].values,
            ),
        )
    else:
        rebased_map = {}

    seen_commit_shas: dict[str, int] = {}
    shas = commits_df[NodeCommit.sha.name].values
    lines = commits_df["lines"].values
    commit_authors = commits_df[NodeCommit.author_user_id.name].values
    has_pr_mask = commits_df[pull_request_id_col.name].values != 0
    pure_commits_count = len(commits_df) - has_pr_mask.sum()
    merge_shas = shas[has_pr_mask]
    pr_ids = commits_df[pull_request_id_col.name].values[has_pr_mask]
    pr_authors = commits_df["pr_author"].values[has_pr_mask]
    pr_repo_names = commits_df[PullRequest.repository_full_name.name].values[has_pr_mask]
    if with_extended_prs:
        pr_numbers = commits_df[NodePullRequest.number.name].values[has_pr_mask]
        pr_createds = commits_df[NodePullRequest.created_at.name].values[has_pr_mask]
        pr_additions = commits_df[NodePullRequest.additions.name].values[has_pr_mask]
        pr_deletions = commits_df[NodePullRequest.deletions.name].values[has_pr_mask]
        pr_lines = pr_additions + pr_deletions
        pr_titles = []
    else:
        pr_lines = commits_df[pr_lines_col.name].values[has_pr_mask]
        if has_logical:
            pr_titles = commits_df[NodePullRequest.title.name].values[has_pr_mask]
    prs_by_repo = defaultdict(list)
    ambiguous_prs = defaultdict(list)
    for pr, repo, sha, title, number in zip(
        pr_ids,
        pr_repo_names,
        merge_shas,
        commits_df[NodePullRequest.title.name].values[has_pr_mask]
        if with_extended_prs
        else repeat(None),
        commits_df[NodePullRequest.number.name].values[has_pr_mask]
        if with_extended_prs
        else repeat(None),
    ):
        if pr in rebased_map:
            ambiguous_prs[repo].append(pr)
        origin = seen_commit_shas.get(sha)
        if with_extended_prs:
            if origin:
                pr_titles.append(f"{title} [duplicate of #{origin}]")
            else:
                seen_commit_shas[sha] = number
                pr_titles.append(title)
        else:
            if not origin:
                seen_commit_shas[sha] = pr
        if not origin:
            prs_by_repo[repo].append(sha)
    if detected_duplicate_prs := (len(seen_commit_shas) + pure_commits_count) < len(commits_df):
        log = logging.getLogger(f"{__name__}._fetch_commit_stats")
        log.warning(
            "detected %d duplicate PRs",
            len(commits_df) - pure_commits_count - len(seen_commit_shas),
        )
    del seen_commit_shas
    if has_logical:
        pr_labels = await fetch_labels_to_filter(pr_ids, meta_ids, mdb)
    else:
        pr_labels = None
    sha_order = np.argsort(shas)
    del shas
    lines = lines[sha_order]
    commit_authors = commit_authors[sha_order]
    pr_repo_names = np.array(pr_repo_names, dtype="U")
    if with_extended_prs:
        pr_titles = np.array(pr_titles, dtype=object)
    if with_jira and jira_ids is not None:
        jira_task = asyncio.create_task(
            fetch_jira_issues_by_prs(pr_ids, jira_ids, meta_ids, mdb),
            name="mine_deployments/_fetch_commit_stats/jira",
        )
    else:
        jira_task = None
    extra_jira_task = None

    if has_logical:
        df_body = {
            PullRequest.node_id.name: pr_ids,
            PullRequest.repository_full_name.name: pr_repo_names,
            PullRequest.title.name: pr_titles,
            PullRequest.merge_commit_sha.name: merge_shas,
            PullRequest.user_node_id.name: pr_authors,
            "lines": pr_lines,
        }
        if with_extended_prs:
            df_body.update(
                {
                    PullRequest.number.name: pr_numbers,
                    PullRequest.additions.name: pr_additions,
                    PullRequest.deletions.name: pr_deletions,
                    PullRequest.created_at.name: pr_createds,
                },
            )
        prs_df = pd.DataFrame(df_body)
        prs_df = split_logical_prs(
            prs_df,
            pr_labels,
            logical_settings.with_logical_prs(unordered_unique(pr_repo_names)),
            logical_settings,
            reindex=False,
            reset_index=False,
        )
        merge_shas = prs_df[PullRequest.merge_commit_sha.name].values
        pr_ids = prs_df[PullRequest.node_id.name].values
        pr_authors = prs_df[PullRequest.user_node_id.name].values
        pr_lines = prs_df["lines"].values
        pr_repo_names = prs_df[PullRequest.repository_full_name.name].values
        if with_extended_prs:
            pr_numbers = prs_df[PullRequest.number.name].values
            pr_titles = prs_df[PullRequest.title.name].values
            pr_additions = prs_df[PullRequest.additions.name].values
            pr_deletions = prs_df[PullRequest.deletions.name].values
            pr_createds = prs_df[PullRequest.created_at.name].values

    sha_order = np.argsort(merge_shas)
    merge_shas = merge_shas[sha_order]
    unique_merge_shas, merge_sha_ff, merge_sha_counts = np.unique(
        merge_shas, return_index=True, return_counts=True,
    )
    pr_ids = pr_ids[sha_order]
    pr_authors = pr_authors[sha_order]
    pr_lines = pr_lines[sha_order]
    pr_repo_names = pr_repo_names[sha_order]
    if with_extended_prs:
        pr_numbers = pr_numbers[sha_order]
        pr_titles = pr_titles[sha_order]
        pr_additions = pr_additions[sha_order]
        pr_deletions = pr_deletions[sha_order]
        pr_createds = pr_createds[sha_order]
    pr_commits = np.zeros_like(pr_lines)
    for repo, hashes in prs_by_repo.items():
        try:
            dag = dags[repo][1]
        except KeyError:
            # this is perfectly valid because the same commit sha can appear in several
            # repositories, not all of them appear in deployments
            continue
        pr_hashes = extract_pr_commits(*dag, np.array(hashes, dtype="S40"))
        commit_counts = nested_lengths(pr_hashes)
        my_sha_indexes = np.searchsorted(unique_merge_shas, hashes)
        if has_logical or detected_duplicate_prs:  # == (len(unique_merge_shas) != len(merge_shas))
            my_merge_counts = merge_sha_counts[my_sha_indexes]
            commit_counts = np.repeat(commit_counts, my_merge_counts)
            my_sha_indexes = merge_sha_ff[my_sha_indexes]
            my_sha_indexes = np.repeat(
                my_sha_indexes + my_merge_counts - my_merge_counts.cumsum(), my_merge_counts,
            ) + np.arange(len(commit_counts))
        pr_commits[my_sha_indexes] = commit_counts
    selected = [
        NodePullRequest.id,
        NodePullRequest.repository_id,
        NodePullRequest.number,
        NodePullRequest.author_id,
    ]
    if with_extended_prs:
        selected.extend(
            [
                NodePullRequest.title,
                NodePullRequest.created_at,
                NodePullRequest.additions,
                NodePullRequest.deletions,
                NodePullRequest.number,
            ],
        )
    else:
        selected.append((NodePullRequest.additions + NodePullRequest.deletions).label("lines"))
        if has_logical:
            selected.append(NodePullRequest.title)
    if rebased_map:

        @sentry_span
        async def fetch_successfully_deployed_rebased_prs() -> dict[str, dict[str, set[int]]]:
            df = await read_sql_query(
                select(
                    GitHubPullRequestDeployment.pull_request_id,
                    GitHubPullRequestDeployment.repository_full_name,
                    GitHubPullRequestDeployment.deployment_name,
                ).where(
                    GitHubPullRequestDeployment.acc_id == account,
                    GitHubPullRequestDeployment.pull_request_id.in_(rebased_map),
                ),
                pdb,
                [
                    GitHubPullRequestDeployment.pull_request_id,
                    GitHubPullRequestDeployment.repository_full_name,
                    GitHubPullRequestDeployment.deployment_name,
                ],
            )
            dep_info = await read_sql_query(
                select(DeploymentNotification.name, DeploymentNotification.environment).where(
                    DeploymentNotification.conclusion
                    == DeploymentNotification.CONCLUSION_SUCCESS.decode(),
                    DeploymentNotification.name.in_(
                        df[GitHubPullRequestDeployment.deployment_name.name].unique(),
                    ),
                ),
                rdb,
                [DeploymentNotification.name, DeploymentNotification.environment],
            )
            dep_info_dict = {}
            for name, env in zip(
                dep_info[DeploymentNotification.name.name].values,
                dep_info[DeploymentNotification.environment.name].values,
            ):
                dep_info_dict[name] = env
            result = {}
            for dep, repo, pr in zip(
                df[GitHubPullRequestDeployment.deployment_name.name].values,
                df[GitHubPullRequestDeployment.repository_full_name.name].values,
                df[GitHubPullRequestDeployment.pull_request_id.name].values,
            ):
                try:
                    result.setdefault(dep_info_dict[dep], {}).setdefault(repo, []).append(pr)
                except KeyError:
                    continue
            for repos in result.values():
                for repo, prs in repos.items():
                    repos[repo] = np.sort(prs)
            return result

        rebased_pr_rows, already_deployed_rebased = await gather(
            mdb.fetch_all(
                select(selected).where(
                    NodePullRequest.acc_id.in_(meta_ids),
                    NodePullRequest.node_id.in_(rebased_map),
                ),
            ),
            fetch_successfully_deployed_rebased_prs(),
            op="_fetch_commit_stats/rebased",
        )
    else:
        rebased_pr_rows = []
        already_deployed_rebased = {}
    rebased_prs_by_repo = {}
    for pr_row in rebased_pr_rows:
        rebased_prs_by_repo.setdefault(pr_row[NodePullRequest.repository_id.name], []).append(
            pr_row,
        )
    del rebased_pr_rows
    repo_node_to_name = prefixer.repo_node_to_name.__getitem__
    extra_merges = []
    extra_pr_ids = []
    extra_pr_authors = []
    extra_pr_lines = []
    extra_pr_commits = []
    extra_pr_repo_names = []
    extra_pr_numbers = []
    extra_pr_titles = []
    extra_pr_createds = []
    extra_pr_additions = []
    extra_pr_deletions = []
    for repo_id, pr_rows in rebased_prs_by_repo.items():
        dag = dags[(repo_name := repo_node_to_name(repo_id))][1]
        rebased_merge_shas = np.array(
            [rebased_map[pr_row[NodePullRequest.id.name]] for pr_row in pr_rows], dtype="S40",
        )
        pr_hashes = extract_pr_commits(*dag, rebased_merge_shas)
        for pr, merge_sha, shas in zip(pr_rows, rebased_merge_shas, pr_hashes):
            extra_merges.append(merge_sha)
            extra_pr_ids.append(pr[NodePullRequest.id.name])
            extra_pr_authors.append(pr[NodePullRequest.author_id.name] or 0)
            extra_pr_commits.append(len(shas))
            extra_pr_repo_names.append(repo_name)
            if with_extended_prs:
                extra_pr_numbers.append(pr[NodePullRequest.number.name])
                extra_pr_titles.append(pr[NodePullRequest.title.name])
                extra_pr_createds.append(pr[NodePullRequest.created_at.name])
                extra_pr_additions.append(additions := pr[NodePullRequest.additions.name])
                extra_pr_deletions.append(deletions := pr[NodePullRequest.deletions.name])
                extra_pr_lines.append(additions + deletions)
            else:
                extra_pr_lines.append(pr["lines"])
                if has_logical:
                    extra_pr_titles.append(pr[NodePullRequest.title.name])
    if with_extended_prs:
        extra_pr_numbers = np.array(extra_pr_numbers, dtype=int)
        extra_pr_titles = np.array(extra_pr_titles, dtype=object)
        extra_pr_additions = np.array(extra_pr_additions, dtype=int)
        extra_pr_deletions = np.array(extra_pr_deletions, dtype=int)
        extra_pr_createds = np.array(extra_pr_createds, dtype="datetime64[s]")
    if extra_merges:
        if with_jira and jira_ids is not None:
            extra_jira_task = asyncio.create_task(
                fetch_jira_issues_by_prs(extra_pr_ids, jira_ids, meta_ids, mdb),
                name="mine_deployments/_fetch_commit_stats/extra_jira",
            )
            await asyncio.sleep(0)
        if has_logical:
            extra_columns = {
                PullRequest.node_id.name: extra_pr_ids,
                PullRequest.repository_full_name.name: extra_pr_repo_names,
                PullRequest.title.name: extra_pr_titles,
                PullRequest.merge_commit_sha.name: extra_merges,
                PullRequest.user_node_id.name: extra_pr_authors,
                "lines": extra_pr_lines,
                "commits": extra_pr_commits,
            }
            if with_extended_prs:
                extra_columns[PullRequest.number.name] = extra_pr_numbers
                extra_columns[PullRequest.created_at.name] = extra_pr_createds
                extra_columns[PullRequest.additions.name] = extra_pr_additions
                extra_columns[PullRequest.deletions.name] = extra_pr_deletions
            prs_df = pd.DataFrame(extra_columns)
            prs_df = split_logical_prs(
                prs_df,
                pr_labels,
                logical_settings.with_logical_prs(extra_pr_repo_names),
                logical_settings,
                reindex=False,
                reset_index=False,
            )
            extra_merges = prs_df[PullRequest.merge_commit_sha.name].values
            extra_pr_ids = prs_df[PullRequest.node_id.name].values
            extra_pr_authors = prs_df[PullRequest.user_node_id.name].values
            extra_pr_lines = prs_df["lines"].values
            extra_pr_repo_names = prs_df[PullRequest.repository_full_name.name].values
            extra_pr_commits = prs_df["commits"].values
            if with_extended_prs:
                extra_pr_numbers = prs_df[PullRequest.number.name].values
                extra_pr_titles = prs_df[PullRequest.title.name].values
                extra_pr_createds = prs_df[PullRequest.created_at.name].values
                extra_pr_additions = prs_df[PullRequest.additions.name].values
                extra_pr_deletions = prs_df[PullRequest.deletions.name].values

        merge_shas = np.concatenate([merge_shas, extra_merges])
        pr_ids = np.concatenate([pr_ids, extra_pr_ids])
        pr_authors = np.concatenate([pr_authors, extra_pr_authors])
        pr_lines = np.concatenate([pr_lines, extra_pr_lines])
        pr_commits = np.concatenate([pr_commits, extra_pr_commits])
        pr_repo_names = np.concatenate([pr_repo_names, extra_pr_repo_names])
        if with_extended_prs:
            pr_numbers = np.concatenate([pr_numbers, extra_pr_numbers])
            pr_titles = np.concatenate([pr_titles, extra_pr_titles])
            pr_additions = np.concatenate([pr_additions, extra_pr_additions])
            pr_deletions = np.concatenate([pr_deletions, extra_pr_deletions])
            pr_createds = np.concatenate([pr_createds, extra_pr_createds])
        order = np.argsort(merge_shas)
        merge_shas = merge_shas[order]
        pr_ids = pr_ids[order]
        pr_authors = pr_authors[order]
        pr_lines = pr_lines[order]
        pr_commits = pr_commits[order]
        pr_repo_names = pr_repo_names[order]
        if with_extended_prs:
            pr_numbers = pr_numbers[order]
            pr_titles = pr_titles[order]
            pr_additions = pr_additions[order]
            pr_deletions = pr_deletions[order]
            pr_createds = pr_createds[order]
    if with_extended_prs:
        extended_prs = {
            PullRequest.number.name: pr_numbers,
            PullRequest.title.name: pr_titles,
            PullRequest.additions.name: pr_additions,
            PullRequest.deletions.name: pr_deletions,
            PullRequest.created_at.name: pr_createds,
        }
    else:
        extended_prs = {}
    return (
        _DeployedCommitStats(
            commit_authors=commit_authors,
            lines=lines,
            merge_shas=merge_shas,
            pull_requests=pr_ids,
            pr_authors=pr_authors,
            pr_lines=pr_lines,
            pr_commit_counts=pr_commits,
            pr_repository_full_names=pr_repo_names,
            ambiguous_prs=ambiguous_prs,
            already_deployed_rebased_by_env_by_repo=already_deployed_rebased,
            extended_prs=extended_prs,
        ),
        jira_task,
        extra_jira_task,
    )


@sentry_span
async def _extract_deployed_commits(
    notifications: pd.DataFrame,
    components: pd.DataFrame,
    deployed_commits_df: pd.DataFrame,
    commit_relationship: dict[str, dict[str, dict[int, dict[int, _CommitRelationship]]]],
    dags: dict[str, tuple[bool, DAG]],
) -> tuple[dict[str, dict[str, _DeployedCommitDetails]], np.ndarray]:
    commit_ids_in_df = deployed_commits_df[PushCommit.node_id.name].values
    commit_shas_in_df = deployed_commits_df[PushCommit.sha.name].values
    joined = notifications.join(components)
    commits = joined[DeployedComponent.resolved_commit_node_id.name].values
    conclusions = joined[DeploymentNotification.conclusion.name].values
    deployment_names = joined.index.values.astype("U")
    deployment_finished_ats = joined[DeploymentNotification.finished_at.name].values
    deployed_commits_per_repo_per_env = defaultdict(dict)
    all_mentioned_hashes = []
    for (env, repo_name), indexes in joined.groupby(
        [DeploymentNotification.environment.name, DeployedComponent.repository_full_name],
        sort=False,
    ).grouper.indices.items():
        dag = dags[drop_logical_repo(repo_name)][1]

        grouped_deployment_finished_ats = deployment_finished_ats[indexes]
        order = np.argsort(grouped_deployment_finished_ats)[::-1]
        indexes = indexes[order]
        deployed_commits = commits[indexes]
        deployed_ats = grouped_deployment_finished_ats[order]
        grouped_deployment_names = deployment_names[indexes]
        grouped_conclusions = conclusions[indexes]
        deployed_shas = commit_shas_in_df[np.searchsorted(commit_ids_in_df, deployed_commits)]
        successful = grouped_conclusions == DeploymentNotification.CONCLUSION_SUCCESS

        grouped_deployed_shas = np.zeros(len(deployed_commits), dtype=object)
        relationships = commit_relationship[env][repo_name]
        if successful.any():
            successful_deployed_shas = deployed_shas[successful]
            parent_shas = []
            for commit, dt in zip(deployed_commits[successful], deployed_ats[successful]):
                try:
                    relationship = relationships[commit][dt]
                except KeyError:
                    # that's OK, we are a duplicate successful deployment
                    # our own sha is already added, do nothing
                    continue
                else:
                    parent_shas.extend(relationship.parent_shas[relationship.externals])
            if len(parent_shas):
                parent_shas = np.unique(np.array(parent_shas, dtype="S40"))
            if len(parent_shas):
                all_shas = np.concatenate([successful_deployed_shas, parent_shas])
            else:
                all_shas = successful_deployed_shas
            # if there are same commits in `successful_deployed_shas` and `parent_shas`,
            # mark_dag_access() guarantees that `parent_shas` have the priority
            ownership = mark_dag_access(*dag, all_shas, True)
            # we have to add np.flatnonzero due to numpy's quirks
            grouped_deployed_shas[np.flatnonzero(successful)] = group_hashes_by_ownership(
                ownership, dag[0], len(all_shas), None,
            )[: len(successful_deployed_shas)]
        failed = np.flatnonzero(~successful)
        if len(failed):
            failed_shas = deployed_shas[failed]
            failed_commits = deployed_commits[failed]
            failed_deployed_ats = deployed_ats[failed]
            failed_parents = np.zeros(len(failed_shas), dtype=object)
            unkeys = np.empty(len(failed_shas), dtype=object)
            for i, (commit, deployed_at) in enumerate(zip(failed_commits, failed_deployed_ats)):
                relationship = relationships[commit][deployed_at]
                failed_parents[i] = relationship.parent_shas
                unkeys[i] = b"".join([commit.data, relationship.parent_node_ids.data])
            unkeys = unkeys.astype("S")
            _, unique_indexes, unique_remap = np.unique(
                unkeys, return_index=True, return_inverse=True,
            )
            grouped_deployed_shas[failed] = extract_independent_ownership(
                *dag,
                failed_shas[unique_indexes],
                failed_parents[unique_indexes],
            )[unique_remap]

        deployed_commits_per_repo_per_env[env][repo_name] = _DeployedCommitDetails(
            grouped_deployed_shas, grouped_deployment_names,
        )
        all_mentioned_hashes.extend(grouped_deployed_shas)
    all_mentioned_hashes = np.unique(np.concatenate(all_mentioned_hashes))
    return deployed_commits_per_repo_per_env, all_mentioned_hashes


@sentry_span
async def _resolve_commit_relationship(
    notifications: pd.DataFrame,
    components: pd.DataFrame,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> tuple[
    dict[str, dict[str, dict[int, dict[int, _CommitRelationship]]]],
    dict[str, tuple[bool, DAG]],
    pd.DataFrame,
]:
    log = logging.getLogger(f"{metadata.__package__}._resolve_commit_relationship")
    until_per_repo_env = defaultdict(dict)
    joined = components.join(notifications)
    finished_ats = joined[DeploymentNotification.finished_at.name].values
    commit_ids = joined[DeployedComponent.resolved_commit_node_id.name].values
    successful = (
        joined[DeploymentNotification.conclusion.name].values
        == DeploymentNotification.CONCLUSION_SUCCESS
    )
    commits_per_repo_per_env = defaultdict(dict)
    for (env, repo_name), indexes in joined.groupby(
        [DeploymentNotification.environment.name, DeployedComponent.repository_full_name],
        sort=False,
    ).grouper.indices.items():
        until_per_repo_env[env][repo_name] = pd.Timestamp(
            finished_ats[indexes].min(), tzinfo=timezone.utc,
        )

        # separate successful and unsuccessful deployments
        env_repo_successful = successful[indexes]
        failed_indexes = indexes[~env_repo_successful]
        indexes = indexes[env_repo_successful]

        # order by deployment date, ascending
        env_repo_deployed = finished_ats[indexes]
        order = np.argsort(env_repo_deployed)
        env_repo_deployed = env_repo_deployed[order]
        env_repo_commit_ids = commit_ids[indexes[order]]

        # there can be commit duplicates, remove them
        env_repo_commit_ids, first_encounters = np.unique(env_repo_commit_ids, return_index=True)
        env_repo_deployed = env_repo_deployed[first_encounters]
        # thus we selected the earliest deployment for each unique commit

        # reverse the time order - required by mark_dag_access
        order = np.argsort(env_repo_deployed)[::-1]
        env_repo_commit_ids = env_repo_commit_ids[order]
        env_repo_deployed = env_repo_deployed[order]

        commits_per_repo_per_env[env][repo_name] = (
            env_repo_commit_ids,
            env_repo_deployed,
            commit_ids[failed_indexes],
            finished_ats[failed_indexes],
        )
    del joined
    del finished_ats
    del commit_ids
    log.info(
        "finding previous successful deployments of %d repositories in %d environments",
        sum(len(v) for v in until_per_repo_env.values()),
        len(until_per_repo_env),
    )
    commits_per_physical_repo = {}
    for env_commits_per_repo in commits_per_repo_per_env.values():
        for repo_name, (successful_commits, _, failed_commits, _) in env_commits_per_repo.items():
            commits_per_physical_repo.setdefault(drop_logical_repo(repo_name), []).extend(
                (successful_commits, failed_commits),
            )
    for repo, commits in commits_per_physical_repo.items():
        commits_per_physical_repo[repo] = np.unique(np.concatenate(commits))
    (dags, deployed_commits_df), previous = await gather(
        fetch_dags_with_commits(
            commits_per_physical_repo, True, account, meta_ids, mdb, pdb, cache,
        ),
        _append_latest_deployed_components(
            until_per_repo_env, {}, logical_settings, prefixer, account, meta_ids, mdb, rdb,
        ),
        op="_compute_deployment_facts/dags_and_latest",
    )
    assert (
        not deployed_commits_df.empty
    ), f"metadata lost all these commits: {commits_per_physical_repo}"
    del commits_per_physical_repo
    deployed_commits_df.sort_values(PushCommit.node_id.name, ignore_index=True, inplace=True)
    commit_ids_in_df = deployed_commits_df[PushCommit.node_id.name].values
    commit_shas_in_df = deployed_commits_df[PushCommit.sha.name].values
    root_details_per_repo = defaultdict(dict)
    commit_relationship = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    missing_commits = {}
    for env, env_commits_per_repo in commits_per_repo_per_env.items():
        for repo_name, (
            successful_commits,
            successful_deployed_ats,
            failed_commits,
            failed_deployed_ats,
        ) in env_commits_per_repo.items():
            my_relationships = commit_relationship[env][repo_name]
            unique_failed_commits, failed_remap = np.unique(failed_commits, return_inverse=True)
            all_commits = np.concatenate([successful_commits, unique_failed_commits])
            found_indexes = searchsorted_inrange(commit_ids_in_df, all_commits)
            missed_mask = commit_ids_in_df[found_indexes] != all_commits
            if missed_mask.any():
                missing_commits[repo_name] = np.unique(all_commits[missed_mask]).tolist()
                continue
            found_successful_indexes = found_indexes[: len(successful_commits)]
            found_failed_indexes = found_indexes[len(successful_commits) :][failed_remap]
            successful_shas = commit_shas_in_df[found_successful_indexes]
            failed_shas = commit_shas_in_df[found_failed_indexes]
            dag = dags[drop_logical_repo(repo_name)][1]
            # commits deployed earlier must steal the ownership despite the DAG relationships
            ownership = mark_dag_access(*dag, successful_shas, True)
            all_shas = np.concatenate([successful_shas, failed_shas])
            all_deployed_ats = np.concatenate([successful_deployed_ats, failed_deployed_ats])
            parents = mark_dag_parents(*dag, all_shas, all_deployed_ats, ownership)
            noroot_mask = nested_lengths(parents).astype(bool)
            root_mask = ~noroot_mask
            all_commits = np.concatenate([successful_commits, failed_commits])
            root_details_per_repo[env][repo_name] = (
                all_commits[root_mask],
                all_shas[root_mask],
                all_deployed_ats[root_mask],
                root_mask[: len(successful_commits)].sum(),
            )
            for index, my_parents in zip(np.flatnonzero(noroot_mask), parents[noroot_mask]):
                my_relationships[all_commits[index]][
                    all_deployed_ats[index]
                ] = _CommitRelationship(
                    all_commits[my_parents],
                    all_shas[my_parents],
                    np.zeros(len(my_parents), dtype=bool),
                )
        assert (
            not missing_commits
        ), f"cannot analyze deployments to {env} with missing commit node IDs: {missing_commits}"
    del commits_per_repo_per_env
    missing_sha = b"0" * 40
    suspects = defaultdict(dict)
    dags = await _extend_dags_with_previous_commits(previous, dags, account, meta_ids, mdb, pdb)

    # there may be older commits deployed earlier, disregarding our loading in batches
    for env, repos in previous.items():
        for repo_name, (_, shas, _) in repos.items():
            if shas[0] == missing_sha:
                suspects[env][repo_name] = []
                continue

            root_ids, root_shas, root_deployed_ats, success_len = root_details_per_repo[env][
                repo_name
            ]
            dag = dags[drop_logical_repo(repo_name)][1]
            successful_shas = np.concatenate([root_shas[:success_len], shas])
            # even if `sha` is a child, it was deployed earlier, hence must steal the ownership
            ownership = mark_dag_access(*dag, successful_shas, True)
            suspects[env][repo_name] = dag[0][ownership < success_len]
    # fetch and merge those "impossible" deployments
    await _append_latest_deployed_components(
        until_per_repo_env,
        previous,
        logical_settings,
        prefixer,
        account,
        meta_ids,
        mdb,
        rdb,
        suspects=suspects,
    )

    while until_per_repo_env:
        removed_previous = []
        for env, repos in previous.items():
            for repo_name, (cids, shas, dep_finished_ats) in repos.items():
                my_relationships = commit_relationship[env][repo_name]
                root_ids, root_shas, root_deployed_ats, success_len = root_details_per_repo[env][
                    repo_name
                ]
                dag = dags[drop_logical_repo(repo_name)][1]

                successful_shas = np.concatenate([root_shas[:success_len], shas])
                # even if `sha` is a child, it was deployed earlier, hence must steal the ownership
                ownership = mark_dag_access(*dag, successful_shas, True)
                all_shas = np.concatenate([successful_shas, root_shas[success_len:]])
                all_deployed_ats = np.concatenate(
                    [
                        root_deployed_ats[:success_len],
                        np.array(dep_finished_ats, dtype="datetime64[s]"),
                        root_deployed_ats[success_len:],
                    ],
                )
                reached_root = until_per_repo_env[env][repo_name] == datetime.min
                parents = mark_dag_parents(
                    *dag, all_shas, all_deployed_ats, ownership, slay_hydra=not reached_root,
                )
                if reached_root:
                    noroot_mask = np.ones(len(parents), dtype=bool)
                else:
                    noroot_mask = nested_lengths(parents).astype(bool)
                root_mask = ~noroot_mask
                # oldest commits that we've just inserted
                root_mask[success_len : success_len + len(cids)] = False
                noroot_mask[success_len : success_len + len(cids)] = False
                all_commit_ids = np.concatenate(
                    [
                        root_ids[:success_len],
                        cids,
                        root_ids[success_len:],
                    ],
                )
                root_details_per_repo[env][repo_name] = (
                    all_commit_ids[root_mask],
                    root_shas := all_shas[root_mask],
                    all_deployed_ats[root_mask],
                    root_mask[:success_len].sum(),
                )
                for index, my_parents in zip(np.flatnonzero(noroot_mask), parents[noroot_mask]):
                    my_relationships[all_commit_ids[index]][
                        all_deployed_ats[index]
                    ] = _CommitRelationship(
                        all_commit_ids[my_parents],
                        all_shas[my_parents],
                        (success_len <= my_parents) & (my_parents < success_len + len(cids)),
                    )
                if len(root_shas) > 0:
                    # there are still unresolved parents, we need to descend deeper
                    until_per_repo_env[env][repo_name] = min(dep_finished_ats)
                else:
                    del until_per_repo_env[env][repo_name]
                    removed_previous.append((env, repo_name))
            if not until_per_repo_env[env]:
                del until_per_repo_env[env]
        for env, repo_name in removed_previous:
            del previous[env][repo_name]
        if until_per_repo_env:
            await _append_latest_deployed_components(
                until_per_repo_env,
                previous,
                logical_settings,
                prefixer,
                account,
                meta_ids,
                mdb,
                rdb,
            )
            dags = await _extend_dags_with_previous_commits(
                previous, dags, account, meta_ids, mdb, pdb,
            )
    return commit_relationship, dags, deployed_commits_df


@sentry_span
async def _extend_dags_with_previous_commits(
    previous: dict[str, dict[str, tuple[list[int], list[bytes], list[datetime]]]],
    dags: dict[str, tuple[bool, DAG]],
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
) -> dict[str, tuple[bool, DAG]]:
    records = {}
    missing_sha = b"0" * 40
    for repos in previous.values():
        for repo, (cids, shas, dep_finished_ats) in repos.items():
            physical_repo = drop_logical_repo(repo)
            for cid, sha, dep_finished_at in zip(cids, shas, dep_finished_ats):
                if sha != missing_sha:
                    records[cid] = (sha, physical_repo, dep_finished_at)
    if not records:
        return dags
    previous_commits_df = pd.DataFrame.from_dict(records, orient="index")
    previous_commits_df.index.name = PushCommit.node_id.name
    previous_commits_df.columns = [
        PushCommit.sha.name,
        PushCommit.repository_full_name.name,
        PushCommit.committed_date.name,
    ]
    previous_commits_df[PushCommit.sha.name] = previous_commits_df[
        PushCommit.sha.name
    ].values.astype("S40")
    previous_commits_df.reset_index(inplace=True)
    return await fetch_repository_commits(
        dags,
        previous_commits_df,
        COMMIT_FETCH_COMMITS_COLUMNS,
        False,
        account,
        meta_ids,
        mdb,
        pdb,
        None,  # disable the cache
    )


@sentry_span
async def _append_latest_deployed_components(
    until_per_repo_env: dict[str, dict[str, datetime]],
    previous: dict[str, dict[str, tuple[list[int], list[bytes], list[datetime]]]],
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    rdb: Database,
    suspects: Optional[dict[str, dict[str, np.ndarray]]] = None,
    batch: int = 10,
) -> dict[str, dict[str, tuple[list[int], list[bytes], list[datetime]]]]:
    until_per_repo_env_logical = {}
    until_per_repo_env_physical = {}
    for env, repos in until_per_repo_env.items():
        for repo, ts in repos.items():
            try:
                logical_settings.deployments(physical_repo := drop_logical_repo(repo))
            except KeyError:
                assert physical_repo == repo, f"{physical_repo} misses logical deployment settings"
                until_per_repo_env_physical.setdefault(env, {})[repo] = ts
            else:
                # fmt: off
                (
                    until_per_repo_env_logical
                    .setdefault(env, {})
                    .setdefault(physical_repo, {})
                )[repo] = ts
                # fmt: on
    if suspects is not None:
        for env, env_suspects in suspects.items():
            for repo, commits in env_suspects.items():
                if len(commits) == 0:
                    try:
                        del until_per_repo_env_physical[env][repo]
                    except KeyError:
                        pass
                    try:
                        del until_per_repo_env_logical[env][repo]
                    except KeyError:
                        pass
        all_shas = list(chain.from_iterable(v.values() for v in suspects.values()))
        all_shas = unordered_unique(np.concatenate(all_shas, dtype="S40"))
        if len(all_shas):
            sha_df = await read_sql_query(
                select(NodeCommit.id, NodeCommit.sha).where(
                    NodeCommit.acc_id.in_(meta_ids), NodeCommit.sha.in_any_values(all_shas),
                ),
                mdb,
                [NodeCommit.id, NodeCommit.sha],
            )
            order = np.argsort(sha_df[NodeCommit.sha.name].values)
            all_shas = sha_df[NodeCommit.sha.name].values[order]
            all_ids = sha_df[NodeCommit.id.name].values[order]
            for repo_suspects in suspects.values():
                for repo, shas in repo_suspects.items():
                    repo_suspects[repo] = all_ids[np.searchsorted(all_shas, shas)]
    queries = [
        *_compose_latest_deployed_components_physical(
            until_per_repo_env_physical, suspects, prefixer, account, batch,
        ),
        *_compose_latest_deployed_components_logical(
            until_per_repo_env_logical, suspects, logical_settings, prefixer, account, batch,
        ),
    ]
    next_previous = await _fetch_latest_deployed_components_queries(queries, meta_ids, mdb, rdb)
    missing_sha = b"0" * 40
    for env, repos in until_per_repo_env.items():
        repo_commits = previous.setdefault(env, {})
        next_repo_commits = next_previous.setdefault(env, {})
        for repo in repos:
            if repo not in next_repo_commits:
                if suspects is None:
                    until_per_repo_env[env][repo] = datetime.min
                    repo_commits.setdefault(repo, ([0], [missing_sha], [datetime.min]))
            else:
                intel = repo_commits.setdefault(repo, ([], [], []))
                for i, val in enumerate(next_repo_commits[repo]):
                    intel[i].extend(val)
    return previous


def _compose_logical_filters_of_deployments(
    repo: str,
    repo_settings: LogicalDeploymentSettings,
) -> list[UnaryExpression]:
    logical_filters = []
    try:
        title_re = repo_settings.title(repo)
    except KeyError:
        pass
    else:
        logical_filters.append(DeploymentNotification.name.regexp_match(title_re.pattern))
    try:
        labels = repo_settings.labels(repo)
    except KeyError:
        pass
    else:
        logical_filters.append(
            exists().where(
                DeploymentNotification.acc_id == DeployedLabel.acc_id,
                DeploymentNotification.name == DeployedLabel.deployment_name,
                or_(
                    *(
                        and_(DeployedLabel.key == key, DeployedLabel.value.in_(vals))
                        for key, vals in labels.items()
                    ),
                ),
            ),
        )
    assert logical_filters
    return logical_filters


def _compose_latest_deployed_components_logical(
    until_per_repo_env: dict[str, dict[str, dict[str, datetime]]],
    suspects_per_repo_env: Optional[dict[str, dict[str, Collection[int]]]],
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    batch: int,
) -> list[Select]:
    if not until_per_repo_env:
        return []
    repo_name_to_node = prefixer.repo_name_to_node
    queries = []
    CONCLUSION_SUCCESS = DeploymentNotification.CONCLUSION_SUCCESS.decode()
    for env, repos in until_per_repo_env.items():
        for repo_root, logicals in repos.items():
            repo_node_id = repo_name_to_node[repo_root]
            repo_settings = logical_settings.deployments(repo_root)
            for repo, until in logicals.items():
                if repo != repo_root:
                    logical_filters = _compose_logical_filters_of_deployments(repo, repo_settings)
                else:
                    logical_filters = [
                        not_(expr)
                        for expr in chain.from_iterable(
                            _compose_logical_filters_of_deployments(other_repo, repo_settings)
                            for other_repo in logicals.keys() - {repo}
                        )
                    ]
                queries.append(
                    select("*").select_from(  # use LIMIT inside UNION hack
                        select(
                            DeploymentNotification.environment,
                            literal_column(f"'{repo}'").label(
                                DeployedComponent.repository_full_name,
                            ),
                            DeploymentNotification.finished_at,
                            DeployedComponent.resolved_commit_node_id,
                        )
                        .select_from(
                            join(
                                DeploymentNotification,
                                DeployedComponent,
                                and_(
                                    DeploymentNotification.account_id
                                    == DeployedComponent.account_id,
                                    DeploymentNotification.name
                                    == DeployedComponent.deployment_name,
                                ),
                            ),
                        )
                        .where(
                            DeploymentNotification.account_id == account,
                            DeploymentNotification.environment == env,
                            DeploymentNotification.conclusion == CONCLUSION_SUCCESS,
                            DeploymentNotification.finished_at < until,
                            DeployedComponent.repository_node_id == repo_node_id,
                            *(
                                (
                                    DeployedComponent.resolved_commit_node_id.in_(
                                        suspects_per_repo_env[env][repo],
                                    ),
                                )
                                if suspects_per_repo_env is not None
                                else ()
                            ),
                            *logical_filters,
                        )
                        .order_by(desc(DeploymentNotification.finished_at))
                        .limit(batch if suspects_per_repo_env is None else None)
                        .subquery(),
                    ),
                )
    return queries


@sentry_span
def _compose_latest_deployed_components_physical(
    until_per_repo_env: dict[str, dict[str, datetime]],
    suspects_per_repo_env: Optional[dict[str, dict[str, Collection[int]]]],
    prefixer: Prefixer,
    account: int,
    batch: int,
) -> list[Select]:
    if not until_per_repo_env:
        return []
    repo_name_to_node = prefixer.repo_name_to_node
    CONCLUSION_SUCCESS = DeploymentNotification.CONCLUSION_SUCCESS.decode()
    queries = [
        select("*").select_from(  # use LIMIT inside UNION hack
            select(
                DeploymentNotification.environment,
                literal_column(f"'{repo}'").label(DeployedComponent.repository_full_name),
                DeploymentNotification.finished_at,
                DeployedComponent.resolved_commit_node_id,
            )
            .select_from(
                join(
                    DeploymentNotification,
                    DeployedComponent,
                    and_(
                        DeploymentNotification.account_id == DeployedComponent.account_id,
                        DeploymentNotification.name == DeployedComponent.deployment_name,
                    ),
                ),
            )
            .where(
                DeploymentNotification.account_id == account,
                DeploymentNotification.environment == env,
                DeploymentNotification.conclusion == CONCLUSION_SUCCESS,
                DeploymentNotification.finished_at < until,
                DeployedComponent.repository_node_id == repo_name_to_node[repo],
                *(
                    (
                        DeployedComponent.resolved_commit_node_id.in_(
                            suspects_per_repo_env[env][repo],
                        ),
                    )
                    if suspects_per_repo_env is not None
                    else ()
                ),
            )
            .order_by(desc(DeploymentNotification.finished_at))
            .limit(batch if suspects_per_repo_env is None else None)
            .subquery(),
        )
        for env, repos in until_per_repo_env.items()
        for repo, until in repos.items()
    ]
    return queries


@sentry_span
async def _fetch_latest_deployed_components_queries(
    queries: list[Select],
    meta_ids: tuple[int, ...],
    mdb: Database,
    rdb: Database,
) -> dict[str, dict[str, tuple[list[int], list[bytes], list[datetime]]]]:
    if not queries:
        return {}
    log = logging.getLogger(f"{__name__}._fetch_latest_deployed_components_queries")
    query = union_all(*queries) if len(queries) > 1 else queries[0]
    latest_df = await read_sql_query(
        query,
        rdb,
        [
            DeploymentNotification.environment,
            DeployedComponent.repository_full_name,
            DeploymentNotification.finished_at,
            DeployedComponent.resolved_commit_node_id,
        ],
    )
    if latest_df.empty:
        result = {}
    else:
        result_cids = defaultdict(lambda: defaultdict(list))
        result_shas = defaultdict(lambda: defaultdict(list))
        result_finisheds = defaultdict(lambda: defaultdict(list))
        commit_ids = latest_df[DeployedComponent.resolved_commit_node_id.name].values
        unique_commit_ids = np.unique(commit_ids)
        if len(unique_commit_ids) and unique_commit_ids[0] == 0:
            unique_commit_ids = unique_commit_ids[1:]
        sha_id_df = await read_sql_query(
            select(NodeCommit.id, NodeCommit.sha)
            .where(NodeCommit.acc_id.in_(meta_ids), NodeCommit.id.in_any_values(unique_commit_ids))
            .order_by(NodeCommit.id),
            mdb,
            [NodeCommit.id, NodeCommit.sha],
        )
        commits_shas = map_array_values(
            commit_ids,
            sha_id_df[NodeCommit.id.name].values,
            sha_id_df[NodeCommit.sha.name].values,
            "",
        )

        result = defaultdict(dict)
        for env, repo, cid, sha, finished_at in zip(
            latest_df[DeploymentNotification.environment.name].values,
            latest_df[DeployedComponent.repository_full_name].values,
            latest_df[DeployedComponent.resolved_commit_node_id.name].values,
            commits_shas,
            latest_df[DeploymentNotification.finished_at.name].values,
        ):
            if not sha:
                if cid != 0:
                    log.error("Commit id %d not found, ignoring deployed component", cid)
                else:
                    log.warning(
                        "Skipping unresolved component %s of notification finished at %s",
                        repo,
                        finished_at,
                    )
                continue
            result_shas[env][repo].append(sha)
            result_cids[env][repo].append(cid)
            result_finisheds[env][repo].append(pd.Timestamp(finished_at, tzinfo=timezone.utc))
        for env, repos_cids in result_cids.items():
            for repo, cids in repos_cids.items():
                result[env][repo] = (cids, result_shas[env][repo], result_finisheds[env][repo])
    return result


@sentry_span
async def _fetch_precomputed_deployment_facts(
    names: Collection[str],
    default_branches: dict[str, str],
    settings: ReleaseSettings,
    account: int,
    pdb: Database,
) -> pd.DataFrame:
    format_version = GitHubDeploymentFacts.__table__.columns[
        GitHubDeploymentFacts.format_version.name
    ].default.arg
    dep_rows = await pdb.fetch_all(
        select(
            GitHubDeploymentFacts.deployment_name,
            GitHubDeploymentFacts.release_matches,
            GitHubDeploymentFacts.data,
        ).where(
            GitHubDeploymentFacts.acc_id == account,
            GitHubDeploymentFacts.format_version == format_version,
            GitHubDeploymentFacts.deployment_name.in_any_values(names),
        ),
    )
    if not dep_rows:
        return pd.DataFrame(columns=DeploymentFacts.fnv)
    structs = []
    for row in dep_rows:
        if not _settings_are_compatible(
            row[GitHubDeploymentFacts.release_matches.name], settings, default_branches,
        ):
            continue
        structs.append(
            DeploymentFacts(
                row[GitHubDeploymentFacts.data.name],
                name=row[GitHubDeploymentFacts.deployment_name.name],
            ),
        )
    facts = df_from_structs(structs)
    facts.index = facts[DeploymentNotification.name.name].values
    return facts


def _settings_are_compatible(
    matches: str,
    settings: ReleaseSettings,
    default_branches: dict[str, str],
) -> bool:
    matches = json.loads(matches)
    for key, val in matches.items():
        if not settings.native[key].compatible_with_db(
            val, default_branches[drop_logical_repo(key)],
        ):
            return False
    return True


def _group_labels(df: pd.DataFrame) -> pd.DataFrame:
    groups = list(df.groupby(DeployedLabel.deployment_name.name, sort=False))
    grouped_labels = pd.DataFrame(
        {
            "deployment_name": [g[0] for g in groups],
            "labels": [g[1] for g in groups],
        },
    )
    for df in grouped_labels["labels"].values:
        df.reset_index(drop=True, inplace=True)
    grouped_labels.set_index("deployment_name", drop=True, inplace=True)
    return grouped_labels


@sentry_span
async def load_jira_issues_for_deployments(
    deployments: pd.DataFrame,
    jira_ids: Optional[JIRAConfig],
    mdb: Database,
) -> dict[str, PullRequestJIRAIssueItem]:
    """Fetch JIRA issues mentioned by deployed PRs."""
    if jira_ids is None or deployments.empty:
        return {}

    unique_jira_keys = unordered_unique(
        np.concatenate(deployments[DeploymentFacts.f.jira_ids].values),
    )
    rows = await fetch_jira_issues_by_keys(unique_jira_keys, jira_ids, mdb)
    return {row["id"]: PullRequestJIRAIssueItem(**row) for row in rows}


async def hide_outlier_first_deployments(
    deployment_facts: pd.DataFrame,
    account: int,
    meta_ids: Sequence[int],
    mdb: Database,
    pdb: Database,
    threshold: float = 100.0,
) -> None:
    """Hide the outlier first deployments by deleting their facts from pdb.

    A deployment is an outlier first for a repo if:
    - it's the first deployment for a given repo and environment
    - the time distance with the repository creation is at least `threshold` times
      the median of time distances of each deployment with the previous one
    """
    log = logging.getLogger(f"{__name__}.hide_outlier_first_deployments")
    log.info("searching for outlier first deployments")
    outlier_deploys = await _search_outlier_first_deployments(
        deployment_facts, meta_ids, mdb, log, threshold,
    )

    tables = (GitHubCommitDeployment, GitHubPullRequestDeployment, GitHubReleaseDeployment)
    # group outlier deploys by name and consider only deploys still to be hidden

    async def deploy_to_be_hidden(name: str) -> bool:
        selects = [
            sa.select(1).where(Table.acc_id == account, Table.deployment_name == name)
            for Table in tables
        ]
        return (await pdb.fetch_val(sa.union_all(*selects))) is not None

    grouped_deploys = [
        (name, [d.repository_full_name for d in name_groups])
        for name, name_groups in groupby(
            sorted(outlier_deploys, key=attrgetter("deployment_name")),
            key=attrgetter("deployment_name"),
        )
        if await deploy_to_be_hidden(name)
    ]

    stmts: list[Executable] = []
    for Table, (deploy_name, repositories) in product(tables, grouped_deploys):
        log.info("hiding outlier first deployment %s for repos %s", deploy_name, repositories)
        stmts.append(
            sa.delete(Table).where(
                Table.acc_id == account,
                Table.deployment_name == deploy_name,
                Table.repository_full_name.in_(repositories),
            ),
        )

    # clear GitHubDeploymentFacts by removing prs, releases, and commits from facts data
    FactsT = GitHubDeploymentFacts
    depl_facts_stmt = sa.select(GitHubDeploymentFacts).where(
        FactsT.acc_id == account, FactsT.deployment_name.in_(d[0] for d in grouped_deploys),
    )
    depl_facts_rows = await pdb.fetch_all(depl_facts_stmt)

    for row in depl_facts_rows:
        facts = DeploymentFacts(row[FactsT.data.name], name=row[FactsT.deployment_name.name])
        # TODO(vmarkovtsev): we currently ignore logical repositories
        new_facts = facts.with_nothing_deployed()
        match_cols = (
            FactsT.acc_id,
            FactsT.deployment_name,
            FactsT.release_matches,
            FactsT.format_version,
        )
        update_stmt = (
            sa.update(FactsT)
            .where(*(col == row[col.name] for col in match_cols))
            .values({FactsT.data: new_facts.data})
        )
        stmts.append(update_stmt)

    await gather(*[pdb.execute(stmt) for stmt in stmts], op="hide_first_deployments")


@dataclass(frozen=True, slots=True)
class _OutlierDeployment:
    repository_full_name: str
    deployment_name: str


async def _search_outlier_first_deployments(
    deployment_facts: pd.DataFrame,
    meta_ids: Sequence[int],
    mdb: Database,
    log: logging.Logger,
    threshold: float = 100.0,
) -> Sequence[_OutlierDeployment]:
    """Search the outlier first deployments."""
    # the dataframe with no deployment_facts will have no columns
    if deployment_facts.empty:
        return ()

    REPOSITORY_COLUMN = "repository"
    exploded_facts = deployment_facts.explode("repositories").rename(
        columns={"repositories": REPOSITORY_COLUMN},
    )

    # retrieve repo => creation time mapping for the physical repos
    mentioned_physical_repos = coerce_logical_repos(exploded_facts.repository.unique())
    repos_stmt = sa.select(Repository.full_name, Repository.created_at).where(
        Repository.acc_id.in_(meta_ids), Repository.full_name.in_(mentioned_physical_repos),
    )
    repo_creation_times: Mapping[str, np.datetime64] = {
        r[0]: np.datetime64(ensure_db_datetime_tz(r[1], mdb))
        for r in await mdb.fetch_all(repos_stmt)
    }

    grouped_facts = exploded_facts.groupby(
        [DeploymentNotification.environment.name, REPOSITORY_COLUMN],
    )

    result = set()

    for group_name, group in grouped_facts:
        env, repo = group_name

        success_mask = (
            group[DeploymentNotification.conclusion.name]
            == DeploymentNotification.CONCLUSION_SUCCESS
        )
        deploy_times = np.sort(group[DeploymentNotification.started_at.name].values[success_mask])

        if len(deploy_times) < 2:
            # missing two successful deployments to compute median deploy interval
            # skip the analysis, don't hide the deployment
            continue

        # deploy_times are declared by user so it's possible to have duplicated values and 0 median
        median_interval = np.median(np.diff(deploy_times)) or np.timedelta64(1, "s")

        first_deploy_idx = np.argmin(group["started_at"].values)
        first_deploy_time = group["started_at"].values[first_deploy_idx]
        try:
            repo_creation_time = repo_creation_times[drop_logical_repo(repo)]
        except KeyError:
            log.warning(
                'Repository "%s" found in deployment facts doesn\'t exist in mdb anymore',
                drop_logical_repo(repo),
            )
            continue

        first_deploy_interval = first_deploy_time - repo_creation_time

        if (first_deploy_interval / median_interval) > threshold:
            deploy = _OutlierDeployment(repo, group["name"].values[first_deploy_idx])
            # same deploy can be selected from different environments
            if deploy not in result:
                log.debug(
                    "Deployment %s detected as outlier first in repository %s",
                    deploy.deployment_name,
                    deploy.repository_full_name,
                )
                result.add(deploy)

    return tuple(result)


async def _fetch_extended_prs_for_facts(
    facts: pd.DataFrame,
    meta_ids: tuple[int, ...],
    mdb: Database,
) -> pd.DataFrame:
    if facts.empty:
        return pd.DataFrame()
    pr_node_ids = np.concatenate(facts[DeploymentFacts.f.prs].values, dtype=int, casting="unsafe")
    unique_pr_node_ids = np.unique(pr_node_ids)
    selected = [
        PullRequest.node_id,
        PullRequest.number,
        PullRequest.title,
        PullRequest.created_at,
        PullRequest.additions,
        PullRequest.deletions,
        PullRequest.user_node_id,
    ]
    any_values = len(unique_pr_node_ids) > 100
    queries = [
        select(*selected)
        .where(
            PullRequest.acc_id == acc_id,
            PullRequest.node_id.in_any_values(unique_pr_node_ids)
            if any_values
            else PullRequest.node_id.in_(unique_pr_node_ids),
        )
        .order_by(PullRequest.node_id)
        for acc_id in meta_ids
    ]
    if len(queries) == 1:
        query = queries[0]
    else:
        query = union_all(*queries)
    if any_values:
        query = (
            query.with_statement_hint("Leading(((pr *VALUES*) repo))")
            .with_statement_hint(f"Rows(pr *VALUES* #{len(unique_pr_node_ids)})")
            .with_statement_hint(f"Rows(pr *VALUES* repo #{len(unique_pr_node_ids)})")
            .with_statement_hint(f"Rows(pr *VALUES* repo ath #{len(unique_pr_node_ids)})")
            .with_statement_hint(f"Rows(pr *VALUES* repo math #{len(unique_pr_node_ids)})")
            .with_statement_hint(f"Rows(pr *VALUES* repo ath math #{len(unique_pr_node_ids)})")
        )
    prs_df = await read_sql_query(query, mdb, selected)
    dep_lengths = nested_lengths(facts[DeploymentFacts.f.prs].values)
    indexes = searchsorted_inrange(prs_df[PullRequest.node_id.name].values, pr_node_ids)
    found_mask = prs_df[PullRequest.node_id.name].values[indexes] == pr_node_ids
    assert found_mask.all()
    # we could support this but will have to carry a separate `prs_offsets` column
    # in reality, this never happens
    """
    indexes = indexes[found_mask]
    not_found_counts = np.sum(~found_mask, where=np.repeat(np.arange(len(facts), dep_lengths)))
    dep_lengths -= not_found_counts
    """
    dep_splits = np.cumsum(dep_lengths)[:-1]
    df = {}
    for col in (
        PullRequest.number,
        PullRequest.title,
        PullRequest.created_at,
        PullRequest.additions,
        PullRequest.deletions,
        PullRequest.user_node_id,
    ):
        col = col.name
        df[col] = np.split(prs_df[col].values[indexes], dep_splits)
    df = pd.DataFrame(df)
    df.index = facts.index
    return df


async def _fetch_deployed_jira(
    facts: pd.DataFrame,
    jira_ids: Optional[JIRAConfig],
    meta_ids: tuple[int, ...],
    mdb: Database,
) -> pd.DataFrame:
    if facts.empty or jira_ids is None:
        return pd.DataFrame()
    unique_pr_node_ids = np.unique(
        np.concatenate(facts[DeploymentFacts.f.prs].values, dtype=int, casting="unsafe"),
    )
    df_map = await fetch_jira_issues_by_prs(unique_pr_node_ids, jira_ids, meta_ids, mdb)
    return _apply_jira_to_deployment_facts(facts, df_map)


def _apply_jira_to_deployment_facts(facts: pd.DataFrame, df_map: pd.DataFrame) -> pd.DataFrame:
    repo_jira_ids, repo_jira_offsets, pr_jira_ids, pr_jira_offsets = split_prs_to_jira_ids(
        facts[DeploymentFacts.f.prs].values,
        facts[DeploymentFacts.f.prs_offsets].values,
        df_map["node_id"].values,
        df_map["jira_id"].values,
    )
    df = pd.DataFrame(
        {
            DeploymentFacts.f.jira_ids: repo_jira_ids,
            DeploymentFacts.f.jira_offsets: repo_jira_offsets,
            DeploymentFacts.f.prs_jira_ids: pr_jira_ids,
            DeploymentFacts.f.prs_jira_offsets: pr_jira_offsets,
        },
    )
    df.index = facts.index
    return df


async def reset_broken_deployments(account: int, pdb: Database, rdb: Database) -> int:
    """
    Detect multiple successful deployments of the same commit and invalidate those deployments.

    We are still investigating all the possible cases when this situation can happen.
    Meanwhile, the best we can do is to invalidate all the duplicates.
    """
    rows_envs, rows_commits = await gather(
        rdb.fetch_all(
            select(DeploymentNotification.name, DeploymentNotification.environment).where(
                DeploymentNotification.account_id == account,
                DeploymentNotification.conclusion
                == DeploymentNotification.CONCLUSION_SUCCESS.decode(),
            ),
        ),
        pdb.fetch_all(
            select(GitHubCommitDeployment.deployment_name, GitHubCommitDeployment.commit_id).where(
                GitHubCommitDeployment.acc_id == account,
            ),
        ),
    )
    env_map = dict(rows_envs)
    del rows_envs
    bad_deps = set()
    commit_env_deps = {}
    for row in rows_commits:
        try:
            key = (row[1], env_map[row[0]])
        except KeyError:
            bad_deps.add(row[0])
            continue
        if (dep := commit_env_deps.get(key)) is not None:
            bad_deps.add(row[0])
            bad_deps.add(dep)
        else:
            commit_env_deps[key] = row[0]
    if bad_deps:
        log = logging.getLogger(f"{metadata.__package__}.reset_broken_deployments")
        await _delete_precomputed_deployments(bad_deps, account, pdb)
        log.warning("invalidated broken deployments: %s", bad_deps)
    return len(bad_deps)
