import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import pickle
import re
from typing import Any, Callable, Collection, Iterable, Mapping, Optional, Sequence, Union

import aiomcache
import medvedi as md
from medvedi.accelerators import in1d_str, unordered_unique
from medvedi.merge_to_str import merge_to_str
import numpy as np
import numpy.typing as npt
import sentry_sdk
import sqlalchemy as sa
from sqlalchemy import and_, func, join, select, true as sql_true, union_all
from sqlalchemy.sql.elements import BinaryExpression

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import CancelCache, cached, middle_term_exptime, short_term_exptime
from athenian.api.db import (
    Database,
    DatabaseLike,
    add_pdb_hits,
    add_pdb_misses,
    dialect_specific_insert,
)
from athenian.api.defer import defer
from athenian.api.internal.logical_repos import (
    coerce_logical_repos,
    drop_logical_repo,
    is_logical_repo,
)
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.commit import (
    RELEASE_FETCH_COMMITS_COLUMNS,
    fetch_precomputed_commit_history_dags,
    fetch_repository_commits,
)
from athenian.api.internal.miners.github.dag_accelerated import (
    extract_subdag,
    mark_dag_access,
    mark_dag_parents,
    searchsorted_inrange,
)
from athenian.api.internal.miners.github.deployment_light import load_included_deployments
from athenian.api.internal.miners.github.label import (
    fetch_labels_to_filter,
    find_left_prs_by_labels,
)
from athenian.api.internal.miners.github.logical import split_logical_prs
from athenian.api.internal.miners.github.precomputed_releases import (
    compose_release_match,
    fetch_precomputed_releases_by_name,
    load_precomputed_release_facts,
    reverse_release_settings,
    store_precomputed_release_facts,
)
from athenian.api.internal.miners.github.pull_request import PullRequestMiner
from athenian.api.internal.miners.github.rebased_pr import match_rebased_prs
from athenian.api.internal.miners.github.release_load import (
    MineReleaseMetrics,
    ReleaseLoader,
    group_repos_by_release_match,
    match_groups_to_conditions,
)
from athenian.api.internal.miners.github.release_match import ReleaseToPullRequestMapper
from athenian.api.internal.miners.github.released_pr import matched_by_column
from athenian.api.internal.miners.github.user import UserAvatarKeys, mine_user_avatars
from athenian.api.internal.miners.jira.issue import PullRequestJiraMapper, generate_jira_prs_query
from athenian.api.internal.miners.participation import (
    ReleaseParticipants,
    ReleaseParticipationKind,
)
from athenian.api.internal.miners.types import (
    Deployment,
    JIRAEntityToFetch,
    LoadedJIRADetails,
    LoadedJIRAReleaseDetails,
    PullRequestFacts,
    ReleaseFacts,
    released_prs_columns,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.refetcher import Refetcher
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseMatch, ReleaseSettings
from athenian.api.io import deserialize_args, serialize_args
from athenian.api.models.metadata.github import (
    NodeCommit,
    NodePullRequest,
    PullRequest,
    PullRequestLabel,
    PushCommit,
    Release,
)
from athenian.api.models.precomputed.models import (
    GitHubDonePullRequestFacts,
    GitHubRebasedPullRequest,
    GitHubReleaseDeployment,
)
from athenian.api.native.mi_heap_destroy_stl_allocator import make_mi_heap_allocator_capsule
from athenian.api.object_arrays import is_null, nested_lengths
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import df_from_structs


async def mine_releases(
    repos: Collection[str],
    participants: ReleaseParticipants,
    branches: md.DataFrame,
    default_branches: dict[str, str],
    time_from: Optional[datetime],
    time_to: Optional[datetime],
    labels: LabelFilter,
    jira: JIRAFilter,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
    *,
    force_fresh: bool = False,
    with_avatars: bool = True,
    with_extended_pr_details: bool = False,
    with_deployments: bool = True,
    with_jira: JIRAEntityToFetch = JIRAEntityToFetch.NOTHING,
    releases_in_time_range: Optional[md.DataFrame] = None,
    metrics: Optional[MineReleaseMetrics] = None,
    refetcher: Optional[Refetcher] = None,
) -> tuple[
    md.DataFrame,
    Union[list[tuple[int, str]], list[int]],
    dict[str, ReleaseMatch],
    dict[str, Deployment],
]:
    """Collect details about each release published between `time_from` and `time_to` and \
    calculate various statistics.

    :param participants: Mapping from roles to node IDs.
    :param force_fresh: Ensure that we load the most up to date releases, no matter the state of \
                        the pdb is.
    :param with_avatars: Indicates whether to return the fetched user avatars or just an array of \
                         unique mentioned user node IDs.
    :param with_extended_pr_details: Indicates whether released PR titles and creation timestamps \
                                     must be fetched.
    :param with_deployments: Indicates whether we must load the deployments to which the filtered
                             releases belong.
    :param with_jira: Indicates which JIRA information to load for each release from \
                      the released PR mapped to JIRA issues.
    :param releases_in_time_range: Shortcut to skip the initial loading of releases in \
                                   [time_from, time_to).
    :param metrics: Report any mining error statistics there.
    :param refetcher: Metadata self-healer to fix broken commit DAGs, etc.
    :return: 1. list of releases (general info, computed facts). \
             2. User avatars if `with_avatars` else *only newly mined* mentioned people nodes. \
             3. Release matched_by-s.
             4. Deployments that mention the returned releases. Empty if not `with_deployments`.
    """
    result = await _mine_releases(
        repos,
        participants,
        branches,
        default_branches,
        time_from,
        time_to,
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
        force_fresh,
        with_avatars,
        with_extended_pr_details,
        with_deployments,
        with_jira,
        releases_in_time_range,
        metrics,
        refetcher,
    )
    if metrics is not None:
        _set_count_metrics(result[0], metrics)
    return result[:4]


def _triage_flags(
    result: tuple[
        md.DataFrame,
        Union[list[tuple[int, str]], list[int]],
        dict[str, ReleaseMatch],
        dict[str, Deployment],
        bool,
        bool,
        bool,
        JIRAEntityToFetch,
    ],
    with_avatars: bool = True,
    with_extended_pr_details: bool = False,
    with_deployments: bool = True,
    with_jira: JIRAEntityToFetch = JIRAEntityToFetch.NOTHING,
    **_,
) -> tuple[
    md.DataFrame,
    Union[list[tuple[int, str]], list[int]],
    dict[str, ReleaseMatch],
    dict[str, Deployment],
    bool,
    bool,
    bool,
    JIRAEntityToFetch,
]:
    (
        main,
        avatars,
        matches,
        deps,
        cached_with_avatars,
        cached_with_extended_pr_details,
        cached_with_deployments,
        cached_with_jira,
    ) = result
    if with_extended_pr_details and not cached_with_extended_pr_details:
        raise CancelCache()
    if with_avatars and not cached_with_avatars:
        raise CancelCache()
    if not with_avatars and cached_with_avatars:
        avatars = [p[0] for p in avatars]
    if with_deployments and not cached_with_deployments:
        raise CancelCache()
    if with_jira & cached_with_jira != with_jira:
        raise CancelCache()
    return (
        main,
        avatars,
        matches,
        deps,
        with_avatars,
        with_extended_pr_details,
        with_deployments,
        with_jira,
    )


@sentry_span
@cached(
    exptime=middle_term_exptime,
    serialize=serialize_args,
    deserialize=deserialize_args,
    key=lambda repos, participants, time_from, time_to, labels, jira, release_settings, logical_settings, releases_in_time_range, **_: (  # noqa
        ",".join(sorted(repos)),
        ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(participants.items())),
        time_from,
        time_to,
        labels,
        jira,
        release_settings,
        logical_settings,
        repr(bytes(merge_to_str(releases_in_time_range[Release.node_id.name]).data))
        if releases_in_time_range is not None
        else "",
    ),
    postprocess=_triage_flags,
)
async def _mine_releases(
    repos: Collection[str],
    participants: ReleaseParticipants,
    branches: md.DataFrame,
    default_branches: dict[str, str],
    time_from: Optional[datetime],
    time_to: Optional[datetime],
    labels: LabelFilter,
    jira: JIRAFilter,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
    force_fresh: bool,
    with_avatars: bool,
    with_extended_pr_details: bool,
    with_deployments: bool,
    with_jira: JIRAEntityToFetch,
    releases_in_time_range: Optional[md.DataFrame],
    metrics: Optional[MineReleaseMetrics],
    refetcher: Optional[Refetcher],
) -> tuple[
    md.DataFrame,
    Union[list[tuple[int, str]], list[int]],
    dict[str, ReleaseMatch],
    dict[str, Deployment],
    bool,
    bool,
    bool,
    JIRAEntityToFetch,
]:
    log = logging.getLogger("%s.mine_releases" % metadata.__package__)
    if internal_releases := (releases_in_time_range is None):
        assert time_from is not None
        assert time_to is not None
        releases_in_time_range, matched_bys = await ReleaseLoader.load_releases(
            repos,
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
            force_fresh=force_fresh,
            metrics=metrics,
            refetcher=refetcher,
        )
        release_settings = ReleaseLoader.disambiguate_release_settings(
            release_settings, matched_bys,
        )
    else:
        assert time_from is None
        assert time_to is None
    if releases_in_time_range.empty:
        return (
            _empty_mined_releases_df(),
            [],
            {r: v.match for r, v in release_settings.prefixed.items()},
            {},
            with_avatars,
            with_extended_pr_details,
            with_deployments,
            with_jira,
        )
    has_logical_prs = logical_settings.has_logical_prs()
    if with_deployments:
        deployments_task = asyncio.create_task(
            _load_release_deployments(
                releases_in_time_range,
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
            ),
            name=f"_load_release_deployments({len(releases_in_time_range)})",
        )
    else:
        deployments_task = None
    precomputed_facts = await load_precomputed_release_facts(
        releases_in_time_range, default_branches, release_settings, account, pdb,
    )
    # uncomment this line to compute releases from scratch:
    # precomputed_facts = {}
    if with_extended_pr_details or labels:
        pr_node_ids, _ = ReleaseFacts.vectorize_field(
            precomputed_facts.values(), "prs_" + PullRequest.node_id.name,
        )
        if labels:
            precomputed_pr_labels_task = asyncio.create_task(
                fetch_labels_to_filter(pr_node_ids, meta_ids, mdb),
                name=f"fetch precomputed released PR labels({len(pr_node_ids)})",
            )
        else:
            precomputed_pr_labels_task = None
        if with_extended_pr_details:
            precomputed_pr_details_task = asyncio.create_task(
                _load_extended_pr_details(pr_node_ids, meta_ids, mdb),
                name=f"fetch precomputed released PR details({len(pr_node_ids)})",
            )
        else:
            precomputed_pr_details_task = None
        del pr_node_ids
    else:
        precomputed_pr_labels_task = precomputed_pr_details_task = None
    add_pdb_hits(pdb, "release_facts", len(precomputed_facts))
    unfiltered_precomputed_facts = precomputed_facts
    if jira:
        precomputed_facts = await _filter_precomputed_release_facts_by_jira(
            precomputed_facts, jira, meta_ids, mdb, cache,
        )
    if with_jira != JIRAEntityToFetch.NOTHING:
        precomputed_jira_entities_task = asyncio.create_task(
            _load_jira_from_precomputed_release_facts(precomputed_facts, with_jira, meta_ids, mdb),
            name="mine_releases/_load_jira_from_precomputed_release_facts",
        )
    else:
        precomputed_jira_entities_task = None
    new_pr_labels_coro = new_pr_details_coro = new_jira_entities_coro = None
    result, mentioned_authors, has_precomputed_facts = _build_mined_releases(
        releases_in_time_range, precomputed_facts, prefixer, True,
    )

    missing_release_indexes = np.flatnonzero(~has_precomputed_facts)
    missed_releases_count = len(missing_release_indexes)
    add_pdb_misses(pdb, "release_facts", missed_releases_count)
    missing_repos = unordered_unique(
        releases_in_time_range[Release.repository_full_name.name][missing_release_indexes],
    )
    commits_authors = prs_authors = []
    commits_authors_nz = prs_authors_nz = slice(0)
    repo_releases_analyzed = {}
    if missed_releases_count > 0:
        time_from = (
            releases_in_time_range[Release.published_at.name][missing_release_indexes[-1]]
            .item()
            .replace(tzinfo=timezone.utc)
        )
        time_to = (
            (
                releases_in_time_range[Release.published_at.name][missing_release_indexes[0]]
                + np.timedelta64(1, "s")
            )
            .item()
            .replace(tzinfo=timezone.utc)
        )
        if internal_releases:
            releases_in_time_range = releases_in_time_range.take(
                releases_in_time_range.isin(Release.repository_full_name.name, missing_repos),
            )
        (releases, *_, dags), first_commit_dates = await gather(
            ReleaseToPullRequestMapper.find_releases_for_matching_prs(
                missing_repos,
                branches,
                default_branches,
                time_from,
                time_to,
                False,
                release_settings,
                logical_settings,
                None,
                prefixer,
                account,
                meta_ids,
                mdb,
                pdb,
                rdb,
                cache,
                # it is critical for the release ownership analysis that we have *all* releases
                # between time_from and time_to
                # if releases_in_time_range was specified by the caller, there is no
                # guarantee
                releases_in_time_range=releases_in_time_range if internal_releases else None,
                metrics=metrics.commits if metrics is not None else None,
                refetcher=refetcher,
            ),
            ReleaseToPullRequestMapper._fetch_repository_first_commit_dates(
                coerce_logical_repos(missing_repos).keys(), account, meta_ids, mdb, pdb, cache,
            ),
            description="mine_releases/commits",
        )

        unfiltered_precomputed_facts_keys = merge_to_str(
            np.fromiter(
                (i for i, _ in unfiltered_precomputed_facts),
                int,
                len(unfiltered_precomputed_facts),
            ),
            np.array([r for _, r in unfiltered_precomputed_facts], dtype="S"),
        )

        required_release_ids = np.sort(
            merge_to_str(
                releases_in_time_range[Release.node_id.name],
                releases_in_time_range[Release.repository_full_name.name].astype("S"),
            ),
        )
        release_ids = merge_to_str(
            releases[Release.node_id.name],
            release_repos := releases[Release.repository_full_name.name].astype("S"),
        )
        # TODO(vmarkovtsev): optimize this
        precomputed_mask = np.in1d(release_ids, unfiltered_precomputed_facts_keys)
        out_of_interest_mask = np.in1d(release_ids, required_release_ids, invert=True)
        release_relevant = ~(precomputed_mask | out_of_interest_mask)
        del required_release_ids, release_ids, precomputed_mask, out_of_interest_mask

        all_hashes = []
        repo_order = np.argsort(release_repos, kind="stable")
        release_hashes = releases[Release.sha.name]
        release_timestamps = releases[Release.published_at.name]
        pos = 0
        alloc = make_mi_heap_allocator_capsule()
        for repo, repo_release_count in zip(
            *np.unique(release_repos[repo_order], return_counts=True),
        ):
            repo = repo.decode()
            repo_indexes = repo_order[pos : pos + repo_release_count]
            pos += repo_release_count
            _, (hashes, vertexes, edges) = dags[drop_logical_repo(repo)]
            if len(hashes) == 0:
                log.error("%s has an empty commit DAG, skipped from mining releases", repo)
                continue
            repo_release_hashes = release_hashes[repo_indexes]
            repo_release_timestamps = release_timestamps[repo_indexes]
            ownership = mark_dag_access(hashes, vertexes, edges, repo_release_hashes, True, alloc)
            parents = mark_dag_parents(
                hashes,
                vertexes,
                edges,
                repo_release_hashes,
                repo_release_timestamps,
                ownership,
                alloc,
            )
            if len(relevant := np.flatnonzero(release_relevant[repo_indexes])) == 0:
                continue
            if len(removed := np.flatnonzero(np.in1d(ownership, relevant, invert=True))) > 0:
                hashes = np.delete(hashes, removed)
                ownership = np.delete(ownership, removed)

            def on_missing(missing: np.ndarray) -> None:
                if len(
                    really_missing := np.flatnonzero(
                        np.in1d(missing, relevant, assume_unique=True),
                    ),
                ):
                    log.warning(
                        "%s has %d / %d releases with 0 commits",
                        repo,
                        len(really_missing),
                        repo_release_count,
                    )
                    log.debug("%s", releases.take(repo_indexes))
                    if metrics is not None:
                        metrics.empty_releases[repo] = len(really_missing)

            grouped_owned_hashes = group_hashes_by_ownership(
                ownership, hashes, repo_release_count, on_missing,
            )
            all_hashes.append(hashes)
            repo_releases_analyzed[repo] = (
                releases.take(repo_indexes[relevant]),
                grouped_owned_hashes[relevant],
                parents[relevant],
                releases[Release.published_at.name][repo_indexes],
            )
        del pos
        all_hashes = np.concatenate(all_hashes) if all_hashes else []
        with sentry_sdk.start_span(
            op="mine_releases/fetch_commits", description=str(len(all_hashes)),
        ):
            commits_df = await _fetch_commits(all_hashes, meta_ids, mdb)
        log.info("Loaded %d commits", len(commits_df))
        commits_index = commits_df[NodeCommit.sha.name]
        commit_ids = commits_df[NodeCommit.node_id.name]
        commits_additions = commits_df[NodeCommit.additions.name]
        commits_deletions = commits_df[NodeCommit.deletions.name]
        add_nans = commits_additions != commits_additions
        del_nans = commits_deletions != commits_deletions
        if (nans := (add_nans & del_nans)).any():
            log.error("null commit additions/deletions for %s", commit_ids[nans])
            commits_additions[nans] = 0
            commits_deletions[nans] = 0
        if (add_nans & ~nans).any():
            log.error("null commit additions for %s", commit_ids[add_nans])
            commits_additions[add_nans] = 0
        if (del_nans & ~nans).any():
            log.error("null commit deletions for %s", commit_ids[del_nans])
            commits_deletions[del_nans] = 0
        commits_authors, commits_authors_nz = _null_to_zero_int(
            commits_df, NodeCommit.author_user_id.name,
        )

        tasks = [
            _load_prs_by_merge_commit_ids(commit_ids, repos, logical_settings, meta_ids, mdb),
            _load_rebased_prs(
                commit_ids,
                repos,
                releases.unique(Release.repository_node_id.name, unordered=True),
                logical_settings,
                account,
                meta_ids,
                mdb,
                pdb,
            ),
        ]
        if jira:
            query = await generate_jira_prs_query(
                [PullRequest.merge_commit_id.in_(commit_ids)],
                jira,
                meta_ids,
                mdb,
                cache,
                columns=[PullRequest.merge_commit_id],
            )
            query = query.with_statement_hint(f"Rows(pr repo #{len(commit_ids)})")
            tasks.append(read_sql_query(query, mdb, columns=[PullRequest.merge_commit_id]))
        (prs_df, prs_commit_ids), (rebased_prs_df, rebased_commit_ids), *rest = await gather(
            *tasks, op="mine_releases/fetch_pull_requests", description=str(len(commit_ids)),
        )
        if not rebased_prs_df.empty:
            prs_df = md.concat(prs_df, rebased_prs_df, ignore_index=True)
            prs_commit_ids = np.concatenate([prs_commit_ids, rebased_commit_ids])
            order = np.argsort(prs_commit_ids)
            prs_commit_ids = prs_commit_ids[order]
            prs_df = prs_df.take(order)
        if jira:
            filtered_prs_commit_ids = rest[0].unique(PullRequest.merge_commit_id.name)
        original_prs_commit_ids = prs_df[PullRequest.merge_commit_id.name]
        prs_authors, prs_authors_nz = _null_to_zero_int(prs_df, PullRequest.user_node_id.name)
        prs_node_ids = prs_df[PullRequest.node_id.name]
        if labels:
            new_pr_labels_coro = fetch_labels_to_filter(prs_node_ids, meta_ids, mdb)
        if with_extended_pr_details:
            new_pr_details_coro = _load_extended_pr_details(prs_node_ids, meta_ids, mdb)
        if with_jira != JIRAEntityToFetch.NOTHING:
            new_jira_entities_coro = PullRequestJiraMapper.load(
                prs_node_ids, with_jira, meta_ids, mdb,
            )
        prs_numbers = prs_df[PullRequest.number.name]
        prs_additions = prs_df[PullRequest.additions.name]
        prs_deletions = prs_df[PullRequest.deletions.name]
        if has_logical_prs:
            prs_commits = prs_df[PullRequest.commits.name]

    @sentry_span
    async def main_flow():
        data = []
        if repo_releases_analyzed:
            log.info("Processing %d repos", len(repo_releases_analyzed))

        for repo, (
            repo_releases,
            owned_hashes,
            parents,
            all_published_at,
        ) in repo_releases_analyzed.items():
            computed_release_info_by_commit = {}
            is_logical = is_logical_repo(repo)
            repo_data = []
            # iterate in the reversed order to correctly handle multiple releases at the same tag
            for i, (
                my_id,
                my_name,
                my_url,
                my_author,
                my_published_at,
                my_matched_by,
                my_commit,
            ) in enumerate(
                zip(
                    repo_releases[Release.node_id.name][::-1],
                    repo_releases[Release.name.name][::-1],
                    repo_releases[Release.url.name][::-1],
                    repo_releases[Release.author_node_id.name][::-1],
                    repo_releases[Release.published_at.name][::-1],  # no values
                    repo_releases[matched_by_column][::-1],
                    repo_releases[Release.sha.name][::-1],
                ),
                start=1,
            ):
                i = len(repo_releases) - i
                my_parents = parents[i]
                if (first_published_at := computed_release_info_by_commit.get(my_commit)) is None:
                    if len(commits_index) > 0:
                        found_indexes = searchsorted_inrange(commits_index, owned_hashes[i])
                        found_indexes = found_indexes[
                            commits_index[found_indexes] == owned_hashes[i]
                        ]
                    else:
                        found_indexes = np.array([], dtype=int)
                    my_commit_ids = commit_ids[found_indexes]
                    if len(prs_commit_ids):
                        if has_logical_prs:
                            repo_bytes = repo.encode()
                            my_commit_ids = merge_to_str(
                                my_commit_ids,
                                np.full(len(my_commit_ids), repo_bytes, f"S{len(repo_bytes)}"),
                            )
                        my_prs_indexes = searchsorted_inrange(prs_commit_ids, my_commit_ids)
                        if len(my_prs_indexes):
                            my_prs_indexes = my_prs_indexes[
                                prs_commit_ids[my_prs_indexes] == my_commit_ids
                            ]
                            my_prs_indexes = my_prs_indexes[
                                np.argsort(prs_numbers[my_prs_indexes])
                            ]
                    else:
                        my_prs_indexes = np.array([], dtype=int)
                    if jira and not len(
                        np.intersect1d(
                            filtered_prs_commit_ids,
                            original_prs_commit_ids[my_prs_indexes],
                            assume_unique=True,
                        ),
                    ):
                        continue
                    my_prs_authors = prs_authors[my_prs_indexes]
                    if is_logical:
                        commits_count = prs_commits[my_prs_indexes].sum()
                        my_additions = prs_additions[my_prs_indexes].sum()
                        my_deletions = prs_deletions[my_prs_indexes].sum()
                        # not exactly correct, but we more care about the performance
                        my_commit_authors = my_prs_authors
                    else:
                        commits_count = len(found_indexes)
                        my_additions = commits_additions[found_indexes].sum()
                        my_deletions = commits_deletions[found_indexes].sum()
                        my_commit_authors = commits_authors[found_indexes]
                    mentioned_authors.update(unordered_unique(my_prs_authors[my_prs_authors > 0]))
                    my_prs = dict(
                        zip(
                            ["prs_" + c.name for c in released_prs_columns(PullRequest)],
                            [
                                prs_node_ids[my_prs_indexes],
                                prs_numbers[my_prs_indexes],
                                prs_additions[my_prs_indexes],
                                prs_deletions[my_prs_indexes],
                                my_prs_authors,
                            ],
                        ),
                    )
                    # must sort commit authors
                    my_commit_authors = np.unique(my_commit_authors[my_commit_authors > 0])
                    mentioned_authors.update(my_commit_authors)
                    if len(my_parents):
                        my_age = my_published_at - all_published_at[my_parents[0]]
                    else:
                        my_age = my_published_at - np.datetime64(
                            first_commit_dates[drop_logical_repo(repo)].replace(tzinfo=None), "us",
                        )
                    if my_author is not None:
                        mentioned_authors.add(my_author)
                    computed_release_info_by_commit[my_commit] = my_published_at
                else:
                    my_additions = my_deletions = commits_count = 0
                    my_commit_authors = commits_authors[:0]
                    my_prs = dict(
                        zip(
                            ["prs_" + c.name for c in released_prs_columns(PullRequest)],
                            [
                                prs_node_ids[:0],
                                prs_numbers[:0],
                                prs_additions[:0],
                                prs_deletions[:0],
                                prs_authors[:0],
                            ],
                        ),
                    )
                    my_age = my_published_at - first_published_at
                repo_data.append(
                    ReleaseFacts.from_fields(
                        published=my_published_at,
                        publisher=my_author,
                        matched_by=ReleaseMatch(my_matched_by),
                        age=my_age,
                        additions=my_additions,
                        deletions=my_deletions,
                        commits_count=commits_count,
                        commit_authors=my_commit_authors,
                        **my_prs,
                        node_id=my_id,
                        name=my_name,
                        url=my_url,
                        sha=my_commit,
                        repository_full_name=repo,
                    ),
                )
                if (len(data) + len(repo_data)) % 500 == 0:
                    await asyncio.sleep(0)
            data.extend(reversed(repo_data))
        if data:
            await defer(
                store_precomputed_release_facts(
                    data, default_branches, release_settings, account, pdb,
                ),
                "store_precomputed_release_facts(%d)" % len(data),
            )
        log.info("mined %d new releases", len(data))
        return data

    tasks = [
        precomputed_pr_labels_task,
        precomputed_pr_details_task,
        new_pr_labels_coro,
        new_pr_details_coro,
        precomputed_jira_entities_task,
        new_jira_entities_coro,
        deployments_task,
        main_flow(),
    ]
    if with_avatars:
        all_authors = np.unique(
            np.concatenate(
                [
                    commits_authors[commits_authors_nz],
                    prs_authors[prs_authors_nz],
                    mentioned_authors,
                ],
                dtype=int,
                casting="unsafe",
            ),
        )
        user_node_to_login = prefixer.user_node_to_login
        all_author_logins = []
        missing_nodes = []
        if len(all_authors) and all_authors[0] == 0:
            all_authors = all_authors[1:]
        for u in all_authors:
            try:
                all_author_logins.append(user_node_to_login[u])
            except KeyError:
                missing_nodes.append(u)
        if missing_nodes:
            log.error(
                "Missing user node in metadata for account %d / %s: %s",
                account,
                meta_ids,
                missing_nodes,
            )
        all_authors = all_author_logins
        tasks.insert(
            0, mine_user_avatars(UserAvatarKeys.NODE, meta_ids, mdb, cache, logins=all_authors),
        )
    else:
        tasks.insert(0, None)
    mentioned_authors = set(mentioned_authors)
    (
        avatars_result,
        precomputed_labels_result,
        precomputed_pr_extended_details_result,
        new_labels_result,
        new_pr_extended_details_result,
        _,
        new_pr_jira_map,
        _,
        mined_releases,
    ) = await gather(*tasks, op="mine missing releases")
    if with_jira != JIRAEntityToFetch.NOTHING:
        release_jira_map = (
            precomputed_jira_entities_task.result()
            | _convert_pr_jira_map_to_release_jira_map(
                ((f.node_id, f.repository_full_name) for f in mined_releases),
                *_take_pr_ids_from_precomputed_release_facts(mined_releases),
                new_pr_jira_map,
                with_jira,
            )
        )
    else:
        release_jira_map = {}
    result.extend(mined_releases)
    if with_avatars:
        avatars = [p for p in avatars_result if p[0] in mentioned_authors]
    else:
        avatars = list(mentioned_authors)
    if participants:
        result = _filter_by_participants(result, participants)
    if labels:
        if precomputed_labels_result is None:
            labels_result = new_labels_result
        elif new_labels_result is None:
            labels_result = precomputed_labels_result
        else:
            labels_result = md.concat(precomputed_labels_result, new_labels_result)
        result = _filter_by_labels(result, labels_result, labels)
    if with_extended_pr_details:
        presorted = True
        if precomputed_pr_extended_details_result is None:
            pr_extended_details_result = new_pr_extended_details_result
        elif new_pr_extended_details_result is None:
            pr_extended_details_result = precomputed_pr_extended_details_result
        else:
            presorted = False
            pr_extended_details_result = md.concat(
                precomputed_pr_extended_details_result,
                new_pr_extended_details_result,
                ignore_index=True,
            )
        _enrich_release_facts_with_extended_pr_details(
            pr_extended_details_result, result, presorted,
        )
    with sentry_sdk.start_span(op="mine_releases/install jira to facts"):
        empty_jira = LoadedJIRAReleaseDetails.empty()
        for facts in result:
            facts.jira = release_jira_map.get(
                (facts.node_id, facts.repository_full_name), empty_jira,
            )
    with sentry_sdk.start_span(op="mine_releases/install deployments to facts"):
        if with_deployments:
            depmap, deployments = deployments_task.result()
            for facts in result:
                facts.deployments = depmap.get(facts.node_id)
        else:
            deployments = {}
    return (
        df_from_structs(result),
        avatars,
        {r: v.match for r, v in release_settings.native.items()},
        deployments,
        with_avatars,
        with_extended_pr_details,
        with_deployments,
        with_jira,
    )


def _set_count_metrics(releases: md.DataFrame, metrics: MineReleaseMetrics) -> None:
    metrics.count_by_tag = int((releases[ReleaseFacts.f.matched_by] == ReleaseMatch.tag).sum())
    metrics.count_by_branch = int(
        (releases[ReleaseFacts.f.matched_by] == ReleaseMatch.branch).sum(),
    )
    metrics.count_by_event = int(
        (releases[ReleaseFacts.f.matched_by] == ReleaseMatch.event).sum(),
    )


def _empty_mined_releases_df():
    df = {}
    for key, val in ReleaseFacts.Immutable.__annotations__.items():
        if isinstance(val, list):
            val = object
        df[key] = np.array([], dtype=val)
    for key, val in ReleaseFacts.Optional.__annotations__.items():
        if key == "jira":
            continue
        if not (
            isinstance(val, np.dtype) or isinstance(val, type) and issubclass(val, (bool, int))
        ):
            val = object
        df[key] = np.array([], dtype=val)
    for key in LoadedJIRAReleaseDetails.__dataclass_fields__:
        df[f"jira_{key}"] = np.array([], dtype=object)
    return md.DataFrame(df)


def _null_to_zero_int(df: md.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    vals = df[col]
    vals_z = is_null(vals)
    vals[vals_z] = 0
    df[col] = df[col].astype(int)
    vals_nz = ~vals_z
    return df[col], vals_nz


def group_hashes_by_ownership(
    ownership: np.ndarray,
    hashes: np.ndarray,
    groups: int,
    on_missing: Optional[Callable[[np.ndarray], None]],
) -> npt.NDArray[npt.NDArray[bytes]]:
    """Return owned commit hashes for each release according to the ownership analysis."""
    order = np.argsort(ownership)
    sorted_hashes = hashes[order]
    sorted_ownership = ownership[order]
    unique_owners, unique_owned_counts = np.unique(sorted_ownership, return_counts=True)
    if len(unique_owned_counts) == 0:
        grouped_owned_hashes = []
    else:
        grouped_owned_hashes = np.split(sorted_hashes, np.cumsum(unique_owned_counts)[:-1])
        if unique_owners[-1] == groups:
            grouped_owned_hashes = grouped_owned_hashes[:-1]
            unique_owners = unique_owners[:-1]
    # fill the gaps for releases with 0 owned commits
    if len(missing := np.setdiff1d(np.arange(groups), unique_owners, assume_unique=True)):
        if on_missing is not None:
            on_missing(missing)
        empty = np.array([], dtype="S40")
        for i in missing:
            grouped_owned_hashes.insert(i, empty)
    assert len(grouped_owned_hashes) == groups
    arr = np.empty(groups, dtype=object)
    arr[:] = grouped_owned_hashes
    return arr


def _build_mined_releases(
    releases: md.DataFrame,
    precomputed_facts: dict[tuple[int, str], ReleaseFacts],
    prefixer: Prefixer,
    with_avatars: bool,
) -> tuple[list[ReleaseFacts], Optional[np.ndarray], np.ndarray]:
    release_repos = releases[Release.repository_full_name.name].astype("S", copy=False)
    release_keys = merge_to_str(releases[Release.node_id.name], release_repos)
    precomputed_keys = merge_to_str(
        np.fromiter((i for i, _ in precomputed_facts), int, len(precomputed_facts)),
        np.fromiter(
            (r for _, r in precomputed_facts), release_repos.dtype, len(precomputed_facts),
        ),
    )
    del release_repos
    has_precomputed_facts = np.in1d(release_keys, precomputed_keys)
    _, unique_releases = np.unique(release_keys, return_index=True)
    mask = np.zeros(len(releases), dtype=bool)
    mask[unique_releases] = True
    mask &= has_precomputed_facts
    result = [
        precomputed_facts[(my_id, my_repo)].with_optional_fields(
            node_id=my_id,
            name=my_name or my_tag,
            repository_full_name=my_repo,
            url=my_url,
            sha=my_commit,
        )
        for my_id, my_name, my_tag, my_repo, my_url, my_commit in zip(
            releases[Release.node_id.name][mask],
            releases[Release.name.name][mask],
            releases[Release.tag.name][mask],
            releases[Release.repository_full_name.name][mask],
            releases[Release.url.name][mask],
            releases[Release.sha.name][mask],
        )
        # "gone" repositories, reposet-sync has not updated yet
        if prefixer.prefix_logical_repo(my_repo) is not None
    ]
    if not with_avatars:
        return result, None, has_precomputed_facts
    mentioned_authors = np.concatenate(
        [
            *(
                getattr(f, "prs_" + PullRequest.user_node_id.name)
                for f in precomputed_facts.values()
            ),
            *(f.commit_authors for f in precomputed_facts.values()),
            releases[Release.author_node_id.name],
        ],
        casting="unsafe",
    )
    mentioned_authors = np.unique(mentioned_authors.astype(int, copy=False))
    if len(mentioned_authors) and mentioned_authors[0] == 0:
        mentioned_authors = mentioned_authors[1:]
    return result, mentioned_authors, has_precomputed_facts


@sentry_span
def _filter_by_participants(
    releases: list[ReleaseFacts],
    participants: ReleaseParticipants,
) -> list[ReleaseFacts]:
    if not releases or not participants:
        return releases
    participants = {k: np.array(v, dtype=int) for k, v in participants.items()}
    if ReleaseParticipationKind.RELEASER in participants:
        missing_indexes = np.flatnonzero(
            np.in1d(
                ReleaseFacts.vectorize_field(releases, ReleaseFacts.f.publisher),
                participants[ReleaseParticipationKind.RELEASER],
                invert=True,
            ),
        )
    else:
        missing_indexes = np.arange(len(releases))
    for rpk, col in [
        (ReleaseParticipationKind.COMMIT_AUTHOR, "commit_authors"),
        (ReleaseParticipationKind.PR_AUTHOR, "prs_" + PullRequest.user_node_id.name),
    ]:
        if len(missing_indexes) == 0:
            break
        if rpk in participants:
            values = [releases[i][col] for i in missing_indexes]
            lengths = nested_lengths(values)
            values.append([-1])
            offsets = np.zeros(len(values), dtype=int)
            np.cumsum(lengths, out=offsets[1:])
            values = np.concatenate(values)
            passed = np.bitwise_or.reduceat(np.in1d(values, participants[rpk]), offsets)[:-1]
            passed[lengths == 0] = False
            missing_indexes = missing_indexes[~passed]
    mask = np.ones(len(releases), bool)
    mask[missing_indexes] = False
    return [releases[i] for i in np.flatnonzero(mask)]


@sentry_span
def _filter_by_labels(
    releases: list[ReleaseFacts],
    labels_df: md.DataFrame,
    labels_filter: LabelFilter,
) -> list[ReleaseFacts]:
    if not releases:
        return releases
    all_pr_node_ids, borders = ReleaseFacts.vectorize_field(
        releases, "prs_" + PullRequest.node_id.name,
    )
    left = find_left_prs_by_labels(
        all_pr_node_ids,
        labels_df.index.values,
        labels_df[PullRequestLabel.name.name],
        labels_filter,
    )
    if len(left) == 0 and labels_filter.include:
        return []
    passed_mask = np.in1d(all_pr_node_ids, left, assume_unique=True)
    # DEV5752: np.argmax(borders) is always the first index of len(all_pr_node_ids)

    if labels_filter.include:
        passed_release_indexes = np.flatnonzero(
            np.logical_or.reduceat(passed_mask, borders[: np.argmax(borders)]),
        )
        return [releases[i] for i in passed_release_indexes]
    # DEV-2962
    # all the releases pass, but we must hide unmatched PRs
    changed_release_indexes = np.flatnonzero(
        ~np.logical_and.reduceat(passed_mask, borders[: np.argmax(borders)]),
    )
    pr_released_prs_columns = released_prs_columns(PullRequest)
    for i in changed_release_indexes:
        mask = passed_mask[borders[i + 1] : borders[i]]
        prs_hidden_release = dict(releases[i])
        for col in pr_released_prs_columns:
            key = "prs_" + col.name
            prs_hidden_release[key] = prs_hidden_release[key][mask]
        releases[i] = ReleaseFacts.from_fields(**prs_hidden_release)
    return releases


@sentry_span
async def _filter_precomputed_release_facts_by_jira(
    precomputed_facts: dict[tuple[int, str], ReleaseFacts],
    jira: JIRAFilter,
    meta_ids: tuple[int, ...],
    mdb: Database,
    cache: Optional[aiomcache.Client],
) -> dict[tuple[int, str], ReleaseFacts]:
    assert jira
    pr_ids, lengths = _take_pr_ids_from_precomputed_release_facts(precomputed_facts.values())
    if pr_ids is None:
        return {}
    # we could run the following in parallel with the rest, but
    # "the rest" is a no-op in most of the cases thanks to preheating
    df = await PullRequestMiner.filter_jira(
        pr_ids, jira, meta_ids, mdb, cache, columns=[PullRequest.node_id],
    )
    release_ids = np.repeat([k[0] for k in precomputed_facts], lengths)
    release_repos = np.repeat([k[1] for k in precomputed_facts], lengths)
    release_keys = np.empty(len(release_ids), dtype=[("id", int), ("repo", release_repos.dtype)])
    release_keys["id"] = release_ids
    release_keys["repo"] = release_repos
    del release_ids
    del release_repos
    order = np.argsort(pr_ids)
    pr_ids = pr_ids[order]
    release_keys = release_keys[order]
    release_keys = np.unique(release_keys[np.searchsorted(pr_ids, df.index.values)])
    return {(tk := tuple(k)): precomputed_facts[tk] for k in release_keys}


@sentry_span
async def _load_jira_from_precomputed_release_facts(
    precomputed_facts: dict[tuple[int, str], ReleaseFacts],
    jira_entities: JIRAEntityToFetch,
    meta_ids: tuple[int, ...],
    mdb: Database,
) -> dict[tuple[int, str], LoadedJIRAReleaseDetails]:
    pr_ids, lengths = _take_pr_ids_from_precomputed_release_facts(precomputed_facts.values())
    if pr_ids is None:
        return {}
    pr_map = await PullRequestJiraMapper.load(pr_ids, jira_entities, meta_ids, mdb)
    return _convert_pr_jira_map_to_release_jira_map(
        precomputed_facts, pr_ids, lengths, pr_map, jira_entities,
    )


@sentry_span
def _convert_pr_jira_map_to_release_jira_map(
    keys: Optional[Iterable[tuple[int, str]]],
    pr_ids: Optional[npt.NDArray[int]],
    lengths: Optional[npt.NDArray[int]],
    pr_map: Mapping[int, LoadedJIRADetails],
    jira_entities: JIRAEntityToFetch,
) -> dict[tuple[int, str], LoadedJIRAReleaseDetails]:
    result = {}
    if pr_ids is None or len(pr_ids) == 0:
        return result
    attr_map = [
        (JIRAEntityToFetch.ISSUES, "ids"),
        (JIRAEntityToFetch.PROJECTS, "projects"),
        (JIRAEntityToFetch.PRIORITIES, "priorities"),
        (JIRAEntityToFetch.TYPES, "types"),
        (JIRAEntityToFetch.LABELS, "labels"),
    ]
    empty = LoadedJIRAReleaseDetails.empty()
    empty_ids = np.array([], dtype=object)
    pos = 0
    for key, prs_count in zip(keys, lengths):
        release_prs = pr_ids[pos : pos + prs_count]
        pos += prs_count
        release_jira_entities = LoadedJIRAReleaseDetails(*([] for _ in attr_map), None)
        for pr in release_prs:
            for flag, attr in attr_map:
                if jira_entities & flag:
                    try:
                        values = getattr(pr_map[pr], attr)
                    except KeyError:
                        if attr == "ids":
                            values = empty_ids
                        else:
                            continue
                    getattr(release_jira_entities, attr).append(values)
        dikt = {
            attr: unordered_unique(np.concatenate(attr_val))
            if (attr_val := getattr(release_jira_entities, attr))
            else getattr(empty, attr)
            for _, attr in attr_map[1:]
        }
        if ids_val := release_jira_entities.ids:
            dikt["ids"] = np.concatenate(ids_val)
            offsets = np.zeros(len(ids_val) + 1, dtype=np.uint32)
            np.cumsum(nested_lengths(ids_val), out=offsets[1:])
            dikt["pr_offsets"] = offsets
        else:
            dikt["ids"] = empty.ids
            dikt["pr_offsets"] = empty.pr_offsets
        result[key] = LoadedJIRAReleaseDetails(**dikt)
    return result


@sentry_span
def _take_pr_ids_from_precomputed_release_facts(
    precomputed_facts: Collection[ReleaseFacts],
) -> tuple[Optional[npt.NDArray[int]], Optional[npt.NDArray[int]]]:
    pr_ids = np.empty(len(precomputed_facts), dtype=object)
    key = "prs_" + PullRequest.node_id.name
    for i, f in enumerate(precomputed_facts):
        pr_ids[i] = getattr(f, key)
    if len(pr_ids) == 0:
        return None, None
    lengths = nested_lengths(pr_ids)
    pr_ids = np.concatenate(pr_ids)
    if len(pr_ids) == 0:
        return None, None
    return pr_ids, lengths


@sentry_span
async def _load_prs_by_merge_commit_ids(
    commit_ids: Sequence[int],
    repos: Collection[str],
    logical_settings: LogicalRepositorySettings,
    meta_ids: tuple[int, ...],
    mdb: Database,
) -> tuple[md.DataFrame, npt.NDArray[int]]:
    return await _load_prs_by_ids(
        commit_ids,
        lambda model, ids: model.merge_commit_id.in_(ids),
        repos,
        logical_settings,
        True,
        meta_ids,
        mdb,
    )


@sentry_span
async def _load_prs_by_ids(
    ids: Sequence[int],
    filter_builder: Callable[[Any, Sequence[int]], BinaryExpression],
    repos: Collection[str],
    logical_settings: LogicalRepositorySettings,
    logical_sort: bool,
    meta_ids: tuple[int, ...],
    mdb: Database,
) -> tuple[md.DataFrame, npt.NDArray[int]]:
    log = logging.getLogger(f"{metadata.__package__}._load_prs_by_ids")
    has_logical_prs = logical_settings.has_logical_prs()
    model = PullRequest if has_logical_prs else NodePullRequest
    columns = [model.merge_commit_id, *released_prs_columns(model)]
    if has_logical_prs:
        columns.extend((PullRequest.title, PullRequest.repository_full_name, PullRequest.commits))
    has_prs_by_label = logical_settings.has_prs_by_label()
    # we can have 7,600,000 commit ids here and the DB breaks
    batch_size = 100_000
    tasks = []
    while len(ids):
        tasks.append(
            read_sql_query(
                select(*columns)
                .where(
                    model.acc_id.in_(meta_ids),
                    filter_builder(model, ids[:batch_size]),
                )
                .order_by(model.merge_commit_id),
                mdb,
                columns,
            ),
        )
        if has_prs_by_label:
            tasks.append(
                read_sql_query(
                    select(
                        PullRequestLabel.pull_request_node_id,
                        func.lower(PullRequestLabel.name).label(PullRequestLabel.name.name),
                    )
                    .select_from(
                        join(
                            NodePullRequest,
                            PullRequestLabel,
                            and_(
                                NodePullRequest.acc_id == PullRequestLabel.acc_id,
                                NodePullRequest.node_id == PullRequestLabel.pull_request_node_id,
                            ),
                        ),
                    )
                    .where(
                        NodePullRequest.acc_id.in_(meta_ids),
                        filter_builder(NodePullRequest, ids[:batch_size]),
                    )
                    .order_by(PullRequestLabel.pull_request_node_id),
                    mdb,
                    [PullRequestLabel.pull_request_node_id, PullRequestLabel.name],
                    index=PullRequestLabel.pull_request_node_id.name,
                ),
            )
        ids = ids[batch_size:]
    if tasks:
        dfs = await gather(*tasks)
        if has_prs_by_label:
            df_prs = dfs[::2]
            df_labels = dfs[1::2]
            df_labels = md.concat(*df_labels, copy=False)
        else:
            df_prs = dfs
            df_labels = None
        df_prs = md.concat(*df_prs, copy=False, ignore_index=True)
        original_prs_count = len(df_prs)
        df_prs = split_logical_prs(
            df_prs, df_labels, repos, logical_settings, reset_index=False, reindex=False,
        )
        if has_logical_prs:
            log.info("Logical PRs: %d -> %d", original_prs_count, len(df_prs))
        prs_commit_ids = df_prs[PullRequest.merge_commit_id.name]
        prs_commit_ids[is_null(prs_commit_ids)] = 0
        prs_commit_ids = prs_commit_ids.astype(int, copy=False)
        if has_logical_prs and logical_sort:
            prs_commit_ids = merge_to_str(
                prs_commit_ids,
                df_prs[PullRequest.repository_full_name.name].astype("S"),
            )
            order = np.argsort(prs_commit_ids)
            prs_commit_ids = prs_commit_ids[order]
            df_prs.take(order, inplace=True)
    else:
        df_prs = md.DataFrame(columns=[c.name for c in columns])
        prs_commit_ids = np.array([], dtype=int)
    return df_prs, prs_commit_ids


@sentry_span
async def _load_rebased_prs(
    commit_ids: npt.NDArray[int],
    repository_names: Collection[str],
    repository_ids: Collection[int],
    logical_settings: LogicalRepositorySettings,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
) -> tuple[md.DataFrame, npt.NDArray[int]]:
    rebased_df = await match_rebased_prs(
        repository_ids, account, meta_ids, mdb, pdb, commit_ids=commit_ids,
    )
    if rebased_df.empty:
        return md.DataFrame(), np.array([], dtype=int)
    rebased_pr_node_ids = rebased_df[GitHubRebasedPullRequest.pr_node_id.name]
    df, _ = await _load_prs_by_ids(
        rebased_pr_node_ids,
        lambda model, ids: model.node_id.in_(ids),
        repository_names,
        logical_settings,
        False,
        meta_ids,
        mdb,
    )
    order = np.argsort(rebased_pr_node_ids)
    rebased_pr_node_ids = rebased_pr_node_ids[order]
    rebased_commit_ids = rebased_df[GitHubRebasedPullRequest.matched_merge_commit_id.name]
    prs_commit_ids = rebased_commit_ids[
        order[np.searchsorted(rebased_pr_node_ids, df[PullRequest.node_id.name])]
    ]
    df[PullRequest.merge_commit_id.name] = prs_commit_ids
    if logical_settings.has_logical_prs():
        prs_commit_ids = merge_to_str(
            prs_commit_ids,
            df[PullRequest.repository_full_name.name].astype("S"),
        )
        order = np.argsort(prs_commit_ids)
        prs_commit_ids = prs_commit_ids[order]
        df.take(order, inplace=True)
    return df, prs_commit_ids


async def _load_extended_pr_details(
    prs: Collection[int],
    meta_ids: tuple[int, ...],
    mdb: Database,
) -> md.DataFrame:
    cols = [NodePullRequest.id, NodePullRequest.title, NodePullRequest.created_at]
    query = (
        select(*cols)
        .where(
            NodePullRequest.acc_id.in_(meta_ids),
            NodePullRequest.id.in_any_values(prs),
        )
        .order_by(NodePullRequest.id)
        .with_statement_hint(f"Rows({NodePullRequest.__tablename__} *VALUES* #{len(prs)})")
    )
    return await read_sql_query(query, mdb, cols)


@sentry_span
def _enrich_release_facts_with_extended_pr_details(
    pr_extended_details: md.DataFrame,
    facts: Iterable[ReleaseFacts],
    presorted: bool,
) -> None:
    pr_extended_node_ids = pr_extended_details[NodePullRequest.id.name]
    pr_extended_titles = pr_extended_details[NodePullRequest.title.name]
    pr_extended_createds = pr_extended_details[NodePullRequest.created_at.name]
    if not presorted:
        order = np.argsort(pr_extended_node_ids)
        pr_extended_node_ids = pr_extended_node_ids[order]
        pr_extended_titles = pr_extended_titles[order]
        pr_extended_createds = pr_extended_createds[order]
    pr_node_id_name = PullRequest.node_id.name
    pr_title_name = PullRequest.title.name
    pr_created_at_name = PullRequest.created_at.name
    for f in facts:
        node_ids = f["prs_" + pr_node_id_name]
        indexes = searchsorted_inrange(pr_extended_node_ids, node_ids)
        none_mask = pr_extended_node_ids[indexes] != node_ids
        f["prs_" + pr_title_name] = titles = pr_extended_titles[indexes]
        f["prs_" + pr_created_at_name] = createds = pr_extended_createds[indexes]
        titles[none_mask] = None
        createds[none_mask] = None


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=serialize_args,
    deserialize=deserialize_args,
    key=lambda names, release_settings, logical_settings, **_: (
        {k: sorted(v) for k, v in sorted(names.items())},
        release_settings,
        logical_settings,
    ),
)
async def mine_releases_by_name(
    names: dict[str, Iterable[str]],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> tuple[md.DataFrame, Collection[tuple[int, str]], dict[str, Deployment]]:
    """Collect details about each release specified by the mapping from repository names to \
    release names."""
    log = logging.getLogger("%s.mine_releases_by_name" % metadata.__package__)
    names = {k: set(v) for k, v in names.items()}
    releases, _, branches, default_branches = await _load_releases_by_name(
        names,
        log,
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
    if releases.empty:
        return _empty_mined_releases_df(), [], {}
    release_settings = release_settings.select(
        releases.unique(Release.repository_full_name.name, unordered=True),
    )
    tag_or_branch = [
        k for k, v in release_settings.native.items() if v.match == ReleaseMatch.tag_or_branch
    ]
    if not tag_or_branch:
        tasks = [
            mine_releases_by_ids(
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
                with_avatars=True,
                with_extended_pr_details=True,
                with_jira=JIRAEntityToFetch.ISSUES,
            ),
            _load_release_deployments(
                releases,
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
            ),
        ]
    else:
        tasks = [
            mine_releases_by_ids(
                tag_releases := releases.take(
                    (releases[matched_by_column] == ReleaseMatch.tag)
                    & (releases.isin(Release.repository_full_name.name, tag_or_branch)),
                ),
                branches,
                default_branches,
                tag_settings := ReleaseLoader.disambiguate_release_settings(
                    release_settings.select(tag_or_branch),
                    {r: ReleaseMatch.tag for r in tag_or_branch},
                ),
                logical_settings,
                prefixer,
                account,
                meta_ids,
                mdb,
                pdb,
                rdb,
                cache,
                with_avatars=True,
                with_extended_pr_details=True,
                with_jira=JIRAEntityToFetch.ISSUES,
            ),
            mine_releases_by_ids(
                branch_releases := releases.take(
                    (releases[matched_by_column] == ReleaseMatch.branch)
                    & (releases.isin(Release.repository_full_name.name, tag_or_branch)),
                ),
                branches,
                default_branches,
                branch_settings := ReleaseLoader.disambiguate_release_settings(
                    release_settings.select(tag_or_branch),
                    {r: ReleaseMatch.branch for r in tag_or_branch},
                ),
                logical_settings,
                prefixer,
                account,
                meta_ids,
                mdb,
                pdb,
                rdb,
                cache,
                with_avatars=True,
                with_extended_pr_details=True,
                with_jira=JIRAEntityToFetch.ISSUES,
            ),
            mine_releases_by_ids(
                remainder_releases := releases.take(
                    releases.isin(Release.repository_full_name.name, tag_or_branch, invert=True),
                ),
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
                with_avatars=True,
                with_extended_pr_details=True,
                with_jira=JIRAEntityToFetch.ISSUES,
            ),
            _load_release_deployments(
                tag_releases,
                default_branches,
                tag_settings,
                logical_settings,
                prefixer,
                account,
                meta_ids,
                mdb,
                pdb,
                rdb,
                cache,
            ),
            _load_release_deployments(
                branch_releases,
                default_branches,
                branch_settings,
                logical_settings,
                prefixer,
                account,
                meta_ids,
                mdb,
                pdb,
                rdb,
                cache,
            ),
            _load_release_deployments(
                remainder_releases,
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
            ),
        ]
    gathered = await gather(*tasks)
    if not tag_or_branch:
        (releases, avatars), (full_depmap, full_deployments) = gathered
    else:
        (
            (releases_tag, avatars_tag),
            (releases_branch, avatars_branch),
            (releases_remainder, avatars_remainder),
            *deployments,
        ) = gathered
        releases = md.concat(releases_tag, releases_branch, releases_remainder, ignore_index=True)
        avatars = set(chain(avatars_tag, avatars_branch, avatars_remainder))
        full_depmap = {}
        full_deployments = {}
        for depmap, deps in deployments:
            full_depmap.update(depmap)
            full_deployments.update(deps)
    if releases.empty:
        return _empty_mined_releases_df(), [], {}
    releases[ReleaseFacts.f.deployments] = [
        full_depmap.get(node_id) for node_id in releases[ReleaseFacts.f.node_id]
    ]
    return releases, avatars, full_deployments


@sentry_span
async def mine_releases_by_ids(
    releases: md.DataFrame,
    branches: md.DataFrame,
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
    *,
    with_avatars: bool,
    with_extended_pr_details: bool,
    with_jira: JIRAEntityToFetch,
) -> tuple[md.DataFrame, list[tuple[int, str]]] | md.DataFrame:
    """Collect details about releases in the DataFrame (`load_releases()`-like)."""
    result, avatars, _, _ = await mine_releases(
        releases.unique(Release.repository_full_name.name, unordered=True),
        {},
        branches,
        default_branches,
        None,
        None,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_settings,
        logical_settings,
        prefixer,
        account,
        meta_ids,
        mdb,
        pdb,
        rdb,
        cache,
        force_fresh=True,
        with_avatars=with_avatars,
        with_extended_pr_details=with_extended_pr_details,
        with_deployments=False,
        with_jira=with_jira,
        releases_in_time_range=releases,
    )
    if with_avatars:
        return result, avatars
    return result


async def _load_releases_by_name(
    names: dict[str, set[str]],
    log: logging.Logger,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> tuple[md.DataFrame, dict[str, dict[str, str]], md.DataFrame, dict[str, str]]:
    names = await _complete_commit_hashes(names, meta_ids, mdb)
    (branches, default_branches), releases = await gather(
        BranchMiner.load_branches(names, prefixer, account, meta_ids, mdb, pdb, cache),
        fetch_precomputed_releases_by_name(names, account, prefixer, pdb),
    )
    prenames = defaultdict(set)
    for repo, name in zip(
        releases[Release.repository_full_name.name], releases[Release.name.name],
    ):
        prenames[repo].add(name)
    missing = {}
    for repo, repo_names in names.items():
        if diff := repo_names.keys() - prenames.get(repo, set()):
            missing[repo] = diff
    if missing:
        now = datetime.now(timezone.utc)
        # There can be fresh new releases that are not in the pdb yet.
        match_groups, repos_count = group_repos_by_release_match(
            missing, default_branches, release_settings,
        )
        # event releases will be loaded in any case
        spans = await ReleaseLoader.fetch_precomputed_release_match_spans(
            match_groups, account, pdb,
        )
        offset = timedelta(hours=2)
        max_offset = timedelta(days=5 * 365)
        for repo in missing:
            try:
                have_not_precomputed = False
                for span_start, span_end in spans[repo].values():
                    if now - span_start < max_offset:
                        have_not_precomputed = True
                        offset = max_offset
                        break
                    else:
                        offset = max(offset, now - span_end)
                if have_not_precomputed:
                    break
            except KeyError:
                offset = max_offset
                break
        new_releases, _ = await ReleaseLoader.load_releases(
            missing,
            branches,
            default_branches,
            now - offset,
            now,
            release_settings,
            logical_settings,
            prefixer,
            account,
            meta_ids,
            mdb,
            pdb,
            rdb,
            cache,
            force_fresh=True,
        )
        new_releases_index = defaultdict(dict)
        for i, (repo, name) in enumerate(
            zip(
                new_releases[Release.repository_full_name.name],
                new_releases[Release.name.name],
            ),
        ):
            new_releases_index[repo][name] = i
        matching_indexes = []
        still_missing = defaultdict(list)
        for repo, repo_names in missing.items():
            for name in repo_names:
                try:
                    matching_indexes.append(new_releases_index[repo][name])
                except KeyError:
                    still_missing[repo].append(name)
        if matching_indexes:
            releases = md.concat(releases, new_releases.take(matching_indexes), ignore_index=True)
            releases.sort_values(
                Release.published_at.name, inplace=True, ascending=False, ignore_index=True,
            )
            if match_groups[ReleaseMatch.event]:
                # we could load them twice
                releases.drop_duplicates(
                    subset=[Release.node_id.name, Release.repository_full_name.name], inplace=True,
                )
        if still_missing:
            log.warning("Some releases were not found: %s", still_missing)
    return releases, names, branches, default_branches


commit_prefix_re = re.compile(r"[a-f0-9]{7}")


@sentry_span
async def _complete_commit_hashes(
    names: dict[str, set[str]],
    meta_ids: tuple[int, ...],
    mdb: Database,
) -> dict[str, dict[str, str]]:
    candidates = defaultdict(list)
    for repo, strs in names.items():
        for name in strs:
            if commit_prefix_re.fullmatch(name):
                candidates[repo].append(name)
    if not candidates:
        return {repo: {s: s for s in strs} for repo, strs in names.items()}
    queries = [
        select(PushCommit.repository_full_name, PushCommit.sha).where(
            PushCommit.acc_id.in_(meta_ids),
            PushCommit.repository_full_name == repo,
            func.substr(PushCommit.sha, 1, 7).in_(prefixes),
        )
        for repo, prefixes in candidates.items()
    ]
    if mdb.url.dialect == "sqlite" and len(queries) == 1:
        query = queries[0]
    else:
        query = union_all(*queries)
    rows = await mdb.fetch_all(query)
    renames = defaultdict(dict)
    renames_reversed = defaultdict(set)
    for row in rows:
        repo = row[PushCommit.repository_full_name.name]
        sha = row[PushCommit.sha.name]
        prefix = sha[:7]
        renames[repo][sha] = prefix
        renames_reversed[repo].add(prefix)
    for repo, strs in names.items():
        repo_renames = renames[repo]
        repo_renames_reversed = renames_reversed[repo]
        for name in strs:
            if name not in repo_renames_reversed:
                repo_renames[name] = name
    return renames


@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda borders, release_settings, logical_settings, **_: (
        {k: sorted(v) for k, v in sorted(borders.items())},
        release_settings,
        logical_settings,
    ),
)
async def diff_releases(
    borders: dict[str, list[tuple[str, str]]],
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    prefixer: Prefixer,
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> tuple[dict[str, list[tuple[str, str, md.DataFrame]]], list[tuple[str, str]]]:
    """Collect details about inner releases between the given boundaries for each repo."""
    log = logging.getLogger("%s.diff_releases" % metadata.__package__)
    names = defaultdict(set)
    for repo, pairs in borders.items():
        for old, new in pairs:
            names[repo].update((old, new))
    border_releases, names, branches, default_branches = await _load_releases_by_name(
        names,
        log,
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
    if border_releases.empty:
        return {}, []
    repos = border_releases.unique(Release.repository_full_name.name, unordered=True)
    time_from = (
        border_releases[Release.published_at.name].min().item().replace(tzinfo=timezone.utc)
    )
    time_to = (
        (border_releases[Release.published_at.name].max() + np.timedelta64(1, "s"))
        .item()
        .replace(tzinfo=timezone.utc)
    )

    async def fetch_dags():
        nonlocal border_releases
        dags = await fetch_precomputed_commit_history_dags(repos, account, pdb, cache)
        return await fetch_repository_commits(
            dags,
            border_releases,
            RELEASE_FETCH_COMMITS_COLUMNS,
            True,
            account,
            meta_ids,
            mdb,
            pdb,
            cache,
        )

    (releases, avatars, _, _), dags = await gather(
        mine_releases(
            repos,
            {},
            branches,
            default_branches,
            time_from,
            time_to,
            LabelFilter.empty(),
            JIRAFilter.empty(),
            release_settings,
            logical_settings,
            prefixer,
            account,
            meta_ids,
            mdb,
            pdb,
            rdb,
            cache,
            force_fresh=True,
            with_extended_pr_details=True,
            with_jira=JIRAEntityToFetch.ISSUES,
        ),
        fetch_dags(),
        op="mine_releases + dags",
    )
    del border_releases
    if releases.empty:
        return {}, []
    order = np.argsort(releases[ReleaseFacts.f.repository_full_name], kind="stable")
    unique_repos, repo_group_counts = np.unique(
        releases[ReleaseFacts.f.repository_full_name][order], return_counts=True,
    )
    result = {}
    release_name_col = releases[ReleaseFacts.f.name]
    sha_col = releases[ReleaseFacts.f.sha]
    pos = 0
    alloc = make_mi_heap_allocator_capsule()
    for repo, repo_group_count in zip(unique_repos, repo_group_counts):
        repo_names = {v: k for k, v in names[repo].items()}
        pairs = borders[repo]
        result[repo] = repo_result = []
        indexes = order[pos : pos + repo_group_count][::-1]
        pos += repo_group_count
        release_name_index = {name: i for i, name in enumerate(release_name_col[indexes])}
        for old, new in pairs:
            try:
                start = release_name_index[repo_names[old]]
                finish = release_name_index[repo_names[new]]
            except KeyError:
                log.warning("Release pair %s, %s was not found for %s", old, new, repo)
                continue
            if start > finish:
                log.warning("Release pair old %s is later than new %s for %s", old, new, repo)
                continue
            start_sha, finish_sha = (sha_col[indexes[x]] for x in (start, finish))
            hashes, _, _ = extract_subdag(*dags[repo][1], np.array([finish_sha]), alloc)
            if hashes[searchsorted_inrange(hashes, np.array([start_sha]))] == start_sha:
                pair_indexes = indexes[start + 1 : finish + 1]
                shas = sha_col[pair_indexes]
                found = hashes[searchsorted_inrange(hashes, shas)] == shas
                diff = pair_indexes[found]
                repo_result.append((old, new, releases.take(diff)))
            else:
                log.warning(
                    "Release pair's old %s is not in the sub-DAG of %s for %s", old, new, repo,
                )
    return result, avatars if any(any(len(d) for _, _, d in v) for v in result.values()) else {}


@sentry_span
async def _load_release_deployments(
    releases_in_time_range: md.DataFrame,
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
) -> tuple[dict[int, np.ndarray], dict[str, Deployment]]:
    if releases_in_time_range.empty:
        return {}, {}
    ghrd = GitHubReleaseDeployment
    reverse_settings = reverse_release_settings(
        releases_in_time_range.unique(Release.repository_full_name.name, unordered=True),
        default_branches,
        release_settings,
    )
    release_ids = releases_in_time_range[Release.node_id.name]
    repo_names = releases_in_time_range[Release.repository_full_name.name].astype("U", copy=False)
    cols = [ghrd.deployment_name, ghrd.release_id]
    reverse_items = list(reverse_settings.items())

    def release_id_in(ids):
        if len(ids) < 100:
            return ghrd.release_id.in_(ids)
        return ghrd.release_id.in_any_values(ids)

    batch_size = 10
    depmaps = await gather(
        *(
            read_sql_query(
                union_all(
                    *(
                        select(*cols).where(
                            ghrd.acc_id == account,
                            ghrd.release_match == compose_release_match(m, v),
                            ghrd.repository_full_name.in_(repos_group),
                            release_id_in(
                                release_ids[
                                    in1d_str(
                                        repo_names, np.array(repos_group, dtype=repo_names.dtype),
                                    )
                                ],
                            ),
                        )
                        for (m, v), repos_group in reverse_items[
                            group * batch_size : (group + 1) * batch_size
                        ]
                    ),
                ),
                pdb,
                cols,
            )
            for group in range((len(reverse_items) + batch_size - 1) // batch_size)
        ),
    )
    depmap = md.concat(*depmaps, ignore_index=True)
    release_ids = depmap[ghrd.release_id.name]
    dep_names = depmap[ghrd.deployment_name.name]
    order = np.argsort(release_ids)
    dep_names = dep_names[order]
    unique_release_ids, counts = np.unique(release_ids, return_counts=True)
    dep_name_groups = np.split(dep_names, np.cumsum(counts[:-1]))
    depmap = dict(zip(unique_release_ids, dep_name_groups))
    deployments = await load_included_deployments(
        np.unique(dep_names), logical_settings, prefixer, account, meta_ids, mdb, rdb, cache,
    )
    return depmap, deployments


def discover_first_outlier_releases(releases: md.DataFrame, threshold_factor=100) -> md.DataFrame:
    """
    Apply heuristics to find first releases that should be hidden from the metrics.

    :param releases: Releases in the format of `mine_releases()`.
    :param threshold_factor: We consider the first release as an outlier if it's age is bigger \
                             than the median age of the others multiplied by this number.
    :return: load_releases()-like dataframe with found outlier releases.
    """
    oldest_release_by_repo = {}
    indexes_by_repo = defaultdict(list)
    for i, (repo, ts) in enumerate(
        zip(
            releases[ReleaseFacts.f.repository_full_name],
            releases[ReleaseFacts.f.published],
        ),
    ):
        indexes_by_repo[repo].append(i)
        try:
            _, min_ts = oldest_release_by_repo[repo]
        except KeyError:
            min_ts = ts
        if min_ts >= ts:
            oldest_release_by_repo[repo] = i, ts
    outlier_releases = {k: [] for k in releases.columns}
    age_col = releases[ReleaseFacts.f.age]
    must_update_values = []
    for repo, (i, _) in oldest_release_by_repo.items():
        ages = np.array([age_col[j] for j in indexes_by_repo[repo] if j != i])
        if len(ages) < 2 or age_col[i] < np.median(ages) * threshold_factor:
            continue
        release = releases.iloc[i]
        must_update = False
        for key, val in release.items():
            if key.startswith("prs_"):
                if val is not None:
                    must_update |= len(val) > 0
                    val = val[:0]
            outlier_releases[key].append(val)
        must_update_values.append(must_update)
    for key, val in outlier_releases.items():
        if (dtype := releases[key].dtype) == object:
            arena = np.empty(len(val), dtype=object)
            arena[:] = val
            val = arena
        else:
            val = np.array(val, dtype=dtype)
        outlier_releases[key] = val
    df = md.DataFrame(outlier_releases)
    df["must_update"] = must_update_values
    return df


async def hide_first_releases(
    releases: md.DataFrame,
    default_branches: dict[str, str],
    release_settings: ReleaseSettings,
    account: int,
    pdb: Database,
) -> None:
    """
    Hide the specified releases from calculating the metrics.

    :param releases: First releases detected by `discover_first_outlier_releases()`.
    """
    if releases.empty:
        return
    log = logging.getLogger(f"{metadata.__package__}.hide_first_releases")
    log.info(
        "processing %d first releases: %s",
        len(releases),
        releases[ReleaseFacts.f.url].tolist(),
    )

    async def set_pr_release_ignored() -> None:
        ghdprf = GitHubDonePullRequestFacts
        format_version = ghdprf.__table__.columns[ghdprf.format_version.key].default.arg
        if pdb.url.dialect == "sqlite":
            extra_cols = [
                ghdprf.pr_created_at,
                ghdprf.number,
                ghdprf.reviewers,
                ghdprf.commenters,
                ghdprf.commit_authors,
                ghdprf.commit_committers,
                ghdprf.labels,
                ghdprf.activity_days,
                ghdprf.release_node_id,
                ghdprf.release_url,
                ghdprf.author,
                ghdprf.merger,
                ghdprf.releaser,
            ]
            byte_cond = sql_true()
        else:
            extra_cols = [ghdprf.pr_created_at, ghdprf.number]
            byte_cond = (
                func.get_byte(
                    ghdprf.data,
                    PullRequestFacts.dtype.fields[PullRequestFacts.f.release_ignored][1],
                )
                == 0
            )

        match_groups, _ = group_repos_by_release_match(
            releases.unique(ReleaseFacts.f.repository_full_name, unordered=True),
            default_branches,
            release_settings,
        )
        or_conditions, _ = match_groups_to_conditions(match_groups, ghdprf)
        release_repo_values = releases[ReleaseFacts.f.repository_full_name]
        release_node_id_values = releases[ReleaseFacts.f.node_id]

        queries = [
            select(
                ghdprf.pr_node_id,
                ghdprf.repository_full_name,
                ghdprf.release_match,
                ghdprf.data,
                *extra_cols,
            ).where(
                ghdprf.acc_id == account,
                ghdprf.format_version == format_version,
                ghdprf.release_match == release_match_cond[ghdprf.release_match.name],
                ghdprf.release_node_id.in_(
                    release_node_id_values[
                        np.in1d(
                            release_repo_values,
                            release_match_cond[ghdprf.repository_full_name.name],
                        )
                    ],
                ),
                byte_cond,
            )
            for release_match_cond in or_conditions
        ]
        if len(queries) == 1:
            query = queries[0]
        else:
            query = union_all(*queries)

        if not (rows := await pdb.fetch_all(query)):
            return

        log.info("setting %d PRs as release ignored", len(rows))
        updates = []
        now = datetime.now(timezone.utc)
        for row in rows:
            args = dict(PullRequestFacts(row[ghdprf.data.name]))
            args[PullRequestFacts.f.release_ignored] = True
            args[PullRequestFacts.f.released] = None
            updates.append(
                {
                    ghdprf.acc_id.name: account,
                    ghdprf.format_version.name: format_version,
                    ghdprf.release_match.name: row[ghdprf.release_match.name],
                    ghdprf.repository_full_name.name: row[ghdprf.repository_full_name.name],
                    ghdprf.pr_node_id.name: row[ghdprf.pr_node_id.name],
                    ghdprf.data.name: PullRequestFacts.from_fields(**args).data,
                    ghdprf.updated_at.name: now,
                    **{k.name: row[k.name] for k in extra_cols},
                },
            )
        sql = (await dialect_specific_insert(pdb))(ghdprf)
        sql = sql.on_conflict_do_update(
            index_elements=ghdprf.__table__.primary_key.columns,
            set_={
                col.name: getattr(sql.excluded, col.name)
                for col in (ghdprf.updated_at, ghdprf.data)
            },
        )
        if pdb.url.dialect == "sqlite":
            async with pdb.connection() as pdb_conn:
                async with pdb_conn.transaction():
                    await pdb_conn.execute_many(sql, updates)
        else:
            # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
            await pdb.execute_many(sql, updates)

    release_settings = ReleaseLoader.disambiguate_release_settings(
        release_settings,
        dict(
            zip(
                releases[ReleaseFacts.f.repository_full_name],
                releases[ReleaseFacts.f.matched_by],
            ),
        ),
    )

    with sentry_sdk.start_span(op="store_precomputed_done_facts/execute_many"):
        keys = releases.columns
        must_update_col = keys.index("must_update")
        await gather(
            store_precomputed_release_facts(
                [
                    ReleaseFacts.from_fields(**dict(zip(keys, row)))
                    for row in zip(*(releases[k] for k in keys))
                    if row[must_update_col]
                ],
                default_branches,
                release_settings,
                account,
                pdb,
                on_conflict_replace=True,
            ),
            set_pr_release_ignored(),
            op=f"override_first_releases/update({len(releases)}",
        )


async def override_first_releases(
    releases: md.DataFrame,
    default_branches: dict[str, str],
    release_settings: ReleaseSettings,
    account: int,
    pdb: Database,
    threshold_factor=100,
) -> int:
    """Exclude outlier first releases from calculating PR and release metrics."""
    first_releases = discover_first_outlier_releases(releases, threshold_factor)
    await hide_first_releases(first_releases, default_branches, release_settings, account, pdb)
    return len(first_releases)


async def _fetch_commits(hashes: npt.NDArray[bytes], meta_ids: Sequence[int], mdb: DatabaseLike):
    """Fetch from mdb all commits with the given hashes."""
    log = logging.getLogger(f"{__package__}._fetch_commits")

    columns = [
        NodeCommit.sha,
        NodeCommit.additions,
        NodeCommit.deletions,
        NodeCommit.author_user_id,
        NodeCommit.node_id,
    ]
    order_column = NodeCommit.sha

    log.info("fetching %d commits", len(hashes))

    query = (
        sa.select(*columns)
        .where(NodeCommit.acc_id.in_(meta_ids), NodeCommit.sha.in_any_values(hashes))
        .order_by(order_column)
        .with_statement_hint(f"Rows({NodeCommit.__tablename__} *VALUES* #{len(hashes)})")
        .with_statement_hint(f"Leading(*VALUES* {NodeCommit.__tablename__})")
    )

    return await read_sql_query(query, mdb, columns)
