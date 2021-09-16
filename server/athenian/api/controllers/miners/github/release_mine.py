import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging
import pickle
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import aiomcache
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, func, select, union_all

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached, CancelCache, short_term_exptime
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.commit import fetch_precomputed_commit_history_dags, \
    fetch_repository_commits, RELEASE_FETCH_COMMITS_COLUMNS
from athenian.api.controllers.miners.github.dag_accelerated import extract_subdag, \
    mark_dag_access, mark_dag_parents, searchsorted_inrange
from athenian.api.controllers.miners.github.precomputed_releases import \
    fetch_precomputed_releases_by_name, load_precomputed_release_facts, \
    store_precomputed_release_facts
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.controllers.miners.github.release_load import \
    group_repos_by_release_match, ReleaseLoader
from athenian.api.controllers.miners.github.release_match import \
    load_commit_dags, ReleaseToPullRequestMapper
from athenian.api.controllers.miners.github.released_pr import matched_by_column
from athenian.api.controllers.miners.github.user import mine_user_avatars, UserAvatarKeys
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.controllers.miners.types import released_prs_columns, ReleaseFacts, \
    ReleaseParticipants, ReleaseParticipationKind
from athenian.api.controllers.prefixer import Prefixer, PrefixerPromise
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting, ReleaseSettings
from athenian.api.db import add_pdb_hits, add_pdb_misses, ParallelDatabase
from athenian.api.defer import defer
from athenian.api.models.metadata.github import NodePullRequest, PullRequest, PullRequestLabel, \
    PushCommit, Release
from athenian.api.tracing import sentry_span


def _reject_releases_without_pr_titles(result: Tuple[List[Tuple[Dict[str, Any], ReleaseFacts]],
                                                     Union[List[Tuple[str, str]], np.ndarray],
                                                     Dict[str, ReleaseMatch]],
                                       with_pr_titles: bool = False,
                                       **_) -> Tuple[List[Tuple[Dict[str, Any], ReleaseFacts]],
                                                     Union[List[Tuple[str, str]], np.ndarray],
                                                     Dict[str, ReleaseMatch]]:
    if with_pr_titles and any(r[1]["prs_" + PullRequest.title.name] is None for r in result[0]):
        raise CancelCache()
    return result


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, participants, time_from, time_to, labels, jira, settings, **_: (
        ",".join(sorted(repos)),
        ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(participants.items())),
        time_from, time_to, labels, jira, settings),
    postprocess=_reject_releases_without_pr_titles,
)
async def mine_releases(repos: Iterable[str],
                        participants: ReleaseParticipants,
                        branches: pd.DataFrame,
                        default_branches: Dict[str, str],
                        time_from: datetime,
                        time_to: datetime,
                        labels: LabelFilter,
                        jira: JIRAFilter,
                        settings: ReleaseSettings,
                        prefixer: PrefixerPromise,
                        account: int,
                        meta_ids: Tuple[int, ...],
                        mdb: ParallelDatabase,
                        pdb: ParallelDatabase,
                        rdb: ParallelDatabase,
                        cache: Optional[aiomcache.Client],
                        force_fresh: bool = False,
                        with_avatars: bool = True,
                        with_pr_titles: bool = False,
                        ) -> Tuple[List[Tuple[Dict[str, Any], ReleaseFacts]],
                                   Union[List[Tuple[int, str]], List[int]],
                                   Dict[str, ReleaseMatch]]:
    """Collect details about each release published between `time_from` and `time_to` and \
    calculate various statistics.

    :param participants: Mapping from roles to node IDs.
    :param force_fresh: Ensure that we load the most up to date releases, no matter the state of \
                        the pdb is.
    :param with_avatars: Indicates whether to return the fetched user avatars or just an array of \
                         unique mentioned logins.
    :param with_pr_titles: Indicates whether released PR titles must be fetched.
    :return: 1. List of releases (general info, computed facts). \
             2. User avatars if `with_avatars` else *only newly mined* mentioned people nodes. \
             3. Release matched_by-s.
    """
    log = logging.getLogger("%s.mine_releases" % metadata.__package__)
    releases_in_time_range, matched_bys = await ReleaseLoader.load_releases(
        repos, branches, default_branches, time_from, time_to,
        settings, prefixer, account, meta_ids, mdb, pdb, rdb, cache, force_fresh=force_fresh)
    settings = ReleaseLoader.disambiguate_release_settings(settings, matched_bys)
    if releases_in_time_range.empty:
        return [], [], {r: v.match for r, v in settings.prefixed.items()}
    precomputed_facts = await load_precomputed_release_facts(
        releases_in_time_range, default_branches, settings, account, pdb)
    # uncomment this line to compute releases from scratch:
    # precomputed_facts = {}
    if with_pr_titles or labels:
        all_pr_node_ids = [
            f["prs_" + PullRequest.node_id.name] for f in precomputed_facts.values()
        ]
    add_pdb_hits(pdb, "release_facts", len(precomputed_facts))
    missed_releases_count = len(releases_in_time_range) - len(precomputed_facts)
    add_pdb_misses(pdb, "release_facts", missed_releases_count)
    unfiltered_precomputed_facts = precomputed_facts
    if jira:
        precomputed_facts = await _filter_precomputed_release_facts_by_jira(
            precomputed_facts, jira, meta_ids, mdb, cache)
    prefixer = await prefixer.load()
    result, mentioned_authors, has_precomputed_facts = _build_mined_releases(
        releases_in_time_range, precomputed_facts, prefixer, True)

    missing_repos = releases_in_time_range[Release.repository_full_name.name].take(
        np.where(~has_precomputed_facts)[0]).unique()
    commits_authors = prs_authors = []
    commits_authors_nz = prs_authors_nz = slice(0)
    repo_releases_analyzed = {}
    if missed_releases_count > 0:
        releases_in_time_range = releases_in_time_range.take(np.where(
            releases_in_time_range[Release.repository_full_name.name].isin(missing_repos).values,
        )[0])
        _, releases, _, _ = await ReleaseToPullRequestMapper._find_releases_for_matching_prs(
            missing_repos, branches, default_branches, time_from, time_to, False,
            settings, prefixer.as_promise(), account, meta_ids, mdb, pdb, rdb, cache,
            releases_in_time_range=releases_in_time_range)
        tasks = [
            load_commit_dags(releases, account, meta_ids, mdb, pdb, cache),
            ReleaseToPullRequestMapper._fetch_repository_first_commit_dates(
                missing_repos, account, meta_ids, mdb, pdb, cache),
        ]
        dags, first_commit_dates = await gather(*tasks, op="mine_releases/commits")

        all_hashes = []
        for repo, repo_releases in releases.groupby(Release.repository_full_name.name, sort=False):
            hashes, vertexes, edges = dags[repo]
            if len(hashes) == 0:
                log.error("%s has an empty commit DAG, skipped from mining releases", repo)
                continue
            release_hashes = repo_releases[Release.sha.name].values.astype("S40")
            release_timestamps = repo_releases[Release.published_at.name].values
            ownership = mark_dag_access(hashes, vertexes, edges, release_hashes)
            parents = mark_dag_parents(
                hashes, vertexes, edges, release_hashes, release_timestamps, ownership)
            precomputed_mask = \
                repo_releases[Release.node_id.name].isin(unfiltered_precomputed_facts).values
            out_of_range_mask = release_timestamps < np.array(time_from.replace(tzinfo=None),
                                                              dtype=release_timestamps.dtype)
            relevant = np.nonzero(~(precomputed_mask | out_of_range_mask))[0]
            if len(relevant) == 0:
                continue
            if len(removed := np.nonzero(np.in1d(ownership, relevant, invert=True))[0]) > 0:
                hashes = np.delete(hashes, removed)
                ownership = np.delete(ownership, removed)

            def on_missing(missing: np.ndarray) -> None:
                if len(really_missing := np.nonzero(np.in1d(
                        missing, relevant, assume_unique=True))[0]):
                    log.warning("%s has %d / %d releases with 0 commits",
                                repo, len(really_missing), len(repo_releases))
                    log.debug("%s", repo_releases.take(really_missing))

            grouped_owned_hashes = group_hashes_by_ownership(
                ownership, hashes, len(repo_releases), on_missing)
            all_hashes.append(hashes)
            repo_releases_analyzed[repo] = repo_releases, grouped_owned_hashes, parents
        commits_df_columns = [
            PushCommit.sha,
            PushCommit.additions,
            PushCommit.deletions,
            PushCommit.author_user_id,
            PushCommit.node_id,
        ]
        all_hashes = np.concatenate(all_hashes).astype("U40") if all_hashes else []
        with sentry_sdk.start_span(op="mine_releases/fetch_commits",
                                   description=str(len(all_hashes))):
            commits_df = await read_sql_query(
                select(commits_df_columns)
                .where(and_(PushCommit.sha.in_(all_hashes), PushCommit.acc_id.in_(meta_ids)))
                .order_by(PushCommit.sha),
                mdb, commits_df_columns, index=PushCommit.sha.name)
        commits_index = commits_df.index.values.astype("S40")
        commit_ids = commits_df[PushCommit.node_id.name].values
        commits_additions = commits_df[PushCommit.additions.name].values
        commits_deletions = commits_df[PushCommit.deletions.name].values
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
            commits_df, PushCommit.author_user_id.name)

        tasks = [_load_prs_by_merge_commit_ids(commit_ids, meta_ids, mdb)]
        if jira:
            query = await generate_jira_prs_query(
                [PullRequest.merge_commit_id.in_(commit_ids),
                 PullRequest.acc_id.in_(meta_ids)],
                jira, mdb, cache, columns=[PullRequest.merge_commit_id])
            tasks.append(mdb.fetch_all(query))
        prs_df, *rest = await gather(*tasks,
                                     op="mine_releases/fetch_pull_requests",
                                     description=str(len(commit_ids)))
        if jira:
            filtered_prs_commit_ids = np.unique(np.array([r[0] for r in rest[0]]))
        prs_commit_ids = prs_df[PullRequest.merge_commit_id.name].values
        prs_authors, prs_authors_nz = _null_to_zero_int(prs_df, PullRequest.user_node_id.name)
        prs_node_ids = prs_df[PullRequest.node_id.name].values
        if with_pr_titles or labels:
            all_pr_node_ids.append(prs_node_ids)
        prs_numbers = prs_df[PullRequest.number.name].values
        prs_additions = prs_df[PullRequest.additions.name].values
        prs_deletions = prs_df[PullRequest.deletions.name].values

    @sentry_span
    async def main_flow():
        data = []
        for repo, (repo_releases, owned_hashes, parents) in repo_releases_analyzed.items():
            computed_release_info_by_commit = {}
            for i, (my_id, my_name, my_tag, my_url, my_author, my_published_at,
                    my_matched_by, my_commit) in \
                    enumerate(zip(repo_releases[Release.node_id.name].values,
                                  repo_releases[Release.name.name].values,
                                  repo_releases[Release.tag.name].values,
                                  repo_releases[Release.url.name].values,
                                  repo_releases[Release.author_node_id.name].values,
                                  repo_releases[Release.published_at.name],  # no values
                                  repo_releases[matched_by_column].values,
                                  repo_releases[Release.sha.name].values)):
                if my_published_at < time_from or my_id in unfiltered_precomputed_facts:
                    continue
                dupe = computed_release_info_by_commit.get(my_commit)
                if dupe is None:
                    if len(commits_index) > 0:
                        found_indexes = searchsorted_inrange(commits_index, owned_hashes[i])
                        found_indexes = \
                            found_indexes[commits_index[found_indexes] == owned_hashes[i]]
                    else:
                        found_indexes = np.array([], dtype=int)
                    my_commit_ids = commit_ids[found_indexes]
                    if jira and not len(np.intersect1d(
                            filtered_prs_commit_ids, my_commit_ids, assume_unique=True)):
                        continue
                    if len(prs_commit_ids):
                        my_prs_indexes = searchsorted_inrange(prs_commit_ids, my_commit_ids)
                        if len(my_prs_indexes):
                            my_prs_indexes = \
                                my_prs_indexes[prs_commit_ids[my_prs_indexes] == my_commit_ids]
                    else:
                        my_prs_indexes = np.array([], dtype=int)
                    commits_count = len(found_indexes)
                    my_additions = commits_additions[found_indexes].sum()
                    my_deletions = commits_deletions[found_indexes].sum()
                    my_commit_authors = commits_authors[found_indexes]
                    my_prs_authors = prs_authors[my_prs_indexes]
                    mentioned_authors.update(np.unique(my_prs_authors[my_prs_authors > 0]))
                    my_prs = dict(zip(
                        ["prs_" + c.name for c in released_prs_columns],
                        [prs_node_ids[my_prs_indexes],
                         prs_numbers[my_prs_indexes],
                         prs_additions[my_prs_indexes],
                         prs_deletions[my_prs_indexes],
                         my_prs_authors]))

                    my_commit_authors = np.unique(my_commit_authors[my_commit_authors > 0])
                    mentioned_authors.update(my_commit_authors)
                    parent = parents[i]
                    if parent < len(repo_releases):
                        my_age = \
                            my_published_at - repo_releases[Release.published_at.name]._ixs(parent)
                    else:
                        my_age = my_published_at - first_commit_dates[repo]
                    if my_author is not None:
                        mentioned_authors.add(my_author)
                    computed_release_info_by_commit[my_commit] = (
                        my_age, my_additions, my_deletions, commits_count, my_prs,
                        my_commit_authors,
                    )
                else:  # dupe
                    (
                        my_age, my_additions, my_deletions, commits_count, my_prs,
                        my_commit_authors,
                    ) = dupe
                data.append(({Release.node_id.name: my_id,
                              Release.name.name: my_name or my_tag,
                              Release.repository_full_name.name:
                                  prefixer.repo_name_to_prefixed_name[repo],
                              Release.url.name: my_url,
                              Release.sha.name: my_commit},
                             ReleaseFacts.from_fields(published=my_published_at,
                                                      publisher=my_author,
                                                      matched_by=ReleaseMatch(my_matched_by),
                                                      age=my_age,
                                                      additions=my_additions,
                                                      deletions=my_deletions,
                                                      commits_count=commits_count,
                                                      commit_authors=my_commit_authors,
                                                      **my_prs,
                                                      repository_full_name=repo)))
                if len(data) % 500 == 0:
                    await asyncio.sleep(0)
        if data:
            await defer(
                store_precomputed_release_facts(
                    data, default_branches, settings, account, pdb),
                "store_precomputed_release_facts(%d)" % len(data))
        log.info("mined %d new releases", len(data))
        return data

    tasks = [main_flow()]
    if with_avatars:
        all_authors = np.unique(np.concatenate([
            commits_authors[commits_authors_nz],
            prs_authors[prs_authors_nz],
            mentioned_authors]))
        all_authors = [prefixer.user_node_to_login[u] for u in all_authors]
        tasks.insert(0, mine_user_avatars(all_authors, UserAvatarKeys.NODE, meta_ids, mdb, cache))
    if (with_pr_titles or labels) and all_pr_node_ids:
        all_pr_node_ids = np.concatenate(all_pr_node_ids)
    if with_pr_titles:
        tasks.insert(0, mdb.fetch_all(
            select([NodePullRequest.id, NodePullRequest.title])
            .where(and_(NodePullRequest.acc_id.in_(meta_ids),
                        NodePullRequest.id.in_any_values(all_pr_node_ids)))))
    if labels:
        tasks.insert(0, read_sql_query(
            select([PullRequestLabel.pull_request_node_id,
                    func.lower(PullRequestLabel.name).label(PullRequestLabel.name.name)])
            .where(and_(PullRequestLabel.acc_id.in_(meta_ids),
                        PullRequestLabel.pull_request_node_id.in_any_values(all_pr_node_ids))),
            mdb, [PullRequestLabel.pull_request_node_id, PullRequestLabel.name],
            index=PullRequestLabel.pull_request_node_id.name))
    mentioned_authors = set(mentioned_authors)
    *secondary, mined_releases = await gather(*tasks, op="mine missing releases")
    result.extend(mined_releases)
    if with_avatars:
        avatars = [p for p in secondary[-1] if p[0] in mentioned_authors]
    else:
        avatars = list(mentioned_authors)
    if participants:
        result = _filter_by_participants(result, participants)
    if labels:
        result = _filter_by_labels(result, secondary[0], labels)
    if with_pr_titles:
        pr_title_map = {row[0]: row[1] for row in secondary[bool(labels)]}
        for _, facts in result:
            facts["prs_" + PullRequest.title.name] = [
                pr_title_map.get(node) for node in facts["prs_" + PullRequest.node_id.name]
            ]
    return result, avatars, {r: v.match for r, v in settings.prefixed.items()}


def _release_facts_with_repository_full_name(facts: ReleaseFacts, repo: str) -> ReleaseFacts:
    facts.repository_full_name = repo
    return facts


def _null_to_zero_int(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray]:
    vals = df[col]
    vals_z = vals.isnull().values
    vals.values[vals_z] = 0
    df[col] = df[col].astype(int)
    vals_nz = ~vals_z
    return df[col].values, vals_nz


def group_hashes_by_ownership(ownership: np.ndarray,
                              hashes: np.ndarray,
                              groups: int,
                              on_missing: Optional[Callable[[np.ndarray], None]],
                              ) -> List[np.ndarray]:
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
    return grouped_owned_hashes


def _build_mined_releases(releases: pd.DataFrame,
                          precomputed_facts: Dict[int, ReleaseFacts],
                          prefixer: Prefixer,
                          with_avatars: bool,
                          ) -> Tuple[List[Tuple[Dict[str, Any], ReleaseFacts]],
                                     Optional[np.ndarray],
                                     np.ndarray]:
    has_precomputed_facts = releases[Release.node_id.name].isin(precomputed_facts).values
    _, unique_releases = np.unique(releases[Release.node_id.name].values, return_index=True)
    mask = np.zeros(len(releases), dtype=bool)
    mask[unique_releases] = True
    mask &= has_precomputed_facts
    result = [
        ({Release.node_id.name: my_id,
          Release.name.name: my_name or my_tag,
          Release.repository_full_name.name: prefixed_repo,
          Release.url.name: my_url,
          Release.sha.name: my_commit},
         _release_facts_with_repository_full_name(precomputed_facts[my_id], my_repo))
        for my_id, my_name, my_tag, my_repo, my_url, my_commit in zip(
            releases[Release.node_id.name].values[mask],
            releases[Release.name.name].values[mask],
            releases[Release.tag.name].values[mask],
            releases[Release.repository_full_name.name].values[mask],
            releases[Release.url.name].values[mask],
            releases[Release.sha.name].values[mask],
        )
        # "gone" repositories, reposet-sync has not updated yet
        if (prefixed_repo := prefixer.repo_name_to_prefixed_name.get(my_repo)) is not None
    ]
    if not with_avatars:
        return result, None, has_precomputed_facts
    mentioned_authors = np.concatenate([
        *(getattr(f, "prs_" + PullRequest.user_node_id.name) for f in precomputed_facts.values()),
        *(f.commit_authors for f in precomputed_facts.values()),
        np.nan_to_num(
            releases[Release.author_node_id.name].values, copy=False,
        ).astype(int, copy=False),
    ])
    mentioned_authors = np.unique(mentioned_authors[mentioned_authors > 0].astype(int, copy=False))
    return result, mentioned_authors, has_precomputed_facts


def _filter_by_participants(releases: List[Tuple[Dict[str, Any], ReleaseFacts]],
                            participants: ReleaseParticipants,
                            ) -> List[Tuple[Dict[str, Any], ReleaseFacts]]:
    if not releases:
        return releases
    participants = {
        k: np.array(v, dtype=int) for k, v in participants.items()
    }
    if ReleaseParticipationKind.COMMIT_AUTHOR in participants:
        commit_authors = [r[1].commit_authors for r in releases]
        lengths = np.asarray([len(ca) for ca in commit_authors])
        offsets = np.zeros(len(lengths) + 1, dtype=int)
        np.cumsum(lengths, out=offsets[1:])
        commit_authors = np.concatenate(commit_authors)
        included_indexes = np.nonzero(np.in1d(
            commit_authors, participants[ReleaseParticipationKind.COMMIT_AUTHOR]))[0]
        passed_indexes = np.unique(np.searchsorted(offsets, included_indexes, side="right") - 1)
        mask = np.full(len(releases), False)
        mask[passed_indexes] = True
        missing_indexes = np.nonzero(~mask)[0]
    else:
        missing_indexes = np.arange(len(releases))
    if len(missing_indexes) == 0:
        return releases
    if ReleaseParticipationKind.RELEASER in participants:
        still_missing = np.in1d(
            np.array([releases[i][1].publisher for i in missing_indexes], dtype=int),
            participants[ReleaseParticipationKind.RELEASER],
            invert=True)
        missing_indexes = missing_indexes[still_missing]
    if len(missing_indexes) == 0:
        return releases
    if ReleaseParticipationKind.PR_AUTHOR in participants:
        pr_authors = [getattr(releases[i][1], "prs_" + PullRequest.user_node_id.name)
                      for i in missing_indexes]
        lengths = np.asarray([len(pra) for pra in pr_authors])
        offsets = np.zeros(len(lengths) + 1, dtype=int)
        np.cumsum(lengths, out=offsets[1:])
        pr_authors = np.concatenate(pr_authors)
        included_indexes = np.nonzero(np.in1d(
            pr_authors, participants[ReleaseParticipationKind.PR_AUTHOR]))[0]
        passed_indexes = np.unique(np.searchsorted(offsets, included_indexes, side="right") - 1)
        mask = np.full(len(missing_indexes), False)
        mask[passed_indexes] = True
        missing_indexes = missing_indexes[~mask]
    if len(missing_indexes) == 0:
        return releases
    mask = np.full(len(releases), True)
    mask[missing_indexes] = False
    return [releases[i] for i in np.nonzero(mask)[0]]


def _filter_by_labels(releases: List[Tuple[Dict[str, Any], ReleaseFacts]],
                      labels_df: pd.DataFrame,
                      labels_filter: LabelFilter,
                      ) -> List[Tuple[Dict[str, Any], ReleaseFacts]]:
    if not releases:
        return releases
    left = PullRequestMiner.find_left_by_labels(
        labels_df.index, labels_df[PullRequestLabel.name.name].values, labels_filter,
    ).values
    if len(left) == 0:
        return []
    key = "prs_" + PullRequest.node_id.name
    pr_node_ids = [r[1][key] for r in releases]
    all_pr_node_ids = np.concatenate(pr_node_ids)
    passed_prs = np.nonzero(np.in1d(all_pr_node_ids, left, assume_unique=True))[0]
    indexes = np.unique(np.digitize(passed_prs, np.cumsum([len(x) for x in pr_node_ids])))
    return [releases[i] for i in indexes]


@sentry_span
async def _filter_precomputed_release_facts_by_jira(precomputed_facts: Dict[int, ReleaseFacts],
                                                    jira: JIRAFilter,
                                                    meta_ids: Tuple[int, ...],
                                                    mdb: ParallelDatabase,
                                                    cache: Optional[aiomcache.Client],
                                                    ) -> Dict[int, ReleaseFacts]:
    assert jira
    pr_ids = [getattr(f, "prs_" + PullRequest.node_id.name) for f in precomputed_facts.values()]
    if not pr_ids:
        return {}
    lengths = [len(ids) for ids in pr_ids]
    pr_ids = np.concatenate(pr_ids)
    if len(pr_ids) == 0:
        return {}
    # we could run the following in parallel with the rest, but
    # "the rest" is a no-op in most of the cases thanks to preheating
    query = await generate_jira_prs_query(
        [PullRequest.node_id.in_(pr_ids), PullRequest.acc_id.in_(meta_ids)],
        jira, mdb, cache, columns=[PullRequest.node_id])
    release_ids = np.repeat(list(precomputed_facts), lengths)
    order = np.argsort(pr_ids)
    pr_ids = pr_ids[order]
    release_ids = release_ids[order]
    rows = await mdb.fetch_all(query)
    matching_pr_ids = np.sort(np.fromiter((r[0] for r in rows), int, len(rows)))
    release_ids = np.unique(release_ids[np.searchsorted(pr_ids, matching_pr_ids)])
    return {k: precomputed_facts[k] for k in release_ids}


@sentry_span
async def _load_prs_by_merge_commit_ids(commit_ids: Sequence[str],
                                        meta_ids: Tuple[int, ...],
                                        mdb: ParallelDatabase,
                                        ) -> pd.DataFrame:
    columns = [PullRequest.merge_commit_id, *released_prs_columns]
    return await read_sql_query(
        select(columns)
        .where(and_(PullRequest.merge_commit_id.in_(commit_ids),
                    PullRequest.acc_id.in_(meta_ids)))
        .order_by(PullRequest.merge_commit_id),
        mdb, columns)


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda names, **_: ({k: sorted(v) for k, v in names.items()},),
)
async def mine_releases_by_name(names: Dict[str, Iterable[str]],
                                settings: ReleaseSettings,
                                prefixer: PrefixerPromise,
                                account: int,
                                meta_ids: Tuple[int, ...],
                                mdb: ParallelDatabase,
                                pdb: ParallelDatabase,
                                rdb: ParallelDatabase,
                                cache: Optional[aiomcache.Client],
                                ) -> Tuple[List[Tuple[Dict[str, Any], ReleaseFacts]],
                                           List[Tuple[str, str]]]:
    """Collect details about each release specified by the mapping from repository names to \
    release names."""
    log = logging.getLogger("%s.mine_releases_by_name" % metadata.__package__)
    names = {k: set(v) for k, v in names.items()}
    releases, _, branches, default_branches = await _load_releases_by_name(
        names, log, settings, prefixer, account, meta_ids, mdb, pdb, rdb, cache)
    if releases.empty:
        return [], []
    return await mine_releases_by_ids(releases, branches, default_branches, settings, prefixer,
                                      account, meta_ids, mdb, pdb, rdb, cache, with_avatars=True)


async def mine_releases_by_ids(releases: pd.DataFrame,
                               branches: pd.DataFrame,
                               default_branches: Dict[str, str],
                               settings: ReleaseSettings,
                               prefixer: PrefixerPromise,
                               account: int,
                               meta_ids: Tuple[int, ...],
                               mdb: ParallelDatabase,
                               pdb: ParallelDatabase,
                               rdb: ParallelDatabase,
                               cache: Optional[aiomcache.Client],
                               *, with_avatars: bool,
                               ) -> Union[Tuple[List[Tuple[Dict[str, Any], ReleaseFacts]],
                                                List[Tuple[str, str]]],
                                          List[Tuple[Dict[str, Any], ReleaseFacts]]]:
    """Collect details about releases in the DataFrame (`load_releases()`-like)."""
    settings_tags, settings_branches = {}, {}
    for k, v in settings.prefixed.items():
        if v.match == ReleaseMatch.tag_or_branch:
            settings_tags[k] = ReleaseMatchSetting(
                match=ReleaseMatch.tag,
                branches=v.branches,
                tags=v.tags,
            )
            settings_branches[k] = ReleaseMatchSetting(
                match=ReleaseMatch.branch,
                branches=v.branches,
                tags=v.tags,
            )
        elif v.match == ReleaseMatch.tag:
            settings_tags[k] = v
        elif v.match == ReleaseMatch.branch:
            settings_branches[k] = v
        else:
            raise AssertionError("Unsupported ReleaseMatch: %s" % v.match)
    tag_releases = releases.take(np.nonzero(
        releases[matched_by_column].values == ReleaseMatch.tag)[0])
    branch_releases = releases.take(np.nonzero(
        releases[matched_by_column].values == ReleaseMatch.branch)[0])
    precomputed_facts_tags, precomputed_facts_branches = await gather(
        load_precomputed_release_facts(
            tag_releases, default_branches, ReleaseSettings(settings_tags), account, pdb),
        load_precomputed_release_facts(
            branch_releases, default_branches, ReleaseSettings(settings_branches), account, pdb),
    )
    precomputed_facts = {**precomputed_facts_tags, **precomputed_facts_branches}
    add_pdb_hits(pdb, "release_facts", len(precomputed_facts))
    add_pdb_misses(pdb, "release_facts", len(releases) - len(precomputed_facts))
    prefixer = await prefixer.load()
    result, mentioned_authors, has_precomputed_facts = _build_mined_releases(
        releases, precomputed_facts, prefixer, with_avatars=with_avatars)
    if not (missing_releases := releases.take(np.flatnonzero(~has_precomputed_facts))).empty:
        repos = missing_releases[Release.repository_full_name.name].unique()
        time_from = missing_releases[Release.published_at.name].iloc[-1]
        time_to = missing_releases[Release.published_at.name].iloc[0] + timedelta(seconds=1)
        mined_result, mined_authors, _ = await mine_releases(
            repos, {}, branches, default_branches, time_from, time_to, LabelFilter.empty(),
            JIRAFilter.empty(), settings, prefixer.as_promise(),
            account, meta_ids, mdb, pdb, rdb, cache,
            force_fresh=True, with_avatars=False, with_pr_titles=True)
        missing_releases_by_repo = defaultdict(set)
        for repo, rid in zip(missing_releases[Release.repository_full_name.name].values,
                             missing_releases[Release.node_id.name].values):
            missing_releases_by_repo[repo].add(rid)
        for r in mined_result:
            if r[0][Release.node_id.name] in missing_releases_by_repo[
                    r[0][Release.repository_full_name.name].split("/", 1)[1]]:
                result.append(r)
        if with_avatars:
            # we don't know which are redundant, so include everyone without filtering
            mentioned_authors = np.union1d(mentioned_authors, mined_authors)

    async def dummy_avatars():
        return None

    tasks = [
        mine_user_avatars([prefixer.user_node_to_login[a] for a in mentioned_authors],
                          UserAvatarKeys.PREFIXED_LOGIN, meta_ids, mdb, cache)
        if with_avatars else dummy_avatars(),
    ]
    if precomputed_facts:
        pr_node_ids = np.concatenate([
            f["prs_" + PullRequest.node_id.name] for f in precomputed_facts.values()
        ])
        tasks.append(mdb.fetch_all(
            select([NodePullRequest.id, NodePullRequest.title])
            .where(and_(NodePullRequest.acc_id.in_(meta_ids),
                        NodePullRequest.id.in_any_values(pr_node_ids)))))
    avatars, *opt = await gather(*tasks)
    if precomputed_facts:
        pr_title_map = {row[0]: row[1] for row in opt[0]}
        for _, facts in result:
            facts["prs_" + PullRequest.title.name] = [
                pr_title_map.get(node) for node in facts["prs_" + PullRequest.node_id.name]
            ]
    if with_avatars:
        return result, avatars
    return result


async def _load_releases_by_name(names: Dict[str, Set[str]],
                                 log: logging.Logger,
                                 settings: ReleaseSettings,
                                 prefixer: PrefixerPromise,
                                 account: int,
                                 meta_ids: Tuple[int, ...],
                                 mdb: ParallelDatabase,
                                 pdb: ParallelDatabase,
                                 rdb: ParallelDatabase,
                                 cache: Optional[aiomcache.Client],
                                 ) -> Tuple[pd.DataFrame,
                                            Dict[str, Dict[str, str]],
                                            pd.DataFrame,
                                            Dict[str, str]]:
    names = await _complete_commit_hashes(names, meta_ids, mdb)
    tasks = [
        BranchMiner.extract_branches(names, meta_ids, mdb, cache),
        fetch_precomputed_releases_by_name(names, account, pdb),
    ]
    (branches, default_branches), releases = await gather(*tasks)
    prenames = defaultdict(set)
    for repo, name in zip(releases[Release.repository_full_name.name].values,
                          releases[Release.name.name].values):
        prenames[repo].add(name)
    missing = {}
    for repo, repo_names in names.items():
        if diff := repo_names.keys() - prenames.get(repo, set()):
            missing[repo] = diff
    if missing:
        now = datetime.now(timezone.utc)
        # There can be fresh new releases that are not in the pdb yet.
        match_groups, event_releases, repos_count = group_repos_by_release_match(
            missing, default_branches, settings)
        # event releases will be loaded in any case
        spans = await ReleaseLoader.fetch_precomputed_release_match_spans(
            match_groups, account, pdb)
        offset = timedelta(hours=2)
        max_offset = timedelta(days=5 * 365)
        for repo in missing:
            try:
                have_not_precomputed = False
                for (span_start, span_end) in spans[repo].values():
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
            missing, branches, default_branches, now - offset, now,
            settings, prefixer, account, meta_ids, mdb, pdb, rdb, cache, force_fresh=True)
        new_releases_index = defaultdict(dict)
        for i, (repo, name) in enumerate(zip(
                new_releases[Release.repository_full_name.name].values,
                new_releases[Release.name.name].values)):
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
            releases = releases.append(new_releases.take(matching_indexes), ignore_index=True)
            releases.sort_values(Release.published_at.name,
                                 inplace=True, ascending=False, ignore_index=True)
            if event_releases:
                # we could load them twice
                releases.drop_duplicates(subset=Release.node_id.name, inplace=True)
        if still_missing:
            log.warning("Some releases were not found: %s", still_missing)
    return releases, names, branches, default_branches


commit_prefix_re = re.compile(r"[a-f0-9]{7}")


@sentry_span
async def _complete_commit_hashes(names: Dict[str, Set[str]],
                                  meta_ids: Tuple[int, ...],
                                  mdb: ParallelDatabase) -> Dict[str, Dict[str, str]]:
    candidates = defaultdict(list)
    for repo, strs in names.items():
        for name in strs:
            if commit_prefix_re.fullmatch(name):
                candidates[repo].append(name)
    if not candidates:
        return {repo: {s: s for s in strs} for repo, strs in names.items()}
    queries = [
        select([PushCommit.repository_full_name, PushCommit.sha])
        .where(and_(PushCommit.acc_id.in_(meta_ids),
                    PushCommit.repository_full_name == repo,
                    func.substr(PushCommit.sha, 1, 7).in_(prefixes)))
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
    key=lambda borders, **_: ({k: sorted(v) for k, v in borders.items()},),
)
async def diff_releases(borders: Dict[str, List[Tuple[str, str]]],
                        settings: ReleaseSettings,
                        prefixer: PrefixerPromise,
                        account: int,
                        meta_ids: Tuple[int, ...],
                        mdb: ParallelDatabase,
                        pdb: ParallelDatabase,
                        rdb: ParallelDatabase,
                        cache: Optional[aiomcache.Client],
                        ) -> Tuple[
        Dict[str, List[Tuple[str, str, List[Tuple[Dict[str, Any], ReleaseFacts]]]]],
        List[Tuple[str, str]]]:
    """Collect details about inner releases between the given boundaries for each repo."""
    log = logging.getLogger("%s.diff_releases" % metadata.__package__)
    names = defaultdict(set)
    for repo, pairs in borders.items():
        for old, new in pairs:
            names[repo].update((old, new))
    border_releases, names, branches, default_branches = await _load_releases_by_name(
        names, log, settings, prefixer, account, meta_ids, mdb, pdb, rdb, cache)
    if border_releases.empty:
        return {}, []
    repos = border_releases[Release.repository_full_name.name].unique()
    time_from = border_releases[Release.published_at.name].min()
    time_to = border_releases[Release.published_at.name].max() + timedelta(seconds=1)

    async def fetch_dags():
        nonlocal border_releases
        dags = await fetch_precomputed_commit_history_dags(repos, account, pdb, cache)
        return await fetch_repository_commits(
            dags, border_releases, RELEASE_FETCH_COMMITS_COLUMNS, True, account, meta_ids,
            mdb, pdb, cache)

    tasks = [
        mine_releases(
            repos, {}, branches, default_branches, time_from, time_to, LabelFilter.empty(),
            JIRAFilter.empty(), settings, prefixer, account, meta_ids, mdb, pdb, rdb, cache,
            force_fresh=True, with_pr_titles=True),
        fetch_dags(),
    ]
    (releases, avatars, _), dags = await gather(*tasks, op="mine_releases + dags")
    del border_releases
    releases_by_repo = defaultdict(list)
    for r in releases:
        releases_by_repo[r[0][Release.repository_full_name.name]].append(r)
    del releases
    result = {}
    for repo, repo_releases in releases_by_repo.items():
        repo = repo.split("/", 1)[1]
        repo_names = {v: k for k, v in names[repo].items()}
        pairs = borders[repo]
        result[repo] = repo_result = []
        repo_releases = sorted(repo_releases, key=lambda r: r[1].published)
        index = {r[0][Release.name.name]: i for i, r in enumerate(repo_releases)}
        for old, new in pairs:
            try:
                start = index[repo_names[old]]
                finish = index[repo_names[new]]
            except KeyError:
                log.warning("Release pair %s, %s was not found for %s", old, new, repo)
                continue
            if start > finish:
                log.warning("Release pair old %s is later than new %s for %s", old, new, repo)
                continue
            start_sha, finish_sha = (repo_releases[x][0][Release.sha.name].encode()
                                     for x in (start, finish))
            hashes, _, _ = extract_subdag(*dags[repo], np.array([finish_sha]))
            if hashes[searchsorted_inrange(hashes, np.array([start_sha]))] == start_sha:
                diff = []
                for i in range(start + 1, finish + 1):
                    r = repo_releases[i]
                    sha = r[0][Release.sha.name].encode()
                    if hashes[searchsorted_inrange(hashes, np.array([sha]))] == sha:
                        diff.append(r)
                repo_result.append((old, new, diff))
            else:
                log.warning("Release pair's old %s is not in the sub-DAG of %s for %s",
                            old, new, repo)
    return result, avatars if any(any(d for _, _, d in v) for v in result.values()) else {}
