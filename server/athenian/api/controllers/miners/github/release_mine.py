import asyncio
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging
import pickle
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import aiomcache
import databases
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, func, select, union_all
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import JIRAFilter
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.commit import fetch_precomputed_commit_history_dags, \
    fetch_repository_commits, RELEASE_FETCH_COMMITS_COLUMNS
from athenian.api.controllers.miners.github.dag_accelerated import extract_subdag, \
    mark_dag_access, mark_dag_parents, searchsorted_inrange
from athenian.api.controllers.miners.github.precomputed_releases import \
    fetch_precomputed_releases_by_name, load_precomputed_release_facts, \
    store_precomputed_release_facts
from athenian.api.controllers.miners.github.release_load import \
    fetch_precomputed_release_match_spans, group_repos_by_release_match, load_releases
from athenian.api.controllers.miners.github.release_match import \
    _fetch_repository_first_commit_dates, _find_releases_for_matching_prs, load_commit_dags
from athenian.api.controllers.miners.github.released_pr import matched_by_column
from athenian.api.controllers.miners.github.users import mine_user_avatars
from athenian.api.controllers.miners.jira.issue import generate_jira_prs_query
from athenian.api.controllers.miners.types import ReleaseFacts, ReleaseParticipants, \
    ReleaseParticipationKind
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting
from athenian.api.db import add_pdb_hits, add_pdb_misses
from athenian.api.defer import defer
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import PullRequest, PushCommit, Release
from athenian.api.tracing import sentry_span


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, participants, time_from, time_to, jira, settings, **_: (
        ",".join(sorted(repos)),
        ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(participants.items())),
        time_from, time_to, jira, settings),
)
async def mine_releases(repos: Iterable[str],
                        participants: ReleaseParticipants,
                        branches: pd.DataFrame,
                        default_branches: Dict[str, str],
                        time_from: datetime,
                        time_to: datetime,
                        jira: JIRAFilter,
                        settings: Dict[str, ReleaseMatchSetting],
                        meta_ids: Tuple[int, ...],
                        mdb: databases.Database,
                        pdb: databases.Database,
                        cache: Optional[aiomcache.Client],
                        force_fresh: bool = False,
                        with_avatars: bool = True,
                        ) -> Tuple[List[Tuple[Dict[str, Any], ReleaseFacts]],
                                   Union[List[Tuple[str, str]], np.ndarray],
                                   Dict[str, ReleaseMatch]]:
    """Collect details about each release published between `time_from` and `time_to` and \
    calculate various statistics.

    :param force_fresh: Ensure that we load the most up to date releases, no matter the state of \
                        the pdb is.
    :param with_avatars: Indicates whether to return the fetched user avatars or just an array of \
                         unique mentioned logins.
    """
    prefix = PREFIXES["github"]
    log = logging.getLogger("%s.mine_releases" % metadata.__package__)
    releases_in_time_range, matched_bys = await load_releases(
        repos, branches, default_branches, time_from, time_to,
        settings, meta_ids, mdb, pdb, cache, force_fresh=force_fresh)
    # resolve ambiguous release match settings
    settings = settings.copy()
    for repo in repos:
        setting = settings[prefix + repo]
        match = ReleaseMatch(matched_bys.get(repo, setting.match))
        settings[prefix + repo] = ReleaseMatchSetting(
            tags=setting.tags,
            branches=setting.branches,
            match=match,
        )
    if releases_in_time_range.empty:
        return [], [], {r: v.match for r, v in settings.items()}
    precomputed_facts = await load_precomputed_release_facts(
        releases_in_time_range, default_branches, settings, pdb)
    # uncomment this to compute releases from scratch
    # precomputed_facts = {}
    add_pdb_hits(pdb, "release_facts", len(precomputed_facts))
    missed_releases_count = len(releases_in_time_range) - len(precomputed_facts)
    add_pdb_misses(pdb, "release_facts", missed_releases_count)
    unfiltered_precomputed_facts = precomputed_facts
    if jira:
        precomputed_facts = await _filter_precomputed_release_facts_by_jira(
            precomputed_facts, jira, meta_ids, mdb, cache)
    result, mentioned_authors, has_precomputed_facts = _build_mined_releases(
        releases_in_time_range, precomputed_facts)

    missing_repos = releases_in_time_range[Release.repository_full_name.key].take(
        np.where(~has_precomputed_facts)[0]).unique()
    commits_authors = prs_authors = []
    commits_authors_nz = prs_authors_nz = slice(0)
    repo_releases_analyzed = {}
    if missed_releases_count > 0:
        releases_in_time_range = releases_in_time_range.take(np.where(
            releases_in_time_range[Release.repository_full_name.key].isin(missing_repos).values,
        )[0])
        _, releases, _, _ = await _find_releases_for_matching_prs(
            missing_repos, branches, default_branches, time_from, time_to, False,
            settings, meta_ids, mdb, pdb, cache, releases_in_time_range=releases_in_time_range)
        tasks = [
            load_commit_dags(releases, meta_ids, mdb, pdb, cache),
            _fetch_repository_first_commit_dates(missing_repos, meta_ids, mdb, pdb, cache),
        ]
        dags, first_commit_dates = await gather(*tasks, op="mine_releases/commits")

        all_hashes = []
        for repo, repo_releases in releases.groupby(Release.repository_full_name.key, sort=False):
            hashes, vertexes, edges = dags[repo]
            release_hashes = repo_releases[Release.sha.key].values
            release_timestamps = repo_releases[Release.published_at.key].values
            parents = mark_dag_parents(hashes, vertexes, edges, release_hashes, release_timestamps)
            ownership = mark_dag_access(hashes, vertexes, edges, release_hashes)
            precomputed_mask = \
                repo_releases[Release.id.key].isin(unfiltered_precomputed_facts).values
            out_of_range_mask = release_timestamps < np.array(time_from.replace(tzinfo=None),
                                                              dtype=release_timestamps.dtype)
            relevant = np.nonzero(~(precomputed_mask | out_of_range_mask))[0]
            if len(relevant) == 0:
                continue
            if len(removed := np.nonzero(np.in1d(ownership, relevant, invert=True))[0]) > 0:
                hashes = np.delete(hashes, removed)
                ownership = np.delete(ownership, removed)
            order = np.argsort(ownership)
            sorted_hashes = hashes[order]
            sorted_ownership = ownership[order]
            unique_owners, unique_owned_counts = np.unique(sorted_ownership, return_counts=True)
            if len(unique_owned_counts) == 0:
                grouped_owned_hashes = []
            else:
                grouped_owned_hashes = np.split(sorted_hashes, np.cumsum(unique_owned_counts)[:-1])
            # fill the gaps for releases with 0 owned commits
            if len(missing := np.setdiff1d(np.arange(len(repo_releases)), unique_owners,
                                           assume_unique=True)):
                if len(really_missing := np.nonzero(np.in1d(
                        missing, relevant, assume_unique=True))[0]):
                    log.warning("%s has releases with 0 commits:\n%s",
                                repo, repo_releases.take(really_missing))
                empty = np.array([], dtype="U40")
                for i in missing:
                    grouped_owned_hashes.insert(i, empty)
            assert len(grouped_owned_hashes) == len(repo_releases)
            all_hashes.append(hashes)
            repo_releases_analyzed[repo] = repo_releases, grouped_owned_hashes, parents
        commits_df_columns = [
            PushCommit.sha,
            PushCommit.additions,
            PushCommit.deletions,
            PushCommit.author_login,
            PushCommit.node_id,
        ]
        all_hashes = np.concatenate(all_hashes) if all_hashes else []
        with sentry_sdk.start_span(op="mine_releases/fetch_commits",
                                   description=str(len(all_hashes))):
            commits_df = await read_sql_query(
                select(commits_df_columns)
                .where(and_(PushCommit.sha.in_(all_hashes), PushCommit.acc_id.in_(meta_ids)))
                .order_by(PushCommit.sha),
                mdb, commits_df_columns, index=PushCommit.sha.key)
        commits_index = commits_df.index.values.astype("U40")
        commit_ids = commits_df[PushCommit.node_id.key].values
        commits_additions = commits_df[PushCommit.additions.key].values
        commits_deletions = commits_df[PushCommit.deletions.key].values
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
        commits_authors = commits_df[PushCommit.author_login.key].values
        commits_authors_nz = commits_authors.nonzero()[0]
        commits_authors[commits_authors_nz] = prefix + commits_authors[commits_authors_nz]

        tasks = [_load_prs_by_merge_commit_ids(commit_ids, meta_ids, mdb)]
        if jira:
            query = await generate_jira_prs_query(
                [PullRequest.merge_commit_id.in_(commit_ids),
                 PullRequest.acc_id.in_(meta_ids)],
                jira, mdb, cache, columns=[PullRequest.merge_commit_id])
            tasks.append(mdb.fetch_all(query))
        results = await gather(*tasks,
                               op="mine_releases/fetch_pull_requests",
                               description=str(len(commit_ids)))
        prs_df, prs_columns = results[0]
        if jira:
            filtered_prs_commit_ids = np.unique(np.array([r[0] for r in results[1]], dtype="U"))
        prs_commit_ids = prs_df[PullRequest.merge_commit_id.key].values.astype("U")
        prs_authors = prs_df[PullRequest.user_login.key].values
        prs_authors_nz = prs_authors.nonzero()[0]
        prs_authors[prs_authors_nz] = prefix + prs_authors[prs_authors_nz]
        prs_node_ids = prs_df[PullRequest.node_id.key].values.astype("U")
        prs_numbers = prs_df[PullRequest.number.key].values
        prs_titles = prs_df[PullRequest.title.key].values
        prs_additions = prs_df[PullRequest.additions.key].values
        prs_deletions = prs_df[PullRequest.deletions.key].values

    @sentry_span
    async def main_flow():
        data = []
        for repo, (repo_releases, owned_hashes, parents) in repo_releases_analyzed.items():
            computed_release_info_by_commit = {}
            for i, (my_id, my_name, my_tag, my_url, my_author, my_published_at,
                    my_matched_by, my_commit) in \
                    enumerate(zip(repo_releases[Release.id.key].values,
                                  repo_releases[Release.name.key].values,
                                  repo_releases[Release.tag.key].values,
                                  repo_releases[Release.url.key].values,
                                  repo_releases[Release.author.key].values,
                                  repo_releases[Release.published_at.key],  # no values
                                  repo_releases[matched_by_column].values,
                                  repo_releases[Release.sha.key].values)):
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
                    mentioned_authors.update(my_prs_authors[my_prs_authors.nonzero()[0]])
                    my_prs = dict(zip(
                        [c.key for c in prs_columns[1:]],
                        [prs_node_ids[my_prs_indexes],
                         prs_numbers[my_prs_indexes],
                         prs_titles[my_prs_indexes],
                         prs_additions[my_prs_indexes],
                         prs_deletions[my_prs_indexes],
                         my_prs_authors]))

                    my_commit_authors = \
                        np.unique(my_commit_authors[my_commit_authors.nonzero()[0]]).tolist()
                    mentioned_authors.update(my_commit_authors)
                    parent = parents[i]
                    if parent < len(repo_releases):
                        my_age = \
                            my_published_at - repo_releases[Release.published_at.key]._ixs(parent)
                    else:
                        my_age = my_published_at - first_commit_dates[repo]
                    if my_author is not None:
                        my_author = prefix + my_author
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
                data.append(({Release.id.key: my_id,
                              Release.name.key: my_name or my_tag,
                              Release.repository_full_name.key: prefix + repo,
                              Release.url.key: my_url,
                              Release.sha.key: my_commit},
                             ReleaseFacts(published=my_published_at,
                                          publisher=my_author,
                                          matched_by=ReleaseMatch(my_matched_by),
                                          age=my_age,
                                          additions=my_additions,
                                          deletions=my_deletions,
                                          commits_count=commits_count,
                                          prs=my_prs,
                                          commit_authors=my_commit_authors)))
            await asyncio.sleep(0)
        if data:
            await defer(store_precomputed_release_facts(data, default_branches, settings, pdb),
                        "store_precomputed_release_facts(%d)" % len(data))
        return data

    if with_avatars:
        all_authors = np.concatenate([commits_authors[commits_authors_nz],
                                      prs_authors[prs_authors_nz],
                                      mentioned_authors])
        all_authors = [p[1] for p in np.char.split(np.unique(all_authors).astype("U"), "/", 1)]
    mentioned_authors = set(mentioned_authors)
    if with_avatars:
        tasks = [
            main_flow(),
            mine_user_avatars(all_authors, meta_ids, mdb, cache, prefix=prefix),
        ]
        mined_releases, avatars = await gather(*tasks, op="main_flow + avatars")
        avatars = [p for p in avatars if p[0] in mentioned_authors]
    else:
        mined_releases = await main_flow()
        avatars = np.array(
            [p[1] for p in np.char.split(np.array(list(mentioned_authors), dtype="U"), "/", 1)])
    result.extend(mined_releases)
    if participants:
        result = _filter_by_participants(result, participants)
    return result, avatars, {r: v.match for r, v in settings.items()}


def _build_mined_releases(releases: pd.DataFrame,
                          precomputed_facts: Dict[str, ReleaseFacts],
                          ) -> Tuple[List[Tuple[Dict[str, Any], ReleaseFacts]],
                                     np.ndarray,
                                     np.ndarray]:
    prefix = PREFIXES["github"]
    has_precomputed_facts = releases[Release.id.key].isin(precomputed_facts).values
    result = [
        ({Release.id.key: my_id,
          Release.name.key: my_name or my_tag,
          Release.repository_full_name.key: prefix + repo,
          Release.url.key: my_url,
          Release.sha.key: my_commit},
         precomputed_facts[my_id])
        for my_id, my_name, my_tag, repo, my_url, my_commit in zip(
            releases[Release.id.key].values[has_precomputed_facts],
            releases[Release.name.key].values[has_precomputed_facts],
            releases[Release.tag.key].values[has_precomputed_facts],
            releases[Release.repository_full_name.key].values[has_precomputed_facts],
            releases[Release.url.key].values[has_precomputed_facts],
            releases[Release.sha.key].values[has_precomputed_facts],
        )
    ]
    release_authors = releases[Release.author.key].values
    mentioned_authors = np.concatenate([
        *(f.prs[PullRequest.user_login.key] for f in precomputed_facts.values()),
        *(f.commit_authors for f in precomputed_facts.values()),
        prefix + release_authors[release_authors.nonzero()[0]],
    ])
    mentioned_authors = np.unique(mentioned_authors[mentioned_authors.nonzero()[0]]).astype("U")
    return result, mentioned_authors, has_precomputed_facts


def _filter_by_participants(releases: List[Tuple[Dict[str, Any], ReleaseFacts]],
                            participants: ReleaseParticipants,
                            ) -> List[Tuple[Dict[str, Any], ReleaseFacts]]:
    participants = participants.copy()
    for k, v in participants.items():
        participants[k] = np.unique(v).astype("U")
    if ReleaseParticipationKind.COMMIT_AUTHOR in participants:
        commit_authors = [r[1].commit_authors for r in releases]
        lengths = np.asarray([len(ca) for ca in commit_authors])
        offsets = np.zeros(len(lengths) + 1, dtype=int)
        np.cumsum(lengths, out=offsets[1:])
        commit_authors = np.concatenate(commit_authors).astype("U")
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
            np.array([releases[i][1].publisher for i in missing_indexes], dtype="U"),
            participants[ReleaseParticipationKind.RELEASER],
            invert=True)
        missing_indexes = missing_indexes[still_missing]
    if len(missing_indexes) == 0:
        return releases
    if ReleaseParticipationKind.PR_AUTHOR in participants:
        key = PullRequest.user_login.key
        pr_authors = [releases[i][1].prs[key] for i in missing_indexes]
        lengths = np.asarray([len(pra) for pra in pr_authors])
        offsets = np.zeros(len(lengths) + 1, dtype=int)
        np.cumsum(lengths, out=offsets[1:])
        pr_authors = np.concatenate(pr_authors).astype("U")
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


@sentry_span
async def _filter_precomputed_release_facts_by_jira(precomputed_facts: Dict[str, ReleaseFacts],
                                                    jira: JIRAFilter,
                                                    meta_ids: Tuple[int, ...],
                                                    mdb: databases.Database,
                                                    cache: Optional[aiomcache.Client],
                                                    ) -> Dict[str, ReleaseFacts]:
    assert jira
    pr_ids = [f.prs[PullRequest.node_id.key] for f in precomputed_facts.values()]
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
    pr_ids = pr_ids.astype("U")
    release_ids = np.repeat(list(precomputed_facts), lengths)
    order = np.argsort(pr_ids)
    pr_ids = pr_ids[order]
    release_ids = release_ids[order]
    matching_pr_ids = np.sort(np.array([r[0] for r in await mdb.fetch_all(query)], dtype="U"))
    release_ids = np.unique(release_ids[np.searchsorted(pr_ids, matching_pr_ids)])
    return {k: precomputed_facts[k] for k in release_ids}


@sentry_span
async def _load_prs_by_merge_commit_ids(commit_ids: Sequence[str],
                                        meta_ids: Tuple[int, ...],
                                        mdb: databases.Database,
                                        ) -> Tuple[pd.DataFrame, List[InstrumentedAttribute]]:
    prs_columns = [
        PullRequest.merge_commit_id,
        PullRequest.node_id,
        PullRequest.number,
        PullRequest.title,
        PullRequest.additions,
        PullRequest.deletions,
        PullRequest.user_login,
    ]
    df = await read_sql_query(
        select(prs_columns)
        .where(and_(PullRequest.merge_commit_id.in_(commit_ids),
                    PullRequest.acc_id.in_(meta_ids)))
        .order_by(PullRequest.merge_commit_id),
        mdb, prs_columns)
    return df, prs_columns


@sentry_span
@cached(
    exptime=5 * 60,  # 5 min
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda names, **_: ({k: sorted(v) for k, v in names.items()},),
)
async def mine_releases_by_name(names: Dict[str, Iterable[str]],
                                settings: Dict[str, ReleaseMatchSetting],
                                meta_ids: Tuple[int, ...],
                                mdb: databases.Database,
                                pdb: databases.Database,
                                cache: Optional[aiomcache.Client],
                                ) -> Tuple[List[Tuple[Dict[str, Any], ReleaseFacts]],
                                           List[Tuple[str, str]]]:
    """Collect details about each release specified by the mapping from repository names to \
    release names."""
    log = logging.getLogger("%s.mine_releases_by_name" % metadata.__package__)
    names = {k: set(v) for k, v in names.items()}
    releases, _, branches, default_branches = await _load_releases_by_name(
        names, log, settings, meta_ids, mdb, pdb, cache)
    if releases.empty:
        return [], []
    settings_tags, settings_branches = {}, {}
    for k, v in settings.items():
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
        load_precomputed_release_facts(tag_releases, default_branches, settings_tags, pdb),
        load_precomputed_release_facts(branch_releases, default_branches, settings_branches, pdb),
    )
    precomputed_facts = {**precomputed_facts_tags, **precomputed_facts_branches}
    add_pdb_hits(pdb, "release_facts", len(precomputed_facts))
    add_pdb_misses(pdb, "release_facts", len(releases) - len(precomputed_facts))
    result, mentioned_authors, has_precomputed_facts = _build_mined_releases(
        releases, precomputed_facts)
    mentioned_authors = [p[1] for p in np.char.split(mentioned_authors, "/", 1)]
    if not (missing_releases := releases.take(np.nonzero(~has_precomputed_facts)[0])).empty:
        repos = missing_releases[Release.repository_full_name.key].unique()
        time_from = missing_releases[Release.published_at.key].iloc[-1]
        time_to = missing_releases[Release.published_at.key].iloc[0] + timedelta(seconds=1)
        mined_result, mined_authors, _ = await mine_releases(
            repos, {}, branches, default_branches, time_from, time_to, JIRAFilter.empty(),
            settings, meta_ids, mdb, pdb, cache, force_fresh=True, with_avatars=False)
        missing_releases_by_repo = defaultdict(set)
        for repo, name in zip(missing_releases[Release.repository_full_name.key].values,
                              missing_releases[Release.name.key].values):
            missing_releases_by_repo[repo].add(name)
        for r in mined_result:
            if r[0][Release.name.key] in missing_releases_by_repo[
                    r[0][Release.repository_full_name.key].split("/", 1)[1]]:
                result.append(r)
        # we don't know which are redundant, so include everyone without filtering
        mentioned_authors = np.unique(np.concatenate([mentioned_authors, mined_authors]))
    avatars = await mine_user_avatars(
        mentioned_authors, meta_ids, mdb, cache, prefix=PREFIXES["github"])
    return result, avatars


async def _load_releases_by_name(names: Dict[str, Set[str]],
                                 log: logging.Logger,
                                 settings: Dict[str, ReleaseMatchSetting],
                                 meta_ids: Tuple[int, ...],
                                 mdb: databases.Database,
                                 pdb: databases.Database,
                                 cache: Optional[aiomcache.Client],
                                 ) -> Tuple[pd.DataFrame,
                                            pd.DataFrame,
                                            Dict[str, Dict[str, str]],
                                            Dict[str, str]]:
    names = await _complete_commit_hashes(names, meta_ids, mdb)
    tasks = [
        extract_branches(names, meta_ids, mdb, cache),
        fetch_precomputed_releases_by_name(names, pdb),
    ]
    (branches, default_branches), releases = await gather(*tasks)
    prenames = defaultdict(set)
    for repo, name in zip(releases[Release.repository_full_name.key].values,
                          releases[Release.name.key].values):
        prenames[repo].add(name)
    missing = {}
    for repo, repo_names in names.items():
        if diff := repo_names.keys() - prenames.get(repo, set()):
            missing[repo] = diff
    if missing:
        now = datetime.now(timezone.utc)
        # There can be fresh new releases that are not in the pdb yet.
        match_groups, repos_count = group_repos_by_release_match(
            missing, default_branches, settings)
        spans = await fetch_precomputed_release_match_spans(match_groups, pdb)
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
        new_releases, _ = await load_releases(
            missing, branches, default_branches, now - offset, now,
            settings, meta_ids, mdb, pdb, cache, force_fresh=True)
        new_releases_index = defaultdict(dict)
        for i, (repo, name) in enumerate(zip(new_releases[Release.repository_full_name.key].values,
                                             new_releases[Release.name.key].values)):
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
            releases.sort_values(Release.published_at.key,
                                 inplace=True, ascending=False, ignore_index=True)
        if still_missing:
            log.warning("Some releases were not found: %s", still_missing)
    return releases, names, branches, default_branches


commit_prefix_re = re.compile(r"[a-f0-9]{7}")


@sentry_span
async def _complete_commit_hashes(names: Dict[str, Set[str]],
                                  meta_ids: Tuple[int, ...],
                                  mdb: databases.Database) -> Dict[str, Dict[str, str]]:
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
        repo = row[PushCommit.repository_full_name.key]
        sha = row[PushCommit.sha.key]
        prefix = sha[:7]
        renames[repo][sha] = prefix
        renames_reversed[repo].add(prefix)
    for repo, strs in names.items():
        repo_renames = renames_reversed[repo]
        for name in strs:
            if name not in repo_renames:
                repo_renames[name] = name
    return renames


@cached(
    exptime=5 * 60,  # 5 min
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda borders, **_: ({k: sorted(v) for k, v in borders.items()},),
)
async def diff_releases(borders: Dict[str, List[Tuple[str, str]]],
                        settings: Dict[str, ReleaseMatchSetting],
                        meta_ids: Tuple[int, ...],
                        mdb: databases.Database,
                        pdb: databases.Database,
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
        names, log, settings, meta_ids, mdb, pdb, cache)
    if border_releases.empty:
        return {}, []
    repos = border_releases[Release.repository_full_name.key].unique()
    time_from = border_releases[Release.published_at.key].min()
    time_to = border_releases[Release.published_at.key].max() + timedelta(seconds=1)

    async def fetch_dags():
        nonlocal border_releases
        dags = await fetch_precomputed_commit_history_dags(repos, pdb, cache)
        return await fetch_repository_commits(
            dags, border_releases, RELEASE_FETCH_COMMITS_COLUMNS, True, meta_ids, mdb, pdb, cache)

    tasks = [
        mine_releases(
            repos, {}, branches, default_branches, time_from, time_to, JIRAFilter.empty(),
            settings, meta_ids, mdb, pdb, cache, force_fresh=True),
        fetch_dags(),
    ]
    (releases, avatars, _), dags = await gather(*tasks, op="mine_releases + dags")
    del border_releases
    releases_by_repo = defaultdict(list)
    for r in releases:
        releases_by_repo[r[0][Release.repository_full_name.key]].append(r)
    del releases
    result = {}
    for repo, repo_releases in releases_by_repo.items():
        repo = repo.split("/", 1)[1]
        repo_names = {v: k for k, v in names[repo].items()}
        pairs = borders[repo]
        result[repo] = repo_result = []
        repo_releases = sorted(repo_releases, key=lambda r: r[1].published)
        index = {r[0][Release.name.key]: i for i, r in enumerate(repo_releases)}
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
            start_sha, finish_sha = (repo_releases[x][0][Release.sha.key] for x in (start, finish))
            hashes, _, _ = extract_subdag(*dags[repo], np.array([finish_sha], dtype="U"))
            if hashes[searchsorted_inrange(hashes, np.array([start_sha], dtype="U"))] == start_sha:
                diff = []
                for i in range(start + 1, finish + 1):
                    r = repo_releases[i]
                    sha = r[0][Release.sha.key]
                    if hashes[searchsorted_inrange(hashes, np.array([sha], dtype="U"))] == sha:
                        diff.append(r)
                repo_result.append((old, new, diff))
            else:
                log.warning("Release pair's old %s is not in the sub-DAG of %s for %s",
                            old, new, repo)
    return result, avatars
