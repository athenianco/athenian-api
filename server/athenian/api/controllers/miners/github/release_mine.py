import asyncio
from datetime import datetime
import logging
import pickle
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aiomcache
import databases
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.filters import JIRAFilter
from athenian.api.controllers.miners.github.dag_accelerated import mark_dag_access, \
    mark_dag_parents, searchsorted_inrange
from athenian.api.controllers.miners.github.precomputed_releases import \
    load_precomputed_release_facts, store_precomputed_release_facts
from athenian.api.controllers.miners.github.release_load import load_releases
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
                        ) -> Tuple[List[Tuple[Dict[str, Any], ReleaseFacts]],
                                   List[Tuple[str, str]],
                                   Dict[str, ReleaseMatch]]:
    """Collect details about each release published between `time_from` and `time_to` and \
    calculate various statistics."""
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
    has_precomputed_facts = releases_in_time_range[Release.id.key].isin(precomputed_facts).values
    missing_repos = releases_in_time_range[Release.repository_full_name.key].take(
        np.where(~has_precomputed_facts)[0]).unique()
    missed_releases_count = len(releases_in_time_range) - len(precomputed_facts)
    add_pdb_misses(pdb, "release_facts", missed_releases_count)
    unfiltered_precomputed_facts = precomputed_facts

    if jira:
        precomputed_facts = await _filter_precomputed_release_facts_by_jira(
            precomputed_facts, jira, meta_ids, mdb, cache)
        has_precomputed_facts = \
            releases_in_time_range[Release.id.key].isin(precomputed_facts).values
    result = [
        ({Release.id.key: my_id,
          Release.name.key: my_name or my_tag,
          Release.repository_full_name.key: prefix + repo,
          Release.url.key: my_url,
          Release.author.key: my_author},
         precomputed_facts[my_id])
        for my_id, my_name, my_tag, repo, my_url, my_author in zip(
            releases_in_time_range[Release.id.key].values[has_precomputed_facts],
            releases_in_time_range[Release.name.key].values[has_precomputed_facts],
            releases_in_time_range[Release.tag.key].values[has_precomputed_facts],
            releases_in_time_range[Release.repository_full_name.key].values[has_precomputed_facts],
            releases_in_time_range[Release.url.key].values[has_precomputed_facts],
            releases_in_time_range[Release.author.key].values[has_precomputed_facts],
        )
    ]
    commits_authors = prs_authors = []
    commits_authors_nz = prs_authors_nz = slice(0)
    release_authors = releases_in_time_range[Release.author.key].values
    release_authors = prefix + release_authors[release_authors.nonzero()[0]]
    mentioned_authors = (
        [f.prs[PullRequest.user_login.key] for f in precomputed_facts.values()
         if time_from <= f.published < time_to] +
        [f.commit_authors for f in precomputed_facts.values()
         if time_from <= f.published < time_to]
    )
    if mentioned_authors:
        mentioned_authors = np.concatenate(mentioned_authors)
        mentioned_authors = np.unique(mentioned_authors[mentioned_authors.nonzero()[0]])
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
                                  repo_releases[Release.commit_id.key].values)):
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
                              Release.url.key: my_url},
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
        return data

    all_authors = np.concatenate([release_authors,
                                  commits_authors[commits_authors_nz],
                                  prs_authors[prs_authors_nz],
                                  mentioned_authors])
    all_authors = [p[1] for p in np.char.split(np.unique(all_authors).astype("U"), "/", 1)]
    mentioned_authors = set(mentioned_authors)
    tasks = [
        main_flow(),
        mine_user_avatars(all_authors, meta_ids, mdb, cache, prefix=prefix),
    ]
    mined_releases, avatars = await gather(*tasks, op="main_flow + avatars")
    await defer(store_precomputed_release_facts(mined_releases, default_branches, settings, pdb),
                "store_precomputed_release_facts(%d)" % len(mined_releases))
    avatars = [p for p in avatars if p[0] in mentioned_authors]
    result.extend(mined_releases)
    if participants:
        result = _filter_by_participants(result, participants)
    return result, avatars, {r: v.match for r, v in settings.items()}


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
