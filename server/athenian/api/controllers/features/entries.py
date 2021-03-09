import asyncio
from datetime import datetime
from functools import partial, reduce
from itertools import chain
import pickle
from typing import Collection, Dict, List, Optional, Sequence, Set, Tuple

import aiomcache
from databases import Database
import numpy as np
import sentry_sdk

from athenian.api import COROUTINE_YIELD_EVERY_ITER
from athenian.api.async_utils import gather
from athenian.api.cache import cached, CancelCache
from athenian.api.controllers.datetime_utils import coarsen_time_interval
from athenian.api.controllers.features.code import CodeStats
from athenian.api.controllers.features.github.code import calc_code_stats
from athenian.api.controllers.features.github.developer_metrics import \
    DeveloperBinnedMetricCalculator, group_actions_by_developers
from athenian.api.controllers.features.github.pull_request_metrics import \
    group_by_lines, group_prs_by_participants, need_jira_mapping, \
    PullRequestBinnedHistogramCalculator, \
    PullRequestBinnedMetricCalculator
from athenian.api.controllers.features.github.release_metrics import \
    group_releases_by_participants, merge_release_participants, ReleaseBinnedMetricCalculator
from athenian.api.controllers.features.github.unfresh_pull_request_metrics import \
    fetch_pull_request_facts_unfresh
from athenian.api.controllers.features.histogram import HistogramParameters
from athenian.api.controllers.features.metric_calculator import df_from_dataclasses, \
    group_by_repo, group_to_indexes
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.bots import bots
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.commit import extract_commits, FilterCommitsProperty
from athenian.api.controllers.miners.github.developer import \
    developer_repository_column, DeveloperTopic, mine_developer_activities
from athenian.api.controllers.miners.github.precomputed_prs import \
    load_precomputed_done_candidates, load_precomputed_done_facts_filters, \
    remove_ambiguous_prs, store_merged_unreleased_pull_request_facts, \
    store_open_pull_request_facts, store_precomputed_done_facts
from athenian.api.controllers.miners.github.pull_request import ImpossiblePullRequest, \
    PullRequestFactsMiner, PullRequestMiner
from athenian.api.controllers.miners.github.release_mine import mine_releases
from athenian.api.controllers.miners.jira.issue import append_pr_jira_mapping, \
    load_pr_jira_mapping
from athenian.api.controllers.miners.types import PRParticipants, PRParticipationKind, \
    PullRequestFacts, ReleaseParticipants
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting
from athenian.api.db import add_pdb_hits, add_pdb_misses
from athenian.api.defer import defer
from athenian.api.models.metadata.github import PullRequest, PushCommit, Release
from athenian.api.tracing import sentry_span


unfresh_prs_threshold = 1000
unfresh_participants_threshold = 50


def _postprocess_cached_facts(result: Tuple[Dict[str, List[PullRequestFacts]], bool],
                              with_jira_map: bool, **_,
                              ) -> Tuple[Dict[str, List[PullRequestFacts]], bool]:
    if with_jira_map and not result[1]:
        raise CancelCache()
    return result


@sentry_span
@cached(
    exptime=PullRequestMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repositories, participants, labels, jira, exclude_inactive, release_settings, fresh, with_jira_map, **_:  # noqa
    (
        time_from,
        time_to,
        ",".join(sorted(repositories)),
        ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(participants.items())),
        labels,
        jira,
        with_jira_map,
        exclude_inactive,
        release_settings,
        fresh,
    ),
    postprocess=_postprocess_cached_facts,
    version=2,
)
async def _calc_pull_request_facts_github(time_from: datetime,
                                          time_to: datetime,
                                          repositories: Set[str],
                                          participants: PRParticipants,
                                          labels: LabelFilter,
                                          jira: JIRAFilter,
                                          exclude_inactive: bool,
                                          release_settings: Dict[str, ReleaseMatchSetting],
                                          fresh: bool,
                                          with_jira_map: bool,
                                          account: int,
                                          meta_ids: Tuple[int, ...],
                                          mdb: Database,
                                          pdb: Database,
                                          rdb: Database,
                                          cache: Optional[aiomcache.Client],
                                          ) -> Tuple[List[PullRequestFacts], bool]:
    assert isinstance(repositories, set)
    branches, default_branches = await extract_branches(repositories, meta_ids, mdb, cache)
    precomputed_tasks = [
        load_precomputed_done_facts_filters(
            time_from, time_to, repositories, participants, labels,
            default_branches, exclude_inactive, release_settings, pdb),
    ]
    if exclude_inactive:
        precomputed_tasks.append(load_precomputed_done_candidates(
            time_from, time_to, repositories, default_branches, release_settings, pdb))
        (precomputed_facts, _), blacklist = await gather(*precomputed_tasks)
    else:
        (precomputed_facts, _) = blacklist = await precomputed_tasks[0]
    if with_jira_map:
        # schedule loading the PR->JIRA mapping
        done_jira_map_task = asyncio.create_task(append_pr_jira_mapping(
            precomputed_facts, meta_ids, mdb))
    ambiguous = blacklist[1]
    add_pdb_hits(pdb, "load_precomputed_done_facts_filters", len(precomputed_facts))

    prpk = PRParticipationKind
    if (len(precomputed_facts) > unfresh_prs_threshold
            or
            len(participants.get(prpk.AUTHOR, [])) > unfresh_participants_threshold) and \
            not fresh and not (participants.keys() - {prpk.AUTHOR, prpk.MERGER}):
        facts = await fetch_pull_request_facts_unfresh(
            precomputed_facts, ambiguous, time_from, time_to, repositories,
            participants, labels, jira, exclude_inactive, branches,
            default_branches, release_settings, account, meta_ids, mdb, pdb, rdb, cache)
        if with_jira_map:
            undone_jira_map_task = asyncio.create_task(append_pr_jira_mapping(
                {k: v for k, v in facts.items() if k not in precomputed_facts}, meta_ids, mdb))
        if with_jira_map:
            await gather(done_jira_map_task, undone_jira_map_task)
        return list(facts.values()), with_jira_map

    add_pdb_misses(pdb, "fresh", 1)
    date_from, date_to = coarsen_time_interval(time_from, time_to)
    # the adjacent out-of-range pieces [date_from, time_from] and [time_to, date_to]
    # are effectively discarded later in BinnedMetricCalculator
    tasks = [
        PullRequestMiner.mine(
            date_from, date_to, time_from, time_to, repositories, participants,
            labels, jira, branches, default_branches, exclude_inactive, release_settings,
            account, meta_ids, mdb, pdb, rdb, cache, pr_blacklist=blacklist),
    ]
    if jira and precomputed_facts:
        tasks.append(PullRequestMiner.filter_jira(
            precomputed_facts, jira, meta_ids, mdb, cache, columns=[PullRequest.node_id]))
        (miner, unreleased_facts, matched_bys, unreleased_prs_event), filtered = \
            await gather(*tasks, op="PullRequestMiner")
        precomputed_facts = {k: precomputed_facts[k] for k in filtered.index.values}
    else:
        miner, unreleased_facts, matched_bys, unreleased_prs_event = await tasks[0]
    precomputed_unreleased_prs = miner.drop(unreleased_facts)
    if with_jira_map:
        precomputed_unreleased_jira_map_task = asyncio.create_task(append_pr_jira_mapping(
            unreleased_facts, meta_ids, mdb))
        new_jira_map_task = load_pr_jira_mapping(miner.dfs.prs.index, meta_ids, mdb)
    await asyncio.sleep(0)
    remove_ambiguous_prs(precomputed_facts, ambiguous, matched_bys)
    add_pdb_hits(pdb, "precomputed_unreleased_facts", len(precomputed_unreleased_prs))
    for node_id in precomputed_unreleased_prs.values:
        precomputed_facts[node_id] = unreleased_facts[node_id]
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    mined_prs = []
    mined_facts = []
    open_pr_facts = []
    merged_unreleased_pr_facts = []
    done_count = 0
    with sentry_sdk.start_span(op="PullRequestMiner.__iter__ + PullRequestFactsMiner.__call__",
                               description=str(len(miner))):
        for i, pr in enumerate(miner):
            if (i + 1) % COROUTINE_YIELD_EVERY_ITER == 0:
                await asyncio.sleep(0)
            try:
                facts = facts_miner(pr)
            except ImpossiblePullRequest:
                continue
            mined_prs.append(pr)
            mined_facts.append(facts)
            if facts.done:
                done_count += 1
            elif not facts.closed:
                open_pr_facts.append((pr, facts))
            else:
                merged_unreleased_pr_facts.append((pr, facts))
    add_pdb_misses(pdb, "precomputed_done_facts", done_count)
    add_pdb_misses(pdb, "precomputed_open_facts", len(open_pr_facts))
    add_pdb_misses(pdb, "precomputed_merged_unreleased_facts", len(merged_unreleased_pr_facts))
    add_pdb_misses(pdb, "facts", len(miner))
    if done_count > 0:
        # we don't care if exclude_inactive is True or False here
        # also, we dump all the `mined_facts`, the called function will figure out who's released
        await defer(store_precomputed_done_facts(
            mined_prs, mined_facts, default_branches, release_settings, pdb),
            "store_precomputed_done_facts(%d/%d)" % (done_count, len(miner)))
    if len(open_pr_facts) > 0:
        await defer(store_open_pull_request_facts(open_pr_facts, pdb),
                    "store_open_pull_request_facts(%d)" % len(open_pr_facts))
    if len(merged_unreleased_pr_facts) > 0:
        await defer(store_merged_unreleased_pull_request_facts(
            merged_unreleased_pr_facts, matched_bys, default_branches, release_settings, pdb,
            unreleased_prs_event),
            "store_merged_unreleased_pull_request_facts(%d)" % len(merged_unreleased_pr_facts))
    if with_jira_map:
        _, _, new_jira_map = await gather(
            done_jira_map_task, precomputed_unreleased_jira_map_task, new_jira_map_task)
        for pr, facts in zip(mined_prs, mined_facts):
            facts.jira_id = new_jira_map.get(pr.pr[PullRequest.node_id.key])
    all_facts = list(chain(precomputed_facts.values(), mined_facts))
    return all_facts, with_jira_map


async def calc_pull_request_facts_github(time_from: datetime,
                                         time_to: datetime,
                                         repositories: Set[str],
                                         participants: PRParticipants,
                                         labels: LabelFilter,
                                         jira: JIRAFilter,
                                         exclude_inactive: bool,
                                         release_settings: Dict[str, ReleaseMatchSetting],
                                         fresh: bool,
                                         with_jira_map: bool,
                                         account: int,
                                         meta_ids: Tuple[int, ...],
                                         mdb: Database,
                                         pdb: Database,
                                         rdb: Database,
                                         cache: Optional[aiomcache.Client],
                                         ) -> List[PullRequestFacts]:
    """
    Calculate facts about pull request on GitHub.

    :param meta_ids: Metadata (GitHub) account IDs (*not the state DB account*) that own the repos.
    :param exclude_inactive: Do not load PRs without events between `time_from` and `time_to`.
    :param fresh: If the number of done PRs for the time period and filters exceeds \
                  `unfresh_mode_threshold`, force querying mdb instead of pdb only.
    :return: Map repository name -> list of PR facts.
    """
    return (await _calc_pull_request_facts_github(
        time_from,
        time_to,
        repositories,
        participants,
        labels,
        jira,
        exclude_inactive,
        release_settings,
        fresh,
        with_jira_map,
        account,
        meta_ids,
        mdb,
        pdb,
        rdb,
        cache,
    ))[0]


def _merge_repositories_and_participants(repositories: Sequence[Collection[str]],
                                         participants: List[PRParticipants],
                                         ) -> Tuple[Set[str], PRParticipants]:
    all_repositories = set(chain.from_iterable(repositories))
    if participants:
        all_participants = {}
        for k in PRParticipationKind:
            if kp := reduce(lambda x, y: x.union(y), [p.get(k, set()) for p in participants]):
                all_participants[k] = kp
    else:
        all_participants = {}
    return all_repositories, all_participants


def _compose_cache_key_repositories(repositories: Sequence[Collection[str]]) -> str:
    return ",".join(str(sorted(r)) for r in repositories)


def _compose_cache_key_participants(participants: List[PRParticipants]) -> str:
    return ";".join(",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(p.items()))
                    for p in participants)


@sentry_span
@cached(
    exptime=PullRequestMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda metrics, time_intervals, quantiles, repositories, participants, labels, jira, exclude_inactive, release_settings, **_:  # noqa
    (
        ",".join(sorted(metrics)),
        ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
        ",".join(str(q) for q in quantiles),
        _compose_cache_key_repositories(repositories),
        _compose_cache_key_participants(participants),
        labels, jira,
        exclude_inactive,
        release_settings,
    ),
)
async def calc_pull_request_metrics_line_github(metrics: Sequence[str],
                                                time_intervals: Sequence[Sequence[datetime]],
                                                quantiles: Sequence[float],
                                                lines: Sequence[int],
                                                repositories: Sequence[Collection[str]],
                                                participants: List[PRParticipants],
                                                labels: LabelFilter,
                                                jira: JIRAFilter,
                                                exclude_inactive: bool,
                                                release_settings: Dict[str, ReleaseMatchSetting],
                                                fresh: bool,
                                                account: int,
                                                meta_ids: Tuple[int, ...],
                                                mdb: Database,
                                                pdb: Database,
                                                rdb: Database,
                                                cache: Optional[aiomcache.Client],
                                                ) -> np.ndarray:
    """
    Calculate pull request metrics on GitHub.

    :return: lines x repositories x participants x granularities x time intervals x metrics.
    """
    assert isinstance(repositories, (tuple, list))
    all_repositories, all_participants = \
        _merge_repositories_and_participants(repositories, participants)
    calc = PullRequestBinnedMetricCalculator(metrics, quantiles, exclude_inactive=exclude_inactive)
    time_from, time_to = time_intervals[0][0], time_intervals[0][-1]
    mined_facts = await calc_pull_request_facts_github(
        time_from, time_to, all_repositories, all_participants, labels, jira, exclude_inactive,
        release_settings, fresh, need_jira_mapping(metrics),
        account, meta_ids, mdb, pdb, rdb, cache)
    df_facts = df_from_dataclasses(mined_facts)
    repo_grouper = partial(group_by_repo, PullRequest.repository_full_name.key, repositories)
    with_grouper = partial(group_prs_by_participants, participants)
    groups = group_to_indexes(df_facts, partial(group_by_lines, lines), repo_grouper, with_grouper)
    return calc(df_facts, time_intervals, groups)


@sentry_span
async def calc_code_metrics_github(prop: FilterCommitsProperty,
                                   time_intervals: Sequence[datetime],
                                   repos: Collection[str],
                                   with_author: Optional[Collection[str]],
                                   with_committer: Optional[Collection[str]],
                                   meta_ids: Tuple[int, ...],
                                   mdb: Database,
                                   cache: Optional[aiomcache.Client],
                                   ) -> List[CodeStats]:
    """Filter code pushed on GitHub according to the specified criteria."""
    time_from, time_to = time_intervals[0], time_intervals[-1]
    x_commits = await extract_commits(
        prop, time_from, time_to, repos, with_author, with_committer, meta_ids, mdb, cache)
    all_commits = await extract_commits(
        FilterCommitsProperty.NO_PR_MERGES, time_from, time_to, repos,
        with_author, with_committer, meta_ids, mdb, cache,
        columns=[PushCommit.committed_date, PushCommit.additions, PushCommit.deletions])
    return calc_code_stats(x_commits, all_commits, time_intervals)


@sentry_span
@cached(
    exptime=PullRequestMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda defs, time_from, time_to, quantiles, repositories, participants, labels, jira, exclude_inactive, release_settings, **_:  # noqa
    (
        ",".join("%s:%s" % (k, sorted(v)) for k, v in sorted(defs.items())),
        time_from, time_to,
        ",".join(str(q) for q in quantiles),
        _compose_cache_key_repositories(repositories),
        _compose_cache_key_participants(participants),
        labels, jira,
        exclude_inactive,
        release_settings,
    ),
)
async def calc_pull_request_histograms_github(defs: Dict[HistogramParameters, List[str]],
                                              time_from: datetime,
                                              time_to: datetime,
                                              quantiles: Sequence[float],
                                              lines: Sequence[int],
                                              repositories: Sequence[Collection[str]],
                                              participants: List[PRParticipants],
                                              labels: LabelFilter,
                                              jira: JIRAFilter,
                                              exclude_inactive: bool,
                                              release_settings: Dict[str, ReleaseMatchSetting],
                                              fresh: bool,
                                              account: int,
                                              meta_ids: Tuple[int, ...],
                                              mdb: Database,
                                              pdb: Database,
                                              rdb: Database,
                                              cache: Optional[aiomcache.Client],
                                              ) -> np.ndarray:
    """
    Calculate the pull request histograms on GitHub.

    :return: defs x lines x repositories x participants -> List[Tuple[metric ID, Histogram]].
    """
    all_repositories, all_participants = \
        _merge_repositories_and_participants(repositories, participants)
    try:
        calc = PullRequestBinnedHistogramCalculator(defs.values(), quantiles)
    except KeyError as e:
        raise ValueError("Unsupported metric") from e
    mined_facts = await calc_pull_request_facts_github(
        time_from, time_to, all_repositories, all_participants, labels, jira,
        exclude_inactive, release_settings, fresh, False, account, meta_ids, mdb, pdb, rdb, cache)
    df_facts = df_from_dataclasses(mined_facts)
    lines_grouper = partial(group_by_lines, lines)
    repo_grouper = partial(group_by_repo, PullRequest.repository_full_name.key, repositories)
    with_grouper = partial(group_prs_by_participants, participants)
    groups = group_to_indexes(df_facts, lines_grouper, repo_grouper, with_grouper)
    hists = calc(df_facts, [[time_from, time_to]], groups, defs)
    reshaped = np.full(hists.shape[:-1], None, object)
    reshaped_seq = reshaped.ravel()
    pos = 0
    for line_groups, metrics in zip(hists, defs.values()):
        for repo_groups in line_groups:
            for participants_groups in repo_groups:
                for group_ts in participants_groups:
                    reshaped_seq[pos] = [(m, hist) for hist, m in zip(group_ts[0][0], metrics)]
                    pos += 1
    return reshaped


@sentry_span
@cached(
    exptime=5 * 60,  # 5 min
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda metrics, time_intervals, quantiles, repositories, participants, jira, release_settings, **_:  # noqa
    (
        ",".join(sorted(metrics)),
        ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
        ",".join(str(q) for q in quantiles),
        ",".join(str(sorted(r)) for r in repositories),
        ";".join(",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(p.items()))
                 for p in participants),
        jira,
        release_settings,
    ),
)
async def calc_release_metrics_line_github(metrics: Sequence[str],
                                           time_intervals: Sequence[Sequence[datetime]],
                                           quantiles: Sequence[float],
                                           repositories: Sequence[Collection[str]],
                                           participants: List[ReleaseParticipants],
                                           jira: JIRAFilter,
                                           release_settings: Dict[str, ReleaseMatchSetting],
                                           account: int,
                                           meta_ids: Tuple[int, ...],
                                           mdb: Database,
                                           pdb: Database,
                                           rdb: Database,
                                           cache: Optional[aiomcache.Client],
                                           ) -> Tuple[np.ndarray, Dict[str, ReleaseMatch]]:
    """
    Calculate the release metrics on GitHub.

    :return: 1. participants x repositories x granularities x time intervals x metrics.
             2. matched_bys - map from repository names to applied release matches.
    """
    time_from, time_to = time_intervals[0][0], time_intervals[0][-1]
    all_repositories = set(chain.from_iterable(repositories))
    calc = ReleaseBinnedMetricCalculator(metrics, quantiles)
    branches, default_branches = await extract_branches(all_repositories, meta_ids, mdb, cache)
    all_participants = merge_release_participants(participants)
    releases, _, matched_bys = await mine_releases(
        all_repositories, all_participants, branches, default_branches,
        time_from, time_to, jira, release_settings, account, meta_ids, mdb, pdb, rdb, cache)
    df_facts = df_from_dataclasses([f for _, f in releases])
    repo_grouper = partial(group_by_repo, Release.repository_full_name.key, repositories)
    participant_grouper = partial(group_releases_by_participants, participants)
    groups = group_to_indexes(df_facts, participant_grouper, repo_grouper)
    values = calc(df_facts, time_intervals, groups)
    return values, matched_bys


@sentry_span
@sentry_span
@cached(
    exptime=5 * 60,  # 5 min
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda devs, repositories, time_intervals, topics, labels, jira, release_settings, **_:  # noqa
    (
        ",".join(t.value for t in sorted(topics)),
        ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
        _compose_cache_key_repositories(repositories),
        _compose_cache_key_repositories(devs),  # yes, _repositories
        labels, jira,
        release_settings,
    ),
)
async def calc_developer_metrics_github(devs: Sequence[Collection[str]],
                                        repositories: Sequence[Collection[str]],
                                        time_intervals: Sequence[Sequence[datetime]],
                                        topics: Set[DeveloperTopic],
                                        labels: LabelFilter,
                                        jira: JIRAFilter,
                                        release_settings: Dict[str, ReleaseMatchSetting],
                                        account: int,
                                        meta_ids: Tuple[int, ...],
                                        mdb: Database,
                                        pdb: Database,
                                        rdb: Database,
                                        cache: Optional[aiomcache.Client],
                                        ) -> Tuple[np.ndarray, List[DeveloperTopic]]:
    """
    Calculate the developer metrics on GitHub.

    :return: repositories x granularities x devs x time intervals x topics.
    """
    all_devs = set(chain.from_iterable(devs))
    all_repos = set(chain.from_iterable(repositories))
    time_from, time_to = time_intervals[0][0], time_intervals[0][-1]
    mined_dfs = await mine_developer_activities(
        all_devs, all_repos, time_from, time_to, topics, labels, jira, release_settings,
        account, meta_ids, mdb, pdb, rdb, cache)
    topics_seq = []
    arrays = []
    repo_grouper = partial(group_by_repo, developer_repository_column, repositories)
    developer_grouper = partial(group_actions_by_developers, devs)
    for mined_topics, mined_df in mined_dfs:
        topics_seq.extend(mined_topics)
        calc = DeveloperBinnedMetricCalculator([t.value for t in mined_topics], (0, 1))
        groups = group_to_indexes(mined_df, repo_grouper, developer_grouper)
        arrays.append(calc(mined_df, time_intervals, groups))
    result = np.full(arrays[0].shape, None)
    result.ravel()[:] = [
        [list(chain.from_iterable(m)) for m in zip(*lists)]
        for lists in zip(*(arr.ravel() for arr in arrays))
    ]
    result = result.swapaxes(1, 2)
    return result, topics_seq


METRIC_ENTRIES = {
    "github": {
        "prs_linear": calc_pull_request_metrics_line_github,
        "prs_histogram": calc_pull_request_histograms_github,
        "code": calc_code_metrics_github,
        "developers": calc_developer_metrics_github,
        "releases_linear": calc_release_metrics_line_github,
    },
}
