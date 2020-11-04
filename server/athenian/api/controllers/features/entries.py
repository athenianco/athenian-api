import asyncio
from collections import defaultdict
from datetime import datetime
from itertools import chain
import pickle
from typing import Collection, Dict, List, Optional, Sequence, Set, Tuple

import aiomcache
from databases import Database
import sentry_sdk

from athenian.api import COROUTINE_YIELD_EVERY_ITER
from athenian.api.async_utils import gather
from athenian.api.cache import cached
from athenian.api.controllers.datetime_utils import coarsen_time_interval
from athenian.api.controllers.features.code import CodeStats
from athenian.api.controllers.features.github.code import calc_code_stats
from athenian.api.controllers.features.github.pull_request_metrics import \
    PullRequestBinnedHistogramCalculator, PullRequestBinnedMetricCalculator
from athenian.api.controllers.features.github.release_metrics import \
    ReleaseBinnedMetricCalculator
from athenian.api.controllers.features.github.unfresh_pull_request_metrics import \
    fetch_pull_request_facts_unfresh
from athenian.api.controllers.features.histogram import Histogram, HistogramParameters
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.bots import bots
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.commit import extract_commits, FilterCommitsProperty
from athenian.api.controllers.miners.github.developer import calc_developer_metrics_github
from athenian.api.controllers.miners.github.precomputed_prs import \
    load_precomputed_done_candidates, load_precomputed_done_facts_filters, \
    store_merged_unreleased_pull_request_facts, store_open_pull_request_facts, \
    store_precomputed_done_facts
from athenian.api.controllers.miners.github.pull_request import ImpossiblePullRequest, \
    PullRequestFactsMiner, PullRequestMiner
from athenian.api.controllers.miners.github.release import mine_releases
from athenian.api.controllers.miners.types import PRParticipants, PRParticipationKind, \
    PullRequestFacts, ReleaseParticipants
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting
from athenian.api.db import add_pdb_hits, add_pdb_misses
from athenian.api.defer import defer
from athenian.api.models.metadata.github import PullRequest, PushCommit, Release
from athenian.api.tracing import sentry_span


unfresh_prs_threshold = 1000
unfresh_participants_threshold = 50


@sentry_span
@cached(
    exptime=PullRequestMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repositories, participants, labels, jira, exclude_inactive, release_settings, fresh, **_:  # noqa
    (
        time_from,
        time_to,
        ",".join(sorted(repositories)),
        ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(participants.items())),
        labels,
        jira,
        exclude_inactive,
        release_settings,
        fresh,
    ),
    version=2,
)
async def calc_pull_request_facts_github(time_from: datetime,
                                         time_to: datetime,
                                         repositories: Set[str],
                                         participants: PRParticipants,
                                         labels: LabelFilter,
                                         jira: JIRAFilter,
                                         exclude_inactive: bool,
                                         release_settings: Dict[str, ReleaseMatchSetting],
                                         fresh: bool,
                                         mdb: Database,
                                         pdb: Database,
                                         cache: Optional[aiomcache.Client],
                                         ) -> Dict[str, List[PullRequestFacts]]:
    """
    Calculate facts about pull request on GitHub.

    :param fresh: If the number of done PRs for the time period and filters exceeds \
                  `unfresh_mode_threshold`, force querying mdb instead of pdb only.
    :return: Map repository name -> list of PR facts.
    """
    assert isinstance(repositories, set)
    branches, default_branches = await extract_branches(repositories, mdb, cache)
    precomputed_tasks = [
        load_precomputed_done_facts_filters(
            time_from, time_to, repositories, participants, labels,
            default_branches, exclude_inactive, release_settings, pdb),
    ]
    if exclude_inactive:
        precomputed_tasks.append(load_precomputed_done_candidates(
            time_from, time_to, repositories, default_branches, release_settings, pdb))
        precomputed_facts, blacklist = await gather(*precomputed_tasks)
    else:
        precomputed_facts = blacklist = await precomputed_tasks[0]
    add_pdb_hits(pdb, "load_precomputed_done_facts_filters", len(precomputed_facts))

    prpk = PRParticipationKind
    if (len(precomputed_facts) > unfresh_prs_threshold
            or
            len(participants.get(prpk.AUTHOR, [])) > unfresh_participants_threshold) and \
            not fresh and not (participants.keys() - {prpk.AUTHOR, prpk.MERGER}):
        return await fetch_pull_request_facts_unfresh(
            precomputed_facts, time_from, time_to, repositories, participants, labels, jira,
            exclude_inactive, branches, default_branches, release_settings, mdb, pdb, cache)

    add_pdb_misses(pdb, "fresh", 1)
    date_from, date_to = coarsen_time_interval(time_from, time_to)
    # the adjacent out-of-range pieces [date_from, time_from] and [time_to, date_to]
    # are effectively discarded later in BinnedMetricCalculator
    tasks = [
        PullRequestMiner.mine(
            date_from, date_to, time_from, time_to, repositories, participants, labels, jira,
            branches, default_branches, exclude_inactive, release_settings,
            mdb, pdb, cache, pr_blacklist=blacklist),
    ]
    if jira and precomputed_facts:
        tasks.append(PullRequestMiner.filter_jira(
            precomputed_facts, jira, mdb, cache, columns=[PullRequest.node_id]))
        (miner, unreleased_facts, matched_bys, unreleased_prs_event), filtered = \
            await gather(*tasks, op="PullRequestMiner")
        precomputed_facts = {k: precomputed_facts[k] for k in filtered.index.values}
    else:
        miner, unreleased_facts, matched_bys, unreleased_prs_event = await tasks[0]
    precomputed_unreleased_prs = miner.drop(unreleased_facts)
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
            mined_facts.append((pr.pr[PullRequest.repository_full_name.key], facts))
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
    by_repo = {}
    for repo, f in chain(precomputed_facts.values(), mined_facts):
        try:
            by_repo[repo].append(f)
        except KeyError:
            by_repo[repo] = [f]
    return by_repo


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
        ",".join(str(sorted(r)) for r in repositories),
        ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(participants.items())),
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
                                                participants: PRParticipants,
                                                labels: LabelFilter,
                                                jira: JIRAFilter,
                                                exclude_inactive: bool,
                                                release_settings: Dict[str, ReleaseMatchSetting],
                                                fresh: bool,
                                                mdb: Database,
                                                pdb: Database,
                                                cache: Optional[aiomcache.Client],
                                                ) -> List[List[List[List[List[Metric]]]]]:
    """
    Calculate pull request metrics on GitHub.

    :return: lines x granularities x groups x time intervals x metrics.
    """
    assert isinstance(repositories, (tuple, list))
    all_repositories = set(chain.from_iterable(repositories))
    calc = PullRequestBinnedMetricCalculator(
        metrics, quantiles, lines, exclude_inactive=exclude_inactive)
    time_from, time_to = time_intervals[0][0], time_intervals[0][-1]
    mined_facts = await calc_pull_request_facts_github(
        time_from, time_to, all_repositories, participants, labels, jira, exclude_inactive,
        release_settings, fresh, mdb, pdb, cache)
    return calc(mined_facts, time_intervals, repositories)


@sentry_span
async def calc_code_metrics_github(prop: FilterCommitsProperty,
                                   time_intervals: Sequence[datetime],
                                   repos: Collection[str],
                                   with_author: Optional[Collection[str]],
                                   with_committer: Optional[Collection[str]],
                                   db: Database,
                                   cache: Optional[aiomcache.Client],
                                   ) -> List[CodeStats]:
    """Filter code pushed on GitHub according to the specified criteria."""
    time_from, time_to = time_intervals[0], time_intervals[-1]
    x_commits = await extract_commits(
        prop, time_from, time_to, repos, with_author, with_committer, db, cache)
    all_commits = await extract_commits(
        FilterCommitsProperty.NO_PR_MERGES, time_from, time_to, repos, with_author, with_committer,
        db, cache, columns=[PushCommit.committed_date, PushCommit.additions, PushCommit.deletions])
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
        ",".join(str(sorted(r)) for r in repositories),
        ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(participants.items())),
        labels, jira,
        exclude_inactive,
        release_settings,
    ),
)
async def calc_pull_request_histogram_github(defs: Dict[HistogramParameters, List[str]],
                                             time_from: datetime,
                                             time_to: datetime,
                                             quantiles: Sequence[float],
                                             lines: Sequence[int],
                                             repositories: Sequence[Collection[str]],
                                             participants: PRParticipants,
                                             labels: LabelFilter,
                                             jira: JIRAFilter,
                                             exclude_inactive: bool,
                                             release_settings: Dict[str, ReleaseMatchSetting],
                                             fresh: bool,
                                             mdb: Database,
                                             pdb: Database,
                                             cache: Optional[aiomcache.Client],
                                             ) -> List[List[Tuple[str, Histogram]]]:
    """Calculate the pull request histograms on GitHub."""
    all_repositories = set(chain.from_iterable(repositories))
    try:
        calc = PullRequestBinnedHistogramCalculator(defs.values(), quantiles, lines)
    except KeyError as e:
        raise ValueError("Unsupported metric: %s" % e)
    mined_facts = await calc_pull_request_facts_github(
        time_from, time_to, all_repositories, participants, labels, jira, exclude_inactive,
        release_settings, fresh, mdb, pdb, cache)
    hists = calc(mined_facts, [[time_from, time_to]], repositories, [k.__dict__ for k in defs])
    result = [[] for _ in range(len(repositories) * (len(lines or [None] * 2) - 1))]
    for defs_hists, metrics in zip(hists, defs.values()):
        for group_result, group_hists in zip(result, defs_hists[0]):
            for hist, m in zip(group_hists[0], metrics):
                group_result.append((m, hist))
    return result


@sentry_span
@cached(
    exptime=5 * 60,  # 5 min
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda metrics, time_intervals, quantiles, repositories, participants, release_settings, **_:  # noqa
    (
        ",".join(sorted(metrics)),
        ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
        ",".join(str(q) for q in quantiles),
        ",".join(str(sorted(r)) for r in repositories),
        ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(participants.items())),
        release_settings,
    ),
)
async def calc_release_metrics_line_github(metrics: Sequence[str],
                                           time_intervals: Sequence[Sequence[datetime]],
                                           quantiles: Sequence[float],
                                           repositories: Sequence[Collection[str]],
                                           participants: ReleaseParticipants,
                                           jira: JIRAFilter,
                                           release_settings: Dict[str, ReleaseMatchSetting],
                                           mdb: Database,
                                           pdb: Database,
                                           cache: Optional[aiomcache.Client],
                                           ) -> Tuple[List[List[List[List[Metric]]]],
                                                      Dict[str, ReleaseMatch]]:
    """Calculate the release metrics on GitHub."""
    time_from, time_to = time_intervals[0][0], time_intervals[0][-1]
    all_repositories = set(chain.from_iterable(repositories))
    calc = ReleaseBinnedMetricCalculator(metrics, quantiles)
    branches, default_branches = await extract_branches(all_repositories, mdb, cache)
    releases, _, matched_bys = await mine_releases(
        all_repositories, participants, branches, default_branches, time_from, time_to, jira,
        release_settings, mdb, pdb, cache)
    mined_facts = defaultdict(list)
    for i, f in releases:
        mined_facts[i[Release.repository_full_name.key].split("/", 1)[1]].append(f)
    values = calc(mined_facts, time_intervals, repositories)
    return values, matched_bys


METRIC_ENTRIES = {
    "github": {
        "prs_linear": calc_pull_request_metrics_line_github,
        "prs_histogram": calc_pull_request_histogram_github,
        "code": calc_code_metrics_github,
        "developers": calc_developer_metrics_github,
        "releases_linear": calc_release_metrics_line_github,
    },
}
