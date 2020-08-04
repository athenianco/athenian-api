import asyncio
from datetime import datetime
import pickle
from typing import Collection, Dict, List, Optional, Sequence, Set

import aiomcache
from databases import Database
import sentry_sdk

from athenian.api import COROUTINE_YIELD_EVERY_ITER
from athenian.api.cache import cached
from athenian.api.controllers.datetime_utils import coarsen_time_interval
from athenian.api.controllers.features.code import CodeStats
from athenian.api.controllers.features.github.code import calc_code_stats
from athenian.api.controllers.features.github.pull_request import \
    BinnedPullRequestMetricCalculator, histogram_calculators, metric_calculators, \
    PullRequestHistogramCalculatorEnsemble
import athenian.api.controllers.features.github.pull_request_metrics  # noqa
from athenian.api.controllers.features.histogram import Histogram, Scale
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.miners.github.bots import bots
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.commit import extract_commits, FilterCommitsProperty
from athenian.api.controllers.miners.github.developer import calc_developer_metrics
from athenian.api.controllers.miners.github.precomputed_prs import \
    load_precomputed_done_candidates, load_precomputed_done_facts_filters, \
    store_open_pull_request_facts, store_precomputed_done_facts
from athenian.api.controllers.miners.github.pull_request import ImpossiblePullRequest, \
    PullRequestFactsMiner, PullRequestMiner
from athenian.api.controllers.miners.types import Participants, PullRequestFacts
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.db import add_pdb_hits, add_pdb_misses
from athenian.api.defer import defer
from athenian.api.models.metadata.github import PushCommit
from athenian.api.tracing import sentry_span


@sentry_span
@cached(
    exptime=PullRequestMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repositories, participants, labels, exclude_inactive, release_settings, **_:  # noqa
    (
        time_from,
        time_to,
        ",".join(sorted(repositories)),
        ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(participants.items())),
        ",".join(sorted(labels)),
        exclude_inactive,
        release_settings,
    ),
)
async def calc_pull_request_facts_github(time_from: datetime,
                                         time_to: datetime,
                                         repositories: Set[str],
                                         participants: Participants,
                                         labels: Set[str],
                                         exclude_inactive: bool,
                                         release_settings: Dict[str, ReleaseMatchSetting],
                                         mdb: Database,
                                         pdb: Database,
                                         cache: Optional[aiomcache.Client],
                                         ) -> List[PullRequestFacts]:
    """Calculate the pull request timestamps on GitHub."""
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
        precomputed_facts, blacklist = await asyncio.gather(
            *precomputed_tasks, return_exceptions=True)
        for r in (precomputed_facts, blacklist):
            if isinstance(r, Exception):
                raise r from None
    else:
        precomputed_facts = blacklist = await precomputed_tasks[0]
    add_pdb_hits(pdb, "load_precomputed_done_facts_filters", len(precomputed_facts))

    date_from, date_to = coarsen_time_interval(time_from, time_to)
    # the adjacent out-of-range pieces [date_from, time_from] and [time_to, date_to]
    # are effectively discarded later in BinnedPullRequestMetricCalculator
    miner, open_facts = await PullRequestMiner.mine(
        date_from, date_to, time_from, time_to, repositories, participants, labels,
        branches, default_branches, exclude_inactive, release_settings,
        mdb, pdb, cache, pr_blacklist=blacklist)
    precomputed_open_prs = miner.drop(open_facts)
    add_pdb_hits(pdb, "load_open_pull_request_facts", len(precomputed_open_prs))
    for node_id in precomputed_open_prs.values:
        precomputed_facts[node_id] = open_facts[node_id]
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    mined_prs = []
    mined_facts = list(precomputed_facts.values())
    open_pr_facts = []
    done_count = 0
    with sentry_sdk.start_span(op="PullRequestMiner.__iter__ + PullRequestFactsMiner.__call__",
                               description=str(len(miner))):
        for i, pr in enumerate(miner):
            if (i + 1) % COROUTINE_YIELD_EVERY_ITER == 0:
                await asyncio.sleep(0)
            mined_prs.append(pr)
            try:
                facts = facts_miner(pr)
            except ImpossiblePullRequest:
                mined_facts.append(None)  # to match the order of mined_prs
                continue
            mined_facts.append(facts)
            if facts.released or (facts.closed and not facts.merged):
                done_count += 1
            if not facts.closed:
                open_pr_facts.append((pr.pr, facts))
    add_pdb_misses(pdb, "load_precomputed_done_facts_filters", done_count)
    add_pdb_misses(pdb, "load_open_pull_request_facts", len(open_pr_facts))
    add_pdb_misses(pdb, "facts", len(miner))
    # we don't care if exclude_inactive is True or False here
    await defer(store_precomputed_done_facts(
        mined_prs, mined_facts[len(precomputed_facts):], default_branches, release_settings, pdb),
        "store_precomputed_done_facts(%d/%d)" % (done_count, len(miner)))
    await defer(store_open_pull_request_facts(open_pr_facts, pdb),
                "store_open_pull_request_facts(%d)" % len(open_pr_facts))
    return mined_facts


@sentry_span
@cached(
    exptime=PullRequestMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda metrics, time_intervals, repositories, participants, labels, exclude_inactive, release_settings, **_:  # noqa
    (
        ",".join(sorted(metrics)),
        ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
        ",".join(sorted(repositories)),
        ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(participants.items())),
        ",".join(sorted(labels)),
        exclude_inactive,
        release_settings,
    ),
)
async def calc_pull_request_metrics_line_github(metrics: Sequence[str],
                                                time_intervals: Sequence[Sequence[datetime]],
                                                repositories: Set[str],
                                                participants: Participants,
                                                labels: Set[str],
                                                exclude_inactive: bool,
                                                release_settings: Dict[str, ReleaseMatchSetting],
                                                mdb: Database,
                                                pdb: Database,
                                                cache: Optional[aiomcache.Client],
                                                ) -> List[List[List[Metric]]]:
    """Calculate pull request metrics on GitHub."""
    time_from, time_to = time_intervals[0][0], time_intervals[0][-1]
    mined_facts = await calc_pull_request_facts_github(
        time_from, time_to, repositories, participants, labels, exclude_inactive,
        release_settings, mdb, pdb, cache)
    with sentry_sdk.start_span(op="BinnedPullRequestMetricCalculator.__call__",
                               description=str(len(mined_facts))):
        return [BinnedPullRequestMetricCalculator(metrics, ts)(mined_facts)
                for ts in time_intervals]


@sentry_span
async def calc_code_metrics(prop: FilterCommitsProperty,
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
async def calc_pull_request_histogram_github(metrics: Sequence[str],
                                             scale: Scale,
                                             bins: int,
                                             time_from: datetime,
                                             time_to: datetime,
                                             repositories: Set[str],
                                             participants: Participants,
                                             labels: Set[str],
                                             exclude_inactive: bool,
                                             release_settings: Dict[str, ReleaseMatchSetting],
                                             mdb: Database,
                                             pdb: Database,
                                             cache: Optional[aiomcache.Client],
                                             ) -> List[Histogram]:
    """Calculate the pull request histograms on GitHub."""
    mined_facts = await calc_pull_request_facts_github(
        time_from, time_to, repositories, participants, labels, exclude_inactive,
        release_settings, mdb, pdb, cache)
    ensemble = PullRequestHistogramCalculatorEnsemble(*metrics)
    for facts in mined_facts:
        ensemble(facts, time_from, time_to)
    histograms = ensemble.histograms(scale, bins)
    histograms = [histograms[m] for m in metrics]
    return histograms


METRIC_ENTRIES = {
    "github": {
        "linear": {k: calc_pull_request_metrics_line_github for k in metric_calculators},
        "histogram": {k: calc_pull_request_histogram_github for k in histogram_calculators},
        "code": calc_code_metrics,
        "developers": calc_developer_metrics,
    },
}
