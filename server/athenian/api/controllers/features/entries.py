import asyncio
from datetime import datetime
import pickle
from typing import Collection, Dict, List, Optional, Sequence, Set

import aiomcache
from databases import Database
import sentry_sdk

from athenian.api.cache import cached
from athenian.api.controllers.datetime_utils import coarsen_time_interval
from athenian.api.controllers.features.code import CodeStats
from athenian.api.controllers.features.github.code import calc_code_stats
from athenian.api.controllers.features.github.pull_request import \
    BinnedPullRequestMetricCalculator, calculators as pull_request_calculators
import athenian.api.controllers.features.github.pull_request_metrics  # noqa
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.commit import extract_commits, FilterCommitsProperty
from athenian.api.controllers.miners.github.developer import calc_developer_metrics
from athenian.api.controllers.miners.github.precomputed_prs import \
    load_precomputed_done_candidates, load_precomputed_done_times, store_precomputed_done_times
from athenian.api.controllers.miners.github.pull_request import ImpossiblePullRequest, \
    PullRequestMiner, PullRequestTimesMiner
from athenian.api.controllers.miners.types import Participants
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.models.metadata.github import PushCommit
from athenian.api.tracing import sentry_span


@sentry_span
@cached(
    exptime=PullRequestMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda metrics, time_intervals, repositories, participants, exclude_inactive, release_settings, **_:  # noqa
    (
        ",".join(sorted(metrics)),
        ";".join(",".join(str(dt.timestamp()) for dt in ts) for ts in time_intervals),
        ",".join(sorted(repositories)),
        ",".join("%s:%s" % (k.name, sorted(v)) for k, v in sorted(participants.items())),
        exclude_inactive,
        release_settings,
    ),
)
async def calc_pull_request_metrics_line_github(metrics: Collection[str],
                                                time_intervals: Sequence[Sequence[datetime]],
                                                repositories: Set[str],
                                                participants: Participants,
                                                exclude_inactive: bool,
                                                release_settings: Dict[str, ReleaseMatchSetting],
                                                mdb: Database,
                                                pdb: Database,
                                                cache: Optional[aiomcache.Client],
                                                ) -> List[List[List[Metric]]]:
    """Calculate pull request metrics on GitHub data."""
    assert isinstance(repositories, set)
    time_from, time_to = time_intervals[0][0], time_intervals[0][-1]
    branches, default_branches = await extract_branches(repositories, mdb, cache)
    precomputed_tasks = [
        load_precomputed_done_times(
            time_from, time_to, repositories, participants, default_branches, exclude_inactive,
            release_settings, pdb),
    ]
    if exclude_inactive:
        precomputed_tasks.append(load_precomputed_done_candidates(
            time_from, time_to, repositories, default_branches, release_settings, pdb))
        done_times, blacklist = await asyncio.gather(*precomputed_tasks, return_exceptions=True)
        for r in (done_times, blacklist):
            if isinstance(r, Exception):
                raise r from None
    else:
        done_times = blacklist = await precomputed_tasks[0]
    pdb.metrics["hits"].get()["load_precomputed_done_times"] += len(done_times)

    date_from, date_to = coarsen_time_interval(time_from, time_to)
    # the adjacent out-of-range pieces [date_from, time_from] and [time_to, date_to]
    # are effectively discarded later in BinnedPullRequestMetricCalculator
    miner = await PullRequestMiner.mine(
        date_from, date_to, time_from, time_to, repositories, participants,
        branches, default_branches, exclude_inactive, release_settings,
        mdb, pdb, cache, pr_blacklist=blacklist)
    times_miner = PullRequestTimesMiner()
    mined_prs = []
    mined_times = []
    with sentry_sdk.start_span(op="PullRequestMiner.__iter__ + PullRequestTimesMiner.__call__"):
        for pr in miner:
            mined_prs.append(pr)
            try:
                times = times_miner(pr)
            except ImpossiblePullRequest:
                continue
            mined_times.append(times)
    pdb.metrics["misses"].get()["load_precomputed_done_times"] += len(mined_times)
    # we don't care if exclude_inactive is True or False here
    await store_precomputed_done_times(
        mined_prs, mined_times, default_branches, release_settings, pdb)
    mined_times.extend(done_times.values())
    with sentry_sdk.start_span(op="BinnedPullRequestMetricCalculator.__call__"):
        calcs = [pull_request_calculators[m]() for m in metrics]
        return [BinnedPullRequestMetricCalculator(calcs, ts)(mined_times) for ts in time_intervals]


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


METRIC_ENTRIES = {
    "github": {
        **{k: calc_pull_request_metrics_line_github for k in pull_request_calculators},
        "code": calc_code_metrics,
        "developers": calc_developer_metrics,
    },
}
