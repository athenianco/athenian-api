from datetime import date, datetime, timezone
import pickle
from typing import List, Mapping, Optional, Sequence, Tuple

import aiomcache
from databases import Database

from athenian.api.cache import cached
from athenian.api.controllers.features.github.pull_request import \
    BinnedPullRequestMetricCalculator, calculators as pull_request_calculators
import athenian.api.controllers.features.github.pull_request_metrics  # noqa
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.miners.github.pull_request import PullRequestListMiner, \
    PullRequestTimesMiner
from athenian.api.controllers.miners.pull_request_list_item import ParticipationKind, Property, \
    PullRequestListItem
from athenian.api.models.web.pull_request_participant import PullRequestParticipant


async def calc_metrics_line_func(
        metrics: Sequence[str], time_intervals: Sequence[date], repos: Sequence[str],
        developers: Sequence[str], db: Database, cache: Optional[aiomcache.Client],
) -> List[Tuple[Metric]]:
    """All the metric calculators must follow this call signature."""
    raise NotImplementedError


@cached(
    exptime=PullRequestTimesMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda metrics, time_intervals, repos, developers, **_: (
        ",".join(sorted(metrics)),
        ",".join(str(dt.toordinal()) for dt in time_intervals),
        ",".join(sorted(repos)),
        ",".join(sorted(developers)),
    ),
)
async def calc_pull_request_metrics_line_github(
        metrics: Sequence[str], time_intervals: Sequence[date], repos: Sequence[str],
        developers: Sequence[str], db: Database, cache: Optional[aiomcache.Client],
) -> List[Tuple[Metric]]:
    """Calculate pull request metrics on GitHub data."""
    miner = await PullRequestTimesMiner.mine(
        time_intervals[0], time_intervals[-1], repos, developers, db, cache)
    calcs = [pull_request_calculators[m]() for m in metrics]
    binned = BinnedPullRequestMetricCalculator(calcs, time_intervals)
    return binned(miner)


async def filter_pull_requests_func(
        time_from: date, time_to: date, repos: Sequence[str],
        properties: Sequence[Property], participants: Mapping[ParticipationKind, Sequence[str]],
        db: Database, cache: Optional[aiomcache.Client],
) -> List[PullRequestListItem]:
    """All the pull request filters must follow this call signature."""
    raise NotImplementedError


@cached(
    exptime=PullRequestListMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repos, properties, participants, **_: (
        time_from.toordinal(),
        time_to.toordinal(),
        ",".join(sorted(repos)),
        ",".join(s.name.lower() for s in sorted(set(properties))),
        sorted((k.name.lower(), sorted(set(v))) for k, v in participants.items()),
    ),
)
async def filter_pull_requests_github(
        time_from: date, time_to: date, repos: Sequence[str],
        properties: Sequence[Property], participants: Mapping[ParticipationKind, Sequence[str]],
        db: Database, cache: Optional[aiomcache.Client],
) -> List[PullRequestListItem]:
    """Filter GitHub pull requests according to the specified criteria."""
    miner = await PullRequestListMiner.mine(
        time_from, time_to, repos, participants.get(PullRequestParticipant.STATUS_AUTHOR, []),
        db, cache)
    miner.properties = properties
    miner.participants = participants
    miner.time_from = datetime(time_from.year, time_from.month, time_from.day,
                               tzinfo=timezone.utc)
    return list(miner)


METRIC_ENTRIES = {
    "github": {
        # there will be other metrics in the future, hence **{}
        **{k: calc_pull_request_metrics_line_github for k in pull_request_calculators},
    },
}  # type: Mapping[str, Mapping[str, calc_metrics_line_func]]

PR_ENTRIES = {
    "github": filter_pull_requests_github,
}  # type: Mapping[str, filter_pull_requests_func]
