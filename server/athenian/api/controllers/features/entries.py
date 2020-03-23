from datetime import date, timezone
import pickle
from typing import Collection, List, Mapping, Optional, Sequence, Tuple, Union

import aiomcache
from databases import Database
import pandas as pd

from athenian.api.cache import cached
from athenian.api.controllers.features.code import calc_code_stats, CodeStats
from athenian.api.controllers.features.github.pull_request import \
    BinnedPullRequestMetricCalculator, calculators as pull_request_calculators
import athenian.api.controllers.features.github.pull_request_metrics  # noqa
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.miners.github.commit import extract_commits, FilterCommitsProperty
from athenian.api.controllers.miners.github.pull_request import PullRequestListMiner, \
    PullRequestTimesMiner
from athenian.api.controllers.miners.pull_request_list_item import ParticipationKind, Property, \
    PullRequestListItem
from athenian.api.models.metadata.github import PushCommit
from athenian.api.models.web.pull_request_participant import PullRequestParticipant


async def calc_metrics_line_func(
        metrics: Collection[str], time_intervals: Collection[date], repos: Collection[str],
        developers: Collection[str], db: Database, cache: Optional[aiomcache.Client],
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
        metrics: Collection[str], time_intervals: Sequence[date], repos: Collection[str],
        developers: Collection[str], db: Database, cache: Optional[aiomcache.Client],
) -> List[Tuple[Metric]]:
    """Calculate pull request metrics on GitHub data."""
    miner = await PullRequestTimesMiner.mine(
        time_intervals[0], time_intervals[-1], repos, developers, db, cache)
    calcs = [pull_request_calculators[m]() for m in metrics]
    binned = BinnedPullRequestMetricCalculator(calcs, time_intervals)
    return binned(miner)


async def filter_pull_requests_func(
        properties: Collection[Property], time_from: date, time_to: date, repos: Collection[str],
        participants: Mapping[ParticipationKind, Collection[str]],
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
        properties: Collection[Property], time_from: date, time_to: date, repos: Collection[str],
        participants: Mapping[ParticipationKind, Collection[str]],
        db: Database, cache: Optional[aiomcache.Client],
) -> List[PullRequestListItem]:
    """Filter GitHub pull requests according to the specified criteria."""
    miner = await PullRequestListMiner.mine(
        time_from, time_to, repos, participants.get(PullRequestParticipant.STATUS_AUTHOR, []),
        db, cache)
    miner.properties = properties
    miner.participants = participants
    miner.time_from = pd.Timestamp(time_from, tzinfo=timezone.utc)
    return list(miner)


async def filter_code_func(
        prop: FilterCommitsProperty, time_intervals: Collection[date], repos: Collection[str],
        with_author: Optional[Collection[str]], with_committer: Optional[Collection[str]],
        db: Database, cache: Optional[aiomcache.Client],
) -> List[CodeStats]:
    """All the code filters must follow this call signature."""
    raise NotImplementedError


async def filter_code_github(
        prop: FilterCommitsProperty, time_intervals: Sequence[date], repos: Collection[str],
        with_author: Optional[Collection[str]], with_committer: Optional[Collection[str]],
        db: Database, cache: Optional[aiomcache.Client],
) -> List[CodeStats]:
    """Filter code pushed on GitHub according to the specified criteria."""
    time_from, time_to = \
        pd.Timestamp(time_intervals[0], tzinfo=timezone.utc), \
        pd.Timestamp(time_intervals[-1], tzinfo=timezone.utc)
    x_commits = await extract_commits(
        prop, time_from, time_to, repos, with_author, with_committer, db, cache)
    all_commits = await extract_commits(
        FilterCommitsProperty.no_pr_merges, time_from, time_to, repos, with_author, with_committer,
        db, cache, columns=[PushCommit.committed_date, PushCommit.additions, PushCommit.deletions])
    return calc_code_stats(x_commits, all_commits, time_intervals)


METRIC_ENTRIES = {
    "github": {
        **{k: calc_pull_request_metrics_line_github for k in pull_request_calculators},
        "code": filter_code_github,
    },
}  # type: Mapping[str, Mapping[str, Union[calc_metrics_line_func, filter_code_func]]]

PR_ENTRIES = {
    "github": filter_pull_requests_github,
}  # type: Mapping[str, filter_pull_requests_func]
