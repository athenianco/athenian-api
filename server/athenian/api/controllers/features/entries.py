from datetime import date, timezone
import pickle
from typing import Collection, Dict, List, Optional, Sequence, Tuple

import aiomcache
from databases import Database
import pandas as pd

from athenian.api.cache import cached
from athenian.api.controllers.features.cached_released import load_cached_released_times, \
    store_cached_released_times
from athenian.api.controllers.features.code import CodeStats
from athenian.api.controllers.features.github.code import calc_code_stats
from athenian.api.controllers.features.github.pull_request import \
    BinnedPullRequestMetricCalculator, calculators as pull_request_calculators
import athenian.api.controllers.features.github.pull_request_metrics  # noqa
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.miners.github.commit import extract_commits, FilterCommitsProperty
from athenian.api.controllers.miners.github.developer import calc_developer_metrics
from athenian.api.controllers.miners.github.pull_request import PullRequestTimesMiner
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.models.metadata.github import PushCommit


@cached(
    exptime=PullRequestTimesMiner.CACHE_TTL,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda metrics, time_intervals, repos, developers, **_: (
        ",".join(sorted(metrics)),
        ";".join(",".join(str(dt.toordinal()) for dt in ts) for ts in time_intervals),
        ",".join(sorted(repos)),
        ",".join(sorted(developers)),
    ),
)
async def calc_pull_request_metrics_line_github(
        metrics: Collection[str], time_intervals: Sequence[Sequence[date]], repos: Collection[str],
        release_settings: Dict[str, ReleaseMatchSetting], developers: Collection[str],
        db: Database, cache: Optional[aiomcache.Client],
) -> List[List[Tuple[Metric]]]:
    """Calculate pull request metrics on GitHub data."""
    date_from, date_to = time_intervals[0][0], time_intervals[0][-1]
    assert date_to > date_from
    released_times = await load_cached_released_times(date_from, date_to, repos, cache)
    miner = await PullRequestTimesMiner.mine(
        date_from, date_to, repos, release_settings, developers, db, cache,
        pr_blacklist=released_times,
    )
    mined_prs = list(miner)
    calcs = [pull_request_calculators[m]() for m in metrics]
    times = [BinnedPullRequestMetricCalculator(calcs, ts)(mined_prs) for ts in time_intervals]
    await store_cached_released_times(mined_prs, times, cache)
    times.extend(released_times.values())
    return times


async def calc_code_metrics(
        prop: FilterCommitsProperty, time_intervals: Sequence[date], repos: Collection[str],
        with_author: Optional[Collection[str]], with_committer: Optional[Collection[str]],
        db: Database, cache: Optional[aiomcache.Client],
) -> List[CodeStats]:
    """Filter code pushed on GitHub according to the specified criteria."""
    time_from, time_to = (pd.Timestamp(time_intervals[i], tzinfo=timezone.utc) for i in (0, -1))
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
