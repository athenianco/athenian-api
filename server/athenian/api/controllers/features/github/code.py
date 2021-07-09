from datetime import date, timedelta
from typing import List, Sequence

import pandas as pd

from athenian.api.controllers.features.code import CodeStats
from athenian.api.models.metadata.github import PushCommit
from athenian.api.tracing import sentry_span


@sentry_span
def calc_code_stats(queried_commits: pd.DataFrame,
                    total_commits: pd.DataFrame,
                    time_intervals: Sequence[date],
                    ) -> List[CodeStats]:
    """
    Calculate the commit statistics grouped by the given time intervals.

    :param queried_commits: DataFrame with the "interesting" commits.
    :param total_commits: DataFrame with "all" the commits in the same time periods.
    :param time_intervals: Series of time boundaries, both ends inclusive.
    :return: List with the calculated stats of length (len(time_intervals) - 1).
    """
    adkeys = [PushCommit.additions.name, PushCommit.deletions.name]
    time_intervals = [pd.Timestamp(d) for d in time_intervals]
    time_intervals[-1] += timedelta(days=1)  # pd.cut will not include the end otherwise
    all_stats = []
    for commits in (queried_commits, total_commits):
        if commits.empty:
            all_stats.append([0 for _ in time_intervals[1:]])
            all_stats.append([0 for _ in time_intervals[1:]])
            continue
        cut = pd.cut(commits[PushCommit.committed_date.name], time_intervals, right=False)
        grouped = commits[adkeys].groupby(cut)
        all_stats.append(grouped.count()[adkeys[0]])
        ad_lines = grouped.sum()
        all_stats.append(ad_lines[adkeys[0]] + ad_lines[adkeys[1]])
    result = [CodeStats(*x) for x in zip(*all_stats)]
    return result
