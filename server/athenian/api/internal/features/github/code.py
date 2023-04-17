from datetime import date
from typing import Sequence

import medvedi as md
import numpy as np

from athenian.api.internal.features.code import CodeStats
from athenian.api.models.metadata.github import PushCommit
from athenian.api.tracing import sentry_span


@sentry_span
def calc_code_stats(
    queried_commits: md.DataFrame,
    total_commits: md.DataFrame,
    time_intervals: Sequence[date],
) -> list[CodeStats]:
    """
    Calculate the commit statistics grouped by the given time intervals.

    :param queried_commits: DataFrame with the "interesting" commits.
    :param total_commits: DataFrame with "all" the commits in the same time periods.
    :param time_intervals: Series of time boundaries, both ends inclusive.
    :return: List with the calculated stats of length (len(time_intervals) - 1).
    """
    time_intervals = np.array(time_intervals, dtype="datetime64[us]")
    time_intervals[-1] += np.timedelta64(1, "D")  # pd.cut will not include the end otherwise
    all_stats = []
    for commits in (queried_commits, total_commits):
        if commits.empty:
            for _ in range(2):
                all_stats.append(np.zeros(len(time_intervals) - 1, dtype=int))
            continue
        assert commits[PushCommit.committed_date.name].dtype == time_intervals.dtype
        cut = np.searchsorted(time_intervals, commits[PushCommit.committed_date.name]) - 1
        if not (range_mask := (cut >= 0) & (cut < len(time_intervals))).all():
            cut = cut[range_mask]
            commits.take(range_mask, inplace=True)
        grouper = commits.groupby(cut)
        group_indexes = cut[grouper.group_indexes()]
        commit_stats = np.zeros(len(time_intervals) - 1, dtype=int)
        line_stats = np.zeros(len(time_intervals) - 1, dtype=int)
        commit_stats[group_indexes] = grouper.counts
        line_stats[group_indexes] = np.add.reduceat(
            (commits[PushCommit.additions.name] + commits[PushCommit.deletions.name])[
                grouper.order
            ],
            grouper.reduceat_indexes(),
        )
        all_stats.append(commit_stats)
        all_stats.append(line_stats)
    result = [CodeStats(*x) for x in zip(*all_stats)]
    return result
