from dataclasses import dataclass
from datetime import date
from typing import List, Sequence

import pandas as pd

from athenian.api.models.metadata.github import PushCommit


@dataclass(frozen=True)
class CodeStats:
    """
    Statistics about a certain group of commits: counted "special" commits and LoC which that \
    group contains and the overall counted commits and LoC in that group.

    What's "special" is abstracted here. For example, pushed without making a PR.
    """

    queried_number_of_commits: int
    queried_number_of_lines: int
    total_number_of_commits: int
    total_number_of_lines: int


def calc_code_stats(queried_commits: pd.DataFrame, total_commits: pd.DataFrame,
                    time_intervals: Sequence[date]) -> List[CodeStats]:
    """
    Calculate the commit statistics grouped by the given time intervals.

    :param queried_commits: DataFrame with the "interesting" commits.
    :param total_commits: DataFrame with "all" the commits in the same time periods.
    :param time_intervals: Time boundaries, sequential.
    :return: List with the calculated stats of length (len(time_intervals) - 1).
    """
    adkeys = [PushCommit.additions.key, PushCommit.deletions.key]
    time_intervals = [pd.Timestamp(d) for d in time_intervals]
    all_stats = []
    for commits in (queried_commits, total_commits):
        cut = pd.cut(commits[PushCommit.committed_date.key], time_intervals)
        grouped = commits[adkeys].groupby(cut)
        all_stats.append(grouped.count()[adkeys[0]])
        ad_lines = grouped.sum()
        all_stats.append(ad_lines[adkeys[0]] + ad_lines[adkeys[1]])
    result = [CodeStats(*x) for x in zip(*all_stats)]
    return result
