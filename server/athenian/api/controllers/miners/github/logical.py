from typing import Set

import numpy as np
import pandas as pd

from athenian.api.controllers.settings import LogicalPRSettings, LogicalRepositorySettings
from athenian.api.models.metadata.github import PullRequest


def split_logical_repositories(prs: pd.DataFrame,
                               labels: pd.DataFrame,
                               logical_repos: Set[str],
                               logical_settings: LogicalRepositorySettings,
                               ) -> pd.DataFrame:
    """Remove and clone PRs according to the logical repository settings."""
    chunks = []
    removed = []
    for physical_repo, indexes in LogicalPRSettings.group_by_repo(
            prs[PullRequest.repository_full_name.name].values, logical_repos):
        try:
            repo_settings = logical_settings.prs(physical_repo)
        except KeyError:
            continue
        for repo, logical_indexes in repo_settings.match(prs, labels, indexes).items():
            sub_df = prs.take(logical_indexes)
            sub_df["repository_full_name"] = repo
            sub_df.reset_index(inplace=True)
            chunks.append(sub_df)
            removed.append(logical_indexes)
    leave_mask = np.ones(len(prs), dtype=bool)
    if removed:
        leave_mask[np.unique(np.concatenate(removed))] = False
        left = prs.take(np.flatnonzero(leave_mask))
    else:
        left = prs
    left.reset_index(inplace=True)
    if chunks:
        prs = pd.concat([left, *chunks], names=(
            PullRequest.node_id.name, PullRequest.repository_full_name.name,
        ))
    else:
        prs = left
        prs.set_index([PullRequest.node_id.name, PullRequest.repository_full_name.name],
                      inplace=True)
    prs.sort_index(inplace=True)
    return prs
