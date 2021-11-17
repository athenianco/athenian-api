from typing import Collection, KeysView, Optional

import pandas as pd

from athenian.api.controllers.logical_repos import coerce_logical_repos
from athenian.api.controllers.settings import LogicalPRSettings, LogicalRepositorySettings
from athenian.api.models.metadata.github import PullRequest


def split_logical_repositories(prs: pd.DataFrame,
                               labels: Optional[pd.DataFrame],
                               logical_repos: Collection[str],
                               logical_settings: LogicalRepositorySettings,
                               ) -> pd.DataFrame:
    """Remove and clone PRs according to the logical repository settings."""
    assert isinstance(prs, pd.DataFrame)
    prs.reset_index(inplace=True)
    if isinstance(logical_repos, dict):
        logical_repos = logical_repos.keys()
    elif not isinstance(logical_repos, (set, KeysView)):
        logical_repos = set(logical_repos)
    physical_repos = coerce_logical_repos(logical_repos)
    if physical_repos.keys() != logical_repos:
        chunks = []
        for physical_repo, indexes in LogicalPRSettings.group_by_repo(
                prs[PullRequest.repository_full_name.name].values, physical_repos):
            try:
                repo_settings = logical_settings.prs(physical_repo)
            except KeyError:
                chunks.append(prs.take(indexes))
                continue
            for repo, logical_indexes in repo_settings.match(prs, labels, indexes).items():
                if repo not in logical_repos:
                    continue
                if len(logical_indexes) < len(prs):
                    sub_df = prs.take(logical_indexes)
                else:
                    sub_df = prs.copy()
                sub_df["repository_full_name"] = repo
                chunks.append(sub_df)
        if len(chunks):
            prs = pd.concat(chunks, copy=False)
        else:
            prs = prs.iloc[:0].copy()
    prs.set_index([PullRequest.node_id.name, PullRequest.repository_full_name.name],
                  inplace=True)
    prs.sort_index(inplace=True)
    return prs
