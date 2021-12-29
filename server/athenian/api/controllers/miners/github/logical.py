from typing import Collection, KeysView, Optional

import pandas as pd

from athenian.api.controllers.logical_repos import coerce_logical_repos
from athenian.api.controllers.settings import LogicalPRSettings, LogicalRepositorySettings
from athenian.api.models.metadata.github import PullRequest


def split_logical_repositories(prs: pd.DataFrame,
                               labels: Optional[pd.DataFrame],
                               logical_repos: Collection[str],
                               logical_settings: LogicalRepositorySettings,
                               reindex: bool = True,
                               reset_index: bool = True,
                               repo_column: str = PullRequest.repository_full_name.name,
                               id_column: str = PullRequest.node_id.name,
                               title_column: str = PullRequest.title.name,
                               ) -> pd.DataFrame:
    """Remove and clone PRs according to the logical repository settings."""
    assert isinstance(prs, pd.DataFrame)
    if labels is None:
        labels = pd.DataFrame()
    if reset_index:
        prs.reset_index(inplace=True)
    if isinstance(logical_repos, dict):
        logical_repos = logical_repos.keys()
    elif not isinstance(logical_repos, (set, KeysView)):
        logical_repos = set(logical_repos)
    physical_repos = coerce_logical_repos(logical_repos)
    if physical_repos.keys() != logical_repos:
        chunks = []
        for physical_repo, indexes in LogicalPRSettings.group_by_repo(
                prs[repo_column].values, physical_repos):
            try:
                repo_settings = logical_settings.prs(physical_repo)
            except KeyError:
                chunks.append(prs.take(indexes))
                continue
            for repo, logical_indexes in repo_settings.match(
                    prs, labels, indexes, id_column=id_column, title_column=title_column).items():
                if repo not in logical_repos:
                    continue
                if len(logical_indexes) < len(prs):
                    sub_df = prs.take(logical_indexes)
                else:
                    sub_df = prs.copy()
                sub_df[repo_column] = repo
                chunks.append(sub_df)
        if len(chunks):
            prs = pd.concat(chunks, copy=False)
        else:
            prs = prs.iloc[:0].copy()
    if reindex:
        prs.set_index([id_column, repo_column], inplace=True)
        prs.sort_index(inplace=True)
    return prs
