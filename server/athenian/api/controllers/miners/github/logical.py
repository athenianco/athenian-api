from typing import Collection, KeysView, Optional

import numpy as np
import pandas as pd

from athenian.api.controllers.logical_repos import coerce_logical_repos
from athenian.api.controllers.settings import LogicalDeploymentSettings, LogicalPRSettings, \
    LogicalRepositorySettings
from athenian.api.models.metadata.github import PullRequest
from athenian.api.models.persistentdata.models import DeployedComponent


def split_logical_prs(prs: pd.DataFrame,
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


def split_logical_deployed_components(notifications: pd.DataFrame,
                                      labels: pd.DataFrame,
                                      components: pd.DataFrame,
                                      logical_repos: Collection[str],
                                      logical_settings: LogicalRepositorySettings,
                                      ) -> pd.DataFrame:
    """Remove and clone deployed components according to the logical repository settings."""
    physical_repos = coerce_logical_repos(logical_repos)
    if physical_repos.keys() == logical_repos:
        return components
    chunks = []
    component_deployment_names = components.index.values.astype("U", copy=False)
    inspected_indexes = []
    for physical_repo, indexes in LogicalDeploymentSettings.group_by_repo(
            components[DeployedComponent.repository_full_name].values, physical_repos):
        try:
            repo_settings = logical_settings.deployments(physical_repo)
        except KeyError:
            inspected_indexes.append(indexes)
            chunks.append(components.take(indexes))
            continue

        for repo, deployment_names in repo_settings.match(notifications, labels).items():
            if repo not in logical_repos:
                continue
            deployment_names = np.array(
                list(deployment_names), dtype=component_deployment_names.dtype)
            indexes = indexes[np.in1d(component_deployment_names[indexes], deployment_names)]
            if len(indexes):
                inspected_indexes.append(indexes)
                sub_df = components.take(indexes)
                sub_df[DeployedComponent.repository_full_name] = repo
                chunks.append(sub_df)
    if inspected_indexes:
        inspected_indexes = np.unique(np.concatenate(inspected_indexes))
        missed_mask = np.ones(len(components), dtype=bool)
        missed_mask[inspected_indexes] = False
        if len(missed_indexes := np.flatnonzero(missed_mask)):
            chunks.append(components.take(missed_indexes))
    if len(chunks):
        components = pd.concat(chunks, copy=False)
    else:
        components = components.iloc[:0].copy()
    return components
