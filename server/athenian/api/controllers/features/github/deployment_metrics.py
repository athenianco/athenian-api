from typing import Collection, Dict, List, Sequence, Type, TypeVar

import numpy as np
import pandas as pd

from athenian.api.controllers.features.metric_calculator import BinnedMetricCalculator, \
    make_register_metric, \
    MetricCalculator, MetricCalculatorEnsemble
from athenian.api.controllers.miners.types import DeploymentFacts, ReleaseParticipants, \
    ReleaseParticipationKind
from athenian.api.models.persistentdata.models import DeployedComponent, DeploymentNotification

metric_calculators: Dict[str, Type[MetricCalculator]] = {}
register_metric = make_register_metric(metric_calculators, None)
T = TypeVar("T")


class DeploymentMetricCalculatorEnsemble(MetricCalculatorEnsemble):
    """MetricCalculatorEnsemble adapted for releases."""

    def __init__(self, *metrics: str, quantiles: Sequence[float], quantile_stride: int):
        """Initialize a new instance of DeploymentMetricCalculatorEnsemble class."""
        super().__init__(*metrics,
                         quantiles=quantiles,
                         quantile_stride=quantile_stride,
                         class_mapping=metric_calculators)


class DeploymentBinnedMetricCalculator(BinnedMetricCalculator):
    """BinnedMetricCalculator adapted for releases."""

    ensemble_class = DeploymentMetricCalculatorEnsemble


def group_deployments_by_repositories(repositories: Sequence[Collection[int]],
                                      df: pd.DataFrame,
                                      ) -> List[np.ndarray]:
    """Group deployments by repository node IDs."""
    if len(repositories) == 0:
        return [np.arange(len(df))]
    if df.empty:
        return [np.ndarray([], dtype=int)]
    df_repos = [c[DeployedComponent.repository_node_id.name].values
                for c in df["components"].values]
    df_repos_flat = np.concatenate(df_repos)
    offsets = np.zeros(len(df), dtype=int)
    np.cumsum([len(c) for c in df_repos[:-1]], out=offsets[1:])
    repositories = [
        np.array(repo_group if not isinstance(repo_group, set) else list(repo_group))
        for repo_group in repositories
    ]
    unique_repos, imap = np.unique(np.concatenate(repositories), return_inverse=True)
    result = []
    if len(unique_repos) <= len(repositories):
        matches = np.array([df_repos_flat == i for i in unique_repos])
        pos = 0
        for repo_group in repositories:
            step = len(repo_group)
            cols = imap[pos:pos + step]
            flags = np.sum(matches[cols], axis=0).astype(bool)
            group = np.flatnonzero(np.bitwise_or.reduceat(flags, offsets))
            pos += step
            result.append(group)
    else:
        for repo_group in repositories:
            flags = np.in1d(df_repos_flat, repo_group)
            group = np.flatnonzero(np.bitwise_or.reduceat(flags, offsets))
            result.append(group)
    return result


def group_deployments_by_participants(participants: List[ReleaseParticipants],
                                      df: pd.DataFrame,
                                      ) -> List[np.ndarray]:
    """Group deployments by participants."""
    if len(participants) == 0:
        return [np.arange(len(df))]
    if df.empty:
        return [np.ndarray([], dtype=int)]
    preprocessed = {}
    for pkind, col in zip(ReleaseParticipationKind, [DeploymentFacts.f.pr_authors,
                                                     DeploymentFacts.f.commit_authors,
                                                     DeploymentFacts.f.release_authors]):
        values = df[col].values
        offsets = np.zeros(len(values), dtype=int)
        np.cumsum(np.array([len(v) for v in values[:-1]]), out=offsets[1:])
        values = np.concatenate(values)
        preprocessed[pkind] = values, offsets
    result = []
    for filters in participants:
        mask = np.zeros(len(df), dtype=bool)
        for pkind in ReleaseParticipationKind:
            if pkind not in filters:
                continue
            people = np.array(participants[pkind])
            values, offsets = preprocessed[pkind]
            passing = np.bitwise_or.reduceat(np.in1d(values, people), offsets)
            mask[passing] = True
        result.append(np.flatnonzero(mask))
    return result


def group_deployments_by_environments(environments: List[List[str]],
                                      df: pd.DataFrame,
                                      ) -> List[np.ndarray]:
    """Group deployments by environments."""
    if len(environments) == 0:
        return [np.arange(len(df))]
    if df.empty:
        return [np.ndarray([], dtype=int)]
    df_envs = df[DeploymentNotification.environment.name].values.astype("U", copy=False)
    unique_envs, imap = np.unique(np.concatenate(environments), return_inverse=True)
    result = []
    if len(unique_envs) <= len(environments):
        matches = np.array([df_envs == env for env in unique_envs])
        pos = 0
        for env_group in environments:
            step = len(env_group)
            cols = imap[pos:pos + step]
            group = np.flatnonzero(np.sum(matches[cols], axis=0).astype(bool))
            pos += step
            result.append(group)
    else:
        result = [
            np.flatnonzero(np.in1d(df_envs, env_group))
            for env_group in environments
        ]
    return result
