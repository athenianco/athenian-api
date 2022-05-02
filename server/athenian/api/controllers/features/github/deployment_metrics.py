from datetime import datetime
from typing import Collection, Dict, List, Sequence, Type, TypeVar

import numpy as np
import pandas as pd

from athenian.api.controllers.features.metric import MetricFloat, MetricInt, MetricTimeDelta
from athenian.api.controllers.features.metric_calculator import AverageMetricCalculator, \
    BinnedMetricCalculator, make_register_metric, MetricCalculator, MetricCalculatorEnsemble, \
    RatioCalculator, SumMetricCalculator
from athenian.api.controllers.miners.types import DeploymentFacts, PullRequestJIRAIssueItem, \
    ReleaseParticipants, \
    ReleaseParticipationKind
from athenian.api.models.persistentdata.models import DeployedComponent, DeploymentNotification
from athenian.api.models.web import DeploymentMetricID

metric_calculators: Dict[str, Type[MetricCalculator]] = {}
register_metric = make_register_metric(metric_calculators, None)
T = TypeVar("T")


class DeploymentMetricCalculatorEnsemble(MetricCalculatorEnsemble):
    """MetricCalculatorEnsemble adapted for releases."""

    def __init__(self,
                 *metrics: str,
                 quantiles: Sequence[float],
                 quantile_stride: int,
                 jira: Dict[str, PullRequestJIRAIssueItem]):
        """Initialize a new instance of DeploymentMetricCalculatorEnsemble class."""
        super().__init__(*metrics,
                         quantiles=quantiles,
                         quantile_stride=quantile_stride,
                         class_mapping=metric_calculators,
                         jira=jira)


class DeploymentBinnedMetricCalculator(BinnedMetricCalculator):
    """BinnedMetricCalculator adapted for releases."""

    ensemble_class = DeploymentMetricCalculatorEnsemble


def group_deployments_by_repositories(repositories: Sequence[Collection[str]],
                                      df: pd.DataFrame,
                                      ) -> List[np.ndarray]:
    """Group deployments by repository node IDs."""
    if len(repositories) == 0:
        return [np.arange(len(df))]
    if df.empty:
        return [np.array([], dtype=int)] * len(repositories)
    df_repos = [c[DeployedComponent.repository_full_name].values
                for c in df["components"].values]
    df_repos_flat = np.concatenate(df_repos).astype("U", copy=False)
    # DEV-4112 exclude empty deployments
    df_commits = np.concatenate(df[DeploymentFacts.f.commits_overall].values,
                                dtype=int, casting="unsafe")
    df_repos_flat[df_commits == 0] = ""

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
        return [np.array([], dtype=int)] * len(participants)
    preprocessed = {}
    for pkind, col in zip(ReleaseParticipationKind, [DeploymentFacts.f.pr_authors,
                                                     DeploymentFacts.f.commit_authors,
                                                     DeploymentFacts.f.release_authors]):
        values = df[col].values
        offsets = np.zeros(len(values) + 1, dtype=int)
        lengths = np.array([len(v) for v in values])
        np.cumsum(lengths, out=offsets[1:])
        values = np.concatenate([np.concatenate(values), [-1]])
        preprocessed[pkind] = values, offsets, lengths == 0
    result = []
    for filters in participants:
        mask = np.zeros(len(df), dtype=bool)
        for pkind in ReleaseParticipationKind:
            if pkind not in filters:
                continue
            people = np.array(filters[pkind])
            values, offsets, empty = preprocessed[pkind]
            passing = np.bitwise_or.reduceat(np.in1d(values, people), offsets)[:-1]
            passing[empty] = False
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
        return [np.array([], dtype=int)] * len(environments)
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


@register_metric(DeploymentMetricID.DEP_COUNT)
class DeploymentsCounter(SumMetricCalculator[int]):
    """Calculate the number of deployments in the time period."""

    metric = MetricInt

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        deployed = facts[DeploymentNotification.finished_at.name].values
        result[(min_times[:, None] <= deployed) & (deployed < max_times[:, None])] = 1
        return result


@register_metric(DeploymentMetricID.DEP_SUCCESS_COUNT)
class SuccessfulDeploymentsCounter(SumMetricCalculator[int]):
    """Calculate the number of successful deployments in the time period."""

    metric = MetricInt

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        deployed = facts[DeploymentNotification.finished_at.name].values.copy()
        unsuccessful = (
            facts[DeploymentNotification.conclusion.name].values
            != DeploymentNotification.CONCLUSION_SUCCESS
        )
        deployed[unsuccessful] = np.datetime64("NaT")
        result[(min_times[:, None] <= deployed) & (deployed < max_times[:, None])] = 1
        return result


@register_metric(DeploymentMetricID.DEP_FAILURE_COUNT)
class FailedDeploymentsCounter(SumMetricCalculator[int]):
    """Calculate the number of failed deployments in the time period."""

    metric = MetricInt

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        deployed = facts[DeploymentNotification.finished_at.name].values.copy()
        unfailed = (
            facts[DeploymentNotification.conclusion.name].values
            != DeploymentNotification.CONCLUSION_FAILURE
        )
        deployed[unfailed] = np.datetime64("NaT")
        result[(min_times[:, None] <= deployed) & (deployed < max_times[:, None])] = 1
        return result


@register_metric(DeploymentMetricID.DEP_DURATION_ALL)
class DurationCalculator(AverageMetricCalculator[datetime]):
    """Calculate the average deployment procedure time - the difference between `date_finished` \
    and `date_started` of `DeploymentNotification`."""

    metric = MetricTimeDelta
    may_have_negative_values = False

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        started = facts[DeploymentNotification.started_at.name].values
        finished = facts[DeploymentNotification.finished_at.name].values.copy()
        cancelled = (
            facts[DeploymentNotification.conclusion.name].values
            == DeploymentNotification.CONCLUSION_CANCELLED
        )
        durations = finished - started
        finished[cancelled] = np.datetime64("NaT")
        mask = (min_times[:, None] <= finished) & (finished < max_times[:, None])
        result[mask] = np.broadcast_to(durations[None, :], result.shape)[mask]
        return result


@register_metric(DeploymentMetricID.DEP_DURATION_SUCCESSFUL)
class SuccessfulDurationCalculator(AverageMetricCalculator[datetime]):
    """Calculate the average successful deployment procedure time - the difference between \
    `date_finished` and `date_started` of `DeploymentNotification`."""

    metric = MetricTimeDelta
    may_have_negative_values = False

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        started = facts[DeploymentNotification.started_at.name].values
        finished = facts[DeploymentNotification.finished_at.name].values.copy()
        unsuccessful = (
            facts[DeploymentNotification.conclusion.name].values
            != DeploymentNotification.CONCLUSION_SUCCESS
        )
        durations = finished - started
        finished[unsuccessful] = np.datetime64("NaT")
        mask = (min_times[:, None] <= finished) & (finished < max_times[:, None])
        result[mask] = np.broadcast_to(durations[None, :], result.shape)[mask]
        return result


@register_metric(DeploymentMetricID.DEP_DURATION_FAILED)
class FailedDurationCalculator(AverageMetricCalculator[datetime]):
    """Calculate the average failed deployment procedure time - the difference between \
    `date_finished` and `date_started` of `DeploymentNotification`."""

    metric = MetricTimeDelta
    may_have_negative_values = False

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        started = facts[DeploymentNotification.started_at.name].values
        finished = facts[DeploymentNotification.finished_at.name].values.copy()
        unfailed = (
            facts[DeploymentNotification.conclusion.name].values
            != DeploymentNotification.CONCLUSION_FAILURE
        )
        durations = finished - started
        finished[unfailed] = np.datetime64("NaT")
        mask = (min_times[:, None] <= finished) & (finished < max_times[:, None])
        result[mask] = np.broadcast_to(durations[None, :], result.shape)[mask]
        return result


@register_metric(DeploymentMetricID.DEP_SUCCESS_RATIO)
class SuccessRatioCalculator(RatioCalculator):
    """Calculate the ratio between successful and all deployments."""

    deps = (SuccessfulDeploymentsCounter, DeploymentsCounter)


class ItemsMixin:
    """Calculate the average `agg` of deployed items in `facts[dimension]`."""

    may_have_negative_values = False
    dimension = ""
    agg = None

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.ndarray:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        agg = self.agg
        items = np.fromiter((agg(v) for v in facts[self.dimension].values), int, len(facts))
        deployed = facts[DeploymentNotification.finished_at.name].values
        mask = (min_times[:, None] <= deployed) & (deployed < max_times[:, None])
        result[mask] = np.broadcast_to(items[None, :], result.shape)[mask]
        return result


class SizeCalculator(ItemsMixin, AverageMetricCalculator[float]):
    """Calculate the average `agg` of deployed items in `facts[dimension]`."""

    metric = MetricFloat


@register_metric(DeploymentMetricID.DEP_SIZE_PRS)
class DeployedPRSizeCalculator(SizeCalculator):
    """Calculate the average number of deployed pull requests."""

    dimension = "prs"
    agg = len


@register_metric(DeploymentMetricID.DEP_SIZE_RELEASES)
class DeployedReleaseSizeCalculator(SizeCalculator):
    """Calculate the average number of deployed releases."""

    dimension = "releases"
    agg = len


@register_metric(DeploymentMetricID.DEP_SIZE_LINES)
class DeployedLineSizeCalculator(SizeCalculator):
    """Calculate the average number of deployed line changes."""

    dimension = "lines_overall"
    agg = sum


@register_metric(DeploymentMetricID.DEP_SIZE_COMMITS)
class DeployedCommitsSizeCalculator(SizeCalculator):
    """Calculate the average number of deployed commits."""

    dimension = "commits_overall"
    agg = sum


class SumCalculator(ItemsMixin, SumMetricCalculator[int]):
    """Calculate the sum of `agg` of deployed items in `facts[dimension]`."""

    metric = MetricInt


@register_metric(DeploymentMetricID.DEP_PRS_COUNT)
class DeployedPRsCounter(SumCalculator):
    """Calculate the number of deployed pull requests."""

    dimension = "prs"
    agg = len


@register_metric(DeploymentMetricID.DEP_RELEASES_COUNT)
class DeployedReleasesCounter(SumCalculator):
    """Calculate the number of deployed releases."""

    dimension = "releases"
    agg = len


@register_metric(DeploymentMetricID.DEP_LINES_COUNT)
class DeployedLinesCounter(SumCalculator):
    """Calculate the number of deployed line changes."""

    dimension = "lines_overall"
    agg = sum


@register_metric(DeploymentMetricID.DEP_COMMITS_COUNT)
class DeployedCommitsCounter(SumCalculator):
    """Calculate the number of deployed commits."""

    dimension = "commits_overall"
    agg = sum


@register_metric(DeploymentMetricID.DEP_JIRA_ISSUES_COUNT)
class DeployedIssuesCounter(SumCalculator):
    """Calculate the number of deployed JIRA issues."""

    dimension = "jira"

    def agg(self, values: np.ndarray) -> int:
        """Calculate the number of unique issues."""
        return len(np.unique(np.concatenate(values)))


@register_metric(DeploymentMetricID.DEP_JIRA_BUG_FIXES_COUNT)
class DeployedBugFixesCounter(SumCalculator):
    """Calculate the number of deployed bug fixes according to mapped JIRA issues."""

    dimension = "jira"

    def agg(self, values: np.ndarray) -> int:
        """Calculate the number of unique issues with lower(type) == "bug"."""
        deployed = np.unique(np.concatenate(values))
        jira = self.jira
        return sum(jira[i.decode()].type.lower() == "bug" for i in deployed)
