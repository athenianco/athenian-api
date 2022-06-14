from typing import Collection, Dict, List, Sequence, Type, TypeVar

import numpy as np
import pandas as pd

from athenian.api.internal.features.metric import Metric, MetricInt
from athenian.api.internal.features.metric_calculator import (
    AnyMetricCalculator,
    BinnedMetricCalculator,
    MetricCalculator,
    MetricCalculatorEnsemble,
    SumMetricCalculator,
)
from athenian.api.internal.miners.github.developer import (
    DeveloperTopic,
    developer_changed_lines_column,
    developer_identity_column,
)
from athenian.api.internal.miners.github.pull_request import ReviewResolution
from athenian.api.models.metadata.github import (
    PullRequest,
    PullRequestComment,
    PullRequestReview,
    PullRequestReviewComment,
    PushCommit,
    Release,
)

metric_calculators: Dict[str, Type[MetricCalculator]] = {}
T = TypeVar("T")


def register_metric(topic: DeveloperTopic):
    """Keep track of the developer metric calculators."""
    assert isinstance(topic, DeveloperTopic)

    def register_with_name(cls: Type[MetricCalculator]):
        metric_calculators[topic.value] = cls
        return cls

    return register_with_name


class DeveloperMetricCalculatorEnsemble(MetricCalculatorEnsemble):
    """MetricCalculatorEnsemble adapted for developers."""

    def __init__(self, *metrics: str, quantiles: Sequence[float], quantile_stride: int):
        """Initialize a new instance of ReleaseMetricCalculatorEnsemble class."""
        super().__init__(
            *metrics,
            quantiles=quantiles,
            quantile_stride=quantile_stride,
            class_mapping=metric_calculators
        )


class DeveloperBinnedMetricCalculator(BinnedMetricCalculator):
    """BinnedMetricCalculator adapted for developers."""

    ensemble_class = DeveloperMetricCalculatorEnsemble


class DeveloperTopicCounter(SumMetricCalculator[int]):
    """Count all `topic` events in each time interval."""

    may_have_negative_values = False
    metric = MetricInt
    timestamp_column: str

    def _analyze(
        self, facts: pd.DataFrame, min_times: np.ndarray, max_times: np.ndarray, **kwargs
    ) -> np.array:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        column = facts[self.timestamp_column].astype(min_times.dtype, copy=False).values
        column_in_range = (min_times[:, None] <= column) & (column < max_times[:, None])
        result[column_in_range] = 1
        return result


class DeveloperTopicSummator(SumMetricCalculator[int]):
    """Sum all `topic` events in each time interval."""

    may_have_negative_values = False
    metric = MetricInt
    topic_column: str
    timestamp_column: str

    def _analyze(
        self, facts: pd.DataFrame, min_times: np.ndarray, max_times: np.ndarray, **kwargs
    ) -> np.array:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        topic_column = facts[self.topic_column].values
        ts_column = facts[self.timestamp_column].values
        column_in_range = (min_times[:, None] <= ts_column) & (ts_column < max_times[:, None])
        for result_dim, column_in_range_dim in zip(result, column_in_range):
            result_dim[column_in_range_dim] = topic_column[column_in_range_dim]
        return result


@register_metric(DeveloperTopic.commits_pushed)
class CommitsPushedCounter(DeveloperTopicCounter):
    """Calculate "dev-commits-pushed" metric."""

    timestamp_column = PushCommit.committed_date.name


@register_metric(DeveloperTopic.lines_changed)
class LinesChangedCounter(DeveloperTopicSummator):
    """Calculate "dev-lines-changed" metric."""

    topic_column = developer_changed_lines_column
    timestamp_column = PushCommit.committed_date.name


@register_metric(DeveloperTopic.active)
class ActiveCounter(MetricCalculator[int]):
    """Calculate "dev-active" metric."""

    ACTIVITY_DAYS_THRESHOLD_DENSITY = 0.2

    may_have_negative_values = False
    metric = MetricInt

    def _value(self, samples: np.ndarray) -> Metric[int]:
        if len(samples) > 0:
            days = samples[0] % 1000000
            active = len(np.unique(samples // 1000000))
        else:
            days = 1
            active = 0
        assert days > 0
        value = int(active / days > self.ACTIVITY_DAYS_THRESHOLD_DENSITY)
        return self.metric.from_fields(True, value, None, None)

    def _analyze(
        self, facts: pd.DataFrame, min_times: np.ndarray, max_times: np.ndarray, **kwargs
    ) -> np.array:
        column = facts[PushCommit.committed_date.name].dt.floor(freq="D").values
        column_in_range = (min_times[:, None] <= column) & (column < max_times[:, None])
        timestamps = np.repeat(column[None, :], len(min_times), axis=0)
        result = timestamps.view(int)
        lengths = (max_times - min_times).astype("timedelta64[D]").view(int)
        result += lengths[:, None]
        result[~column_in_range] = self.nan
        return result


@register_metric(DeveloperTopic.active0)
class Active0Counter(AnyMetricCalculator[int]):
    """Calculate "dev-active0" metric."""

    deps = (ActiveCounter,)
    may_have_negative_values = False
    metric = MetricInt

    def _analyze(
        self, facts: pd.DataFrame, min_times: np.ndarray, max_times: np.ndarray, **kwargs
    ) -> np.array:
        return self._calcs[0].peek


@register_metric(DeveloperTopic.prs_created)
class PRsCreatedCounter(DeveloperTopicCounter):
    """Calculate "dev-prs-created" metric."""

    timestamp_column = PullRequest.created_at.name


@register_metric(DeveloperTopic.prs_merged)
class PRsMergedCounter(DeveloperTopicCounter):
    """Calculate "dev-prs-merged" metric."""

    timestamp_column = PullRequest.merged_at.name


@register_metric(DeveloperTopic.releases)
class ReleasesCounter(DeveloperTopicCounter):
    """Calculate "dev-releases" metric."""

    timestamp_column = Release.published_at.name


@register_metric(DeveloperTopic.regular_pr_comments)
class RegularPRCommentsCounter(DeveloperTopicCounter):
    """Calculate "dev-regular-pr-comments" metric."""

    timestamp_column = PullRequestComment.created_at.name


@register_metric(DeveloperTopic.review_pr_comments)
class ReviewPRCommentsCounter(DeveloperTopicCounter):
    """Calculate "dev-review-pr-comments" metric."""

    timestamp_column = PullRequestReviewComment.created_at.name


@register_metric(DeveloperTopic.pr_comments)
class PRCommentsCounter(DeveloperTopicCounter):
    """Calculate "dev-pr-comments" metric."""

    timestamp_column = "created_at"


@register_metric(DeveloperTopic.prs_reviewed)
class PRReviewedCounter(SumMetricCalculator[int]):
    """Calculate "dev-prs-reviewed" metric."""

    may_have_negative_values = False
    metric = MetricInt

    def _analyze(
        self, facts: pd.DataFrame, min_times: np.ndarray, max_times: np.ndarray, **kwargs
    ) -> np.array:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        column = facts[PullRequestReview.submitted_at.name].values
        column_in_range = (min_times[:, None] <= column) & (column < max_times[:, None])
        duplicated = facts.duplicated(
            [
                PullRequestReview.pull_request_node_id.name,
                developer_identity_column,
            ]
        ).values
        column_in_range[np.broadcast_to(duplicated[None, :], result.shape)] = False
        result[column_in_range] = 1
        return result


@register_metric(DeveloperTopic.reviews)
class ReviewsCounter(DeveloperTopicCounter):
    """Calculate "dev-reviews" metric."""

    timestamp_column = PullRequestReview.submitted_at.name


class ReviewStatesCounter(SumMetricCalculator[int]):
    """Count reviews with the specified outcome in `state`."""

    may_have_negative_values = False
    metric = MetricInt
    state = None

    def _analyze(
        self, facts: pd.DataFrame, min_times: np.ndarray, max_times: np.ndarray, **kwargs
    ) -> np.array:
        result = np.full((len(min_times), len(facts)), self.nan, self.dtype)
        column = facts[PullRequestReview.submitted_at.name].values
        column_in_range = (min_times[:, None] <= column) & (column < max_times[:, None])
        wrong_state = facts[PullRequestReview.state.name].values != self.state.value
        column_in_range[np.broadcast_to(wrong_state[None, :], result.shape)] = False
        result[column_in_range] = 1
        return result


@register_metric(DeveloperTopic.review_approvals)
class ApprovalsCounter(ReviewStatesCounter):
    """Calculate "dev-review-approved" metric."""

    state = ReviewResolution.APPROVED


@register_metric(DeveloperTopic.review_rejections)
class RejectionsCounter(ReviewStatesCounter):
    """Calculate "dev-review-rejected" metric."""

    state = ReviewResolution.CHANGES_REQUESTED


@register_metric(DeveloperTopic.review_neutrals)
class NeutralReviewsCounter(ReviewStatesCounter):
    """Calculate "dev-review-neutrals" metric."""

    state = ReviewResolution.COMMENTED


@register_metric(DeveloperTopic.worked)
class WorkedCounter(AnyMetricCalculator[int]):
    """Calculate "dev-worked" metric."""

    deps = (
        PRsCreatedCounter,
        PRsMergedCounter,
        ReleasesCounter,
        CommitsPushedCounter,
        ReviewsCounter,
        RegularPRCommentsCounter,
    )
    may_have_negative_values = False
    metric = MetricInt

    def _analyze(
        self, facts: pd.DataFrame, min_times: np.ndarray, max_times: np.ndarray, **kwargs
    ) -> np.array:
        result = np.full((len(min_times), len(facts)), 0, self.dtype)
        for calc in self._calcs:
            result |= calc.peek > 0
        result[result == 0] = self.nan
        return result


def group_actions_by_developers(
    devs: Sequence[Collection[str]],
    df: pd.DataFrame,
) -> List[np.ndarray]:
    """Group developer actions by developer groups."""
    indexes = []
    identities = df[developer_identity_column].values.astype("S")
    for group in devs:
        if len(group) == 1:
            dev = next(iter(group))
            indexes.append(np.nonzero(identities == dev.encode())[0])
            continue
        if isinstance(group, set):
            group = list(group)
        indexes.append(np.nonzero(np.in1d(identities, np.array(group, dtype="S")))[0])
    return indexes
