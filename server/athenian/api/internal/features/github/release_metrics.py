from datetime import timedelta
from typing import Generic, Optional, Sequence, Type, TypeVar

import numpy as np
import pandas as pd

from athenian.api.internal.features.metric import MetricInt, MetricTimeDelta
from athenian.api.internal.features.metric_calculator import (
    AverageMetricCalculator,
    BinnedMetricCalculator,
    MetricCalculator,
    MetricCalculatorEnsemble,
    SumMetricCalculator,
    calculate_logical_duplication_mask,
    make_register_metric,
)
from athenian.api.internal.miners.github.released_pr import matched_by_column
from athenian.api.internal.miners.participation import (
    ReleaseParticipants,
    ReleaseParticipationKind,
)
from athenian.api.internal.miners.types import ReleaseFacts
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseMatch, ReleaseSettings
from athenian.api.models.metadata.github import PullRequest
from athenian.api.models.web import ReleaseMetricID
from athenian.api.to_object_arrays import nested_lengths

metric_calculators: dict[str, Type[MetricCalculator]] = {}
register_metric = make_register_metric(metric_calculators, None)
T = TypeVar("T")


def group_releases_by_participants(
    participants: list[ReleaseParticipants],
    df: pd.DataFrame,
) -> list[np.ndarray]:
    """Triage releases by their contributors."""
    if not participants or df.empty:
        return [np.arange(len(df))]
    indexes = []
    for group in participants:
        if ReleaseParticipationKind.RELEASER in group:
            missing_indexes = np.flatnonzero(
                np.in1d(
                    np.array(df["publisher"].values),
                    group[ReleaseParticipationKind.RELEASER],
                    invert=True,
                ),
            )
        else:
            missing_indexes = np.arange(len(df))
        for rpk, col in [
            (ReleaseParticipationKind.COMMIT_AUTHOR, "commit_authors"),
            (ReleaseParticipationKind.PR_AUTHOR, "prs_" + PullRequest.user_node_id.name),
        ]:
            if len(missing_indexes) == 0:
                break
            if rpk in group:
                values = df[col].values[missing_indexes]
                lengths = nested_lengths(values)
                offsets = np.zeros(len(values) + 1, dtype=int)
                np.cumsum(lengths, out=offsets[1:])
                values = np.concatenate([np.concatenate(values), [-1]])
                passed = np.bitwise_or.reduceat(np.in1d(values, group[rpk]), offsets)[:-1]
                passed[lengths == 0] = False
                missing_indexes = missing_indexes[~passed]
        mask = np.ones(len(df), bool)
        mask[missing_indexes] = False
        indexes.append(np.flatnonzero(mask))
    return indexes


def calculate_logical_release_duplication_mask(
    items: pd.DataFrame,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
) -> Optional[np.ndarray]:
    """Assign indexes to releases with the same settings for each logical repository."""
    if items.empty or not logical_settings.has_logical_prs():
        return None
    repos_column = items[ReleaseFacts.f.repository_full_name].values.astype("S", copy=False)
    return calculate_logical_duplication_mask(repos_column, release_settings, logical_settings)


class ReleaseMetricCalculatorEnsemble(MetricCalculatorEnsemble):
    """MetricCalculatorEnsemble adapted for releases."""

    def __init__(self, *metrics: str, quantiles: Sequence[float], quantile_stride: int, **kwargs):
        """Initialize a new instance of ReleaseMetricCalculatorEnsemble class."""
        super().__init__(
            *metrics,
            quantiles=quantiles,
            quantile_stride=quantile_stride,
            class_mapping=metric_calculators,
            **kwargs,
        )


class ReleaseBinnedMetricCalculator(BinnedMetricCalculator):
    """BinnedMetricCalculator adapted for releases."""

    ensemble_class = ReleaseMetricCalculatorEnsemble


class ReleaseMetricCalculatorMixin(Generic[T]):
    """
    Split _analyze() to _check() and _extract().

    _check() decides whether _extract() must be called and thus deals with Optional[T].
    """

    def _analyze(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.array:
        fill_value = None
        dtype = object
        if self.has_nan:
            dtype = self.dtype
        elif self.nan is not None:
            fill_value = self.nan
            dtype = self.dtype
        result = np.full((len(min_times), len(facts)), fill_value, dtype)
        checked_mask = self._check(facts, min_times, max_times)
        extracted = np.repeat(self._extract(facts)[None, :], len(min_times), axis=0)
        result[checked_mask] = extracted[checked_mask]
        return result

    def _check(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
    ) -> np.ndarray:
        published = facts[ReleaseFacts.f.published].values
        return (min_times[:, None] <= published) & (published < max_times[:, None])

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class TagReleaseMetricCalculatorMixin(ReleaseMetricCalculatorMixin[T]):
    """Augment _check() to pass tag releases only."""

    def _check(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
    ) -> np.ndarray:
        return super()._check(facts, min_times, max_times) & (
            facts[matched_by_column].values == ReleaseMatch.tag
        )


class BranchReleaseMetricCalculatorMixin(ReleaseMetricCalculatorMixin[T]):
    """Augment _check() to pass branch releases only."""

    def _check(
        self,
        facts: pd.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
    ) -> np.ndarray:
        return super()._check(facts, min_times, max_times) & (
            facts[matched_by_column].values == ReleaseMatch.branch
        )


class ReleaseCounterMixin:
    """Count the number of matched releases."""

    may_have_negative_values = False
    metric = MetricInt

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        return np.full(len(facts), 1, self.dtype)


class ReleasePRsMixin:
    """Extract the number of PRs belonging to the matched release."""

    may_have_negative_values = False
    metric = MetricInt

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        return np.array(
            [len(arr) for arr in facts["prs_" + PullRequest.number.name]], dtype=self.dtype,
        )


class ReleaseCommitsMixin:
    """Extract the number of commits belonging to the matched release."""

    may_have_negative_values = False
    metric = MetricInt

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        return facts[ReleaseFacts.f.commits_count].values


class ReleaseLinesMixin:
    """Extract the sum of added + deleted lines in the commits belonging to the matched release."""

    may_have_negative_values = False
    metric = MetricInt

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        return facts[ReleaseFacts.f.additions].values + facts[ReleaseFacts.f.deletions].values


class ReleaseAgeMixin:
    """Extract the age of the matched release."""

    may_have_negative_values = False
    metric = MetricTimeDelta

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        return facts[ReleaseFacts.f.age].values.astype(self.dtype).view(int)


@register_metric(ReleaseMetricID.RELEASE_COUNT)
class ReleaseCounter(
    ReleaseCounterMixin, ReleaseMetricCalculatorMixin[int], SumMetricCalculator[int],
):
    """Count releases."""


@register_metric(ReleaseMetricID.TAG_RELEASE_COUNT)
class TagReleaseCounter(
    ReleaseCounterMixin, TagReleaseMetricCalculatorMixin[int], SumMetricCalculator[int],
):
    """Count tag releases."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_COUNT)
class BranchReleaseCounter(
    ReleaseCounterMixin, BranchReleaseMetricCalculatorMixin[int], SumMetricCalculator[int],
):
    """Count branch releases."""


@register_metric(ReleaseMetricID.RELEASE_PRS)
class ReleasePRsCounter(
    ReleasePRsMixin, ReleaseMetricCalculatorMixin[int], SumMetricCalculator[int],
):
    """Count released PRs."""


@register_metric(ReleaseMetricID.TAG_RELEASE_PRS)
class TagReleasePRsCounter(
    ReleasePRsMixin, TagReleaseMetricCalculatorMixin[int], SumMetricCalculator[int],
):
    """Count PRs released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_PRS)
class BranchReleasePRsCounter(
    ReleasePRsMixin, BranchReleaseMetricCalculatorMixin[int], SumMetricCalculator[int],
):
    """Count PRs released by branch."""


@register_metric(ReleaseMetricID.RELEASE_COMMITS)
class ReleaseCommitsCounter(
    ReleaseCommitsMixin, ReleaseMetricCalculatorMixin[int], SumMetricCalculator[int],
):
    """Count released commits."""


@register_metric(ReleaseMetricID.TAG_RELEASE_COMMITS)
class TagReleaseCommitsCounter(
    ReleaseCommitsMixin, TagReleaseMetricCalculatorMixin[int], SumMetricCalculator[int],
):
    """Count commits released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_COMMITS)
class BranchReleaseCommitsCounter(
    ReleaseCommitsMixin, BranchReleaseMetricCalculatorMixin[int], SumMetricCalculator[int],
):
    """Count commits released by branch."""


@register_metric(ReleaseMetricID.RELEASE_LINES)
class ReleaseLinesCounter(
    ReleaseLinesMixin, ReleaseMetricCalculatorMixin[int], SumMetricCalculator[int],
):
    """Count changed lines belonging to released commits."""


@register_metric(ReleaseMetricID.TAG_RELEASE_LINES)
class TagReleaseLinesCounter(
    ReleaseLinesMixin, TagReleaseMetricCalculatorMixin[int], SumMetricCalculator[int],
):
    """Count changed lines belonging to commits released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_LINES)
class BranchReleaseLinesCounter(
    ReleaseLinesMixin, BranchReleaseMetricCalculatorMixin[int], SumMetricCalculator[int],
):
    """Count changed lines belonging to commits released by branch."""


@register_metric(ReleaseMetricID.RELEASE_AVG_PRS)
class ReleasePRsCalculator(
    ReleasePRsMixin, ReleaseMetricCalculatorMixin[float], AverageMetricCalculator[float],
):
    """Measure average number of released PRs."""


@register_metric(ReleaseMetricID.TAG_RELEASE_AVG_PRS)
class TagReleasePRsCalculator(
    ReleasePRsMixin, TagReleaseMetricCalculatorMixin[float], AverageMetricCalculator[float],
):
    """Measure average number of PRs released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_AVG_PRS)
class BranchReleasePRsCalculator(
    ReleasePRsMixin, BranchReleaseMetricCalculatorMixin[float], AverageMetricCalculator[float],
):
    """Measure average number of PRs released by branch."""


@register_metric(ReleaseMetricID.RELEASE_AVG_COMMITS)
class ReleaseCommitsCalculator(
    ReleaseCommitsMixin, ReleaseMetricCalculatorMixin[float], AverageMetricCalculator[float],
):
    """Measure average number of released commits."""


@register_metric(ReleaseMetricID.TAG_RELEASE_AVG_COMMITS)
class TagReleaseCommitsCalculator(
    ReleaseCommitsMixin, TagReleaseMetricCalculatorMixin[float], AverageMetricCalculator[float],
):
    """Measure average number of commits released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_AVG_COMMITS)
class BranchReleaseCommitsCalculator(
    ReleaseCommitsMixin, BranchReleaseMetricCalculatorMixin[float], AverageMetricCalculator[float],
):
    """Measure average number of commits released by branch."""


@register_metric(ReleaseMetricID.RELEASE_AVG_LINES)
class ReleaseLinesCalculator(
    ReleaseLinesMixin, ReleaseMetricCalculatorMixin[float], AverageMetricCalculator[float],
):
    """Measure average number of changed lines belonging to released commits."""


@register_metric(ReleaseMetricID.TAG_RELEASE_AVG_LINES)
class TagReleaseLinesCalculator(
    ReleaseLinesMixin, TagReleaseMetricCalculatorMixin[float], AverageMetricCalculator[float],
):
    """Measure average number of changed lines belonging to commits released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_AVG_LINES)
class BranchReleaseLinesCalculator(
    ReleaseLinesMixin, BranchReleaseMetricCalculatorMixin[float], AverageMetricCalculator[float],
):
    """Measure average number of changed lines belonging to commits released by branch."""


@register_metric(ReleaseMetricID.RELEASE_AGE)
class ReleaseAgeCalculator(
    ReleaseAgeMixin, ReleaseMetricCalculatorMixin[timedelta], AverageMetricCalculator[timedelta],
):
    """Measure average release age."""


@register_metric(ReleaseMetricID.TAG_RELEASE_AGE)
class TagReleaseAgeCalculator(
    ReleaseAgeMixin,
    TagReleaseMetricCalculatorMixin[timedelta],
    AverageMetricCalculator[timedelta],
):
    """Measure average tag release age."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_AGE)
class BranchReleaseAgeCalculator(
    ReleaseAgeMixin,
    BranchReleaseMetricCalculatorMixin[timedelta],
    AverageMetricCalculator[timedelta],
):
    """Measure average branch release age."""
