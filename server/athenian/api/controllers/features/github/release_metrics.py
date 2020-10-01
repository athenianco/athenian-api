from datetime import datetime, timedelta
from typing import Dict, Generic, Sequence, Type

import numpy as np
import pandas as pd

from athenian.api.controllers.features.metric_calculator import AverageMetricCalculator, \
    BinnedMetricCalculator, MetricCalculator, MetricCalculatorEnsemble, SumMetricCalculator
from athenian.api.controllers.miners.github.released_pr import matched_by_column
from athenian.api.controllers.miners.types import T
from athenian.api.controllers.settings import ReleaseMatch
from athenian.api.models.metadata.github import PullRequest
from athenian.api.models.web import ReleaseMetricID

metric_calculators: Dict[str, Type[MetricCalculator]] = {}


def register_metric(name: str):
    """Keep track of the release metric calculators."""
    assert isinstance(name, str)

    def register_with_name(cls: Type[MetricCalculator]):
        metric_calculators[name] = cls
        return cls

    return register_with_name


class ReleaseMetricCalculatorEnsemble(MetricCalculatorEnsemble):
    """MetricCalculatorEnsemble adapted for pull requests."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of ReleaseMetricCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=metric_calculators)


class ReleaseBinnedMetricCalculator(BinnedMetricCalculator):
    """BinnedMetricCalculator adapted for pull requests."""

    def __init__(self,
                 metrics: Sequence[str],
                 time_intervals: Sequence[datetime],
                 quantiles: Sequence[float]):
        """Initialize a new instance of ReleaseBinnedMetricCalculator class."""
        super().__init__(metrics=metrics, time_intervals=time_intervals, quantiles=quantiles,
                         class_mapping=metric_calculators,
                         start_time_getter=lambda r: r.published,
                         finish_time_getter=lambda r: r.published)


class ReleaseMetricCalculatorMixin(Generic[T]):
    """
    Split _analyze() to _check() and _extract().

    _check() decides whether _extract() must be called and thus deals with Optional[T].
    """

    def _analyze(self, facts: np.ndarray, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.array:
        result = np.full(len(facts), None, object)
        indexes = np.where(self._check(facts, min_time, max_time))[0]
        result[indexes] = self._extract(facts.take(indexes))
        return result

    def _check(self, facts: np.ndarray, min_time: datetime, max_time: datetime) -> np.ndarray:
        dtype = facts["published"].dtype
        min_time = np.array(min_time, dtype=dtype)
        max_time = np.array(max_time, dtype=dtype)
        published = facts["published"].values
        return (min_time <= published) & (published < max_time)

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class TagReleaseMetricCalculatorMixin(ReleaseMetricCalculatorMixin[T]):
    """Augment _check() to pass tag releases only."""

    def _check(self, facts: np.ndarray, min_time: datetime, max_time: datetime) -> np.ndarray:
        return super()._check(facts, min_time, max_time) & \
            (facts[matched_by_column] == ReleaseMatch.tag)


class BranchReleaseMetricCalculatorMixin(ReleaseMetricCalculatorMixin[T]):
    """Augment _check() to pass branch releases only."""

    def _check(self, facts: np.ndarray, min_time: datetime, max_time: datetime) -> np.ndarray:
        return super()._check(facts, min_time, max_time) & \
            (facts[matched_by_column] == ReleaseMatch.branch)


class ReleaseCounterMixin:
    """Count the number of matched releases."""

    may_have_negative_values = False
    dtype = int

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        return np.full(len(facts), 1)


class ReleasePRsMixin:
    """Extract the number of PRs belonging to the matched release."""

    may_have_negative_values = False
    dtype = int

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        return np.array([len(prs[PullRequest.number.key]) for prs in facts["prs"].values])


class ReleaseCommitsMixin:
    """Extract the number of commits belonging to the matched release."""

    may_have_negative_values = False
    dtype = int

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        return facts["commits_count"].values


class ReleaseLinesMixin:
    """Extract the sum of added + deleted lines in the commits belonging to the matched release."""

    may_have_negative_values = False
    dtype = int

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        return facts["additions"].values + facts["deletions"].values


class ReleaseAgeMixin:
    """Extract the age of the matched release."""

    may_have_negative_values = False
    dtype = "timedelta64[s]"

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        return facts["age"].values.astype(self.dtype).view(int)


@register_metric(ReleaseMetricID.RELEASE_COUNT)
class ReleaseCounter(ReleaseCounterMixin,
                     ReleaseMetricCalculatorMixin[int],
                     SumMetricCalculator[int]):
    """Count releases."""


@register_metric(ReleaseMetricID.TAG_RELEASE_COUNT)
class TagReleaseCounter(ReleaseCounterMixin,
                        TagReleaseMetricCalculatorMixin[int],
                        SumMetricCalculator[int]):
    """Count tag releases."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_COUNT)
class BranchReleaseCounter(ReleaseCounterMixin,
                           BranchReleaseMetricCalculatorMixin[int],
                           SumMetricCalculator[int]):
    """Count branch releases."""


@register_metric(ReleaseMetricID.RELEASE_PRS)
class ReleasePRsCounter(ReleasePRsMixin,
                        ReleaseMetricCalculatorMixin[int],
                        SumMetricCalculator[int]):
    """Count released PRs."""


@register_metric(ReleaseMetricID.TAG_RELEASE_PRS)
class TagReleasePRsCounter(ReleasePRsMixin,
                           TagReleaseMetricCalculatorMixin[int],
                           SumMetricCalculator[int]):
    """Count PRs released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_PRS)
class BranchReleasePRsCounter(ReleasePRsMixin,
                              BranchReleaseMetricCalculatorMixin[int],
                              SumMetricCalculator[int]):
    """Count PRs released by branch."""


@register_metric(ReleaseMetricID.RELEASE_COMMITS)
class ReleaseCommitsCounter(ReleaseCommitsMixin,
                            ReleaseMetricCalculatorMixin[int],
                            SumMetricCalculator[int]):
    """Count released commits."""


@register_metric(ReleaseMetricID.TAG_RELEASE_COMMITS)
class TagReleaseCommitsCounter(ReleaseCommitsMixin,
                               TagReleaseMetricCalculatorMixin[int],
                               SumMetricCalculator[int]):
    """Count commits released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_COMMITS)
class BranchReleaseCommitsCounter(ReleaseCommitsMixin,
                                  BranchReleaseMetricCalculatorMixin[int],
                                  SumMetricCalculator[int]):
    """Count commits released by branch."""


@register_metric(ReleaseMetricID.RELEASE_LINES)
class ReleaseLinesCounter(ReleaseLinesMixin,
                          ReleaseMetricCalculatorMixin[int],
                          SumMetricCalculator[int]):
    """Count changed lines belonging to released commits."""


@register_metric(ReleaseMetricID.TAG_RELEASE_LINES)
class TagReleaseLinesCounter(ReleaseLinesMixin,
                             TagReleaseMetricCalculatorMixin[int],
                             SumMetricCalculator[int]):
    """Count changed lines belonging to commits released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_LINES)
class BranchReleaseLinesCounter(ReleaseLinesMixin,
                                BranchReleaseMetricCalculatorMixin[int],
                                SumMetricCalculator[int]):
    """Count changed lines belonging to commits released by branch."""


@register_metric(ReleaseMetricID.RELEASE_AVG_PRS)
class ReleasePRsCalculator(ReleasePRsMixin,
                           ReleaseMetricCalculatorMixin[float],
                           AverageMetricCalculator[float]):
    """Measure average number of released PRs."""


@register_metric(ReleaseMetricID.TAG_RELEASE_AVG_PRS)
class TagReleasePRsCalculator(ReleasePRsMixin,
                              TagReleaseMetricCalculatorMixin[float],
                              AverageMetricCalculator[float]):
    """Measure average number of PRs released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_AVG_PRS)
class BranchReleasePRsCalculator(ReleasePRsMixin,
                                 BranchReleaseMetricCalculatorMixin[float],
                                 AverageMetricCalculator[float]):
    """Measure average number of PRs released by branch."""


@register_metric(ReleaseMetricID.RELEASE_AVG_COMMITS)
class ReleaseCommitsCalculator(ReleaseCommitsMixin,
                               ReleaseMetricCalculatorMixin[float],
                               AverageMetricCalculator[float]):
    """Measure average number of released commits."""


@register_metric(ReleaseMetricID.TAG_RELEASE_AVG_COMMITS)
class TagReleaseCommitsCalculator(ReleaseCommitsMixin,
                                  TagReleaseMetricCalculatorMixin[float],
                                  AverageMetricCalculator[float]):
    """Measure average number of commits released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_AVG_COMMITS)
class BranchReleaseCommitsCalculator(ReleaseCommitsMixin,
                                     BranchReleaseMetricCalculatorMixin[float],
                                     AverageMetricCalculator[float]):
    """Measure average number of commits released by branch."""


@register_metric(ReleaseMetricID.RELEASE_AVG_LINES)
class ReleaseLinesCalculator(ReleaseLinesMixin,
                             ReleaseMetricCalculatorMixin[float],
                             AverageMetricCalculator[float]):
    """Measure average number of changed lines belonging to released commits."""


@register_metric(ReleaseMetricID.TAG_RELEASE_AVG_LINES)
class TagReleaseLinesCalculator(ReleaseLinesMixin,
                                TagReleaseMetricCalculatorMixin[float],
                                AverageMetricCalculator[float]):
    """Measure average number of changed lines belonging to commits released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_AVG_LINES)
class BranchReleaseLinesCalculator(ReleaseLinesMixin,
                                   BranchReleaseMetricCalculatorMixin[float],
                                   AverageMetricCalculator[float]):
    """Measure average number of changed lines belonging to commits released by branch."""


@register_metric(ReleaseMetricID.RELEASE_AGE)
class ReleaseAgeCalculator(ReleaseAgeMixin,
                           ReleaseMetricCalculatorMixin[timedelta],
                           AverageMetricCalculator[timedelta]):
    """Measure average release age."""


@register_metric(ReleaseMetricID.TAG_RELEASE_AGE)
class TagReleaseAgeCalculator(ReleaseAgeMixin,
                              TagReleaseMetricCalculatorMixin[timedelta],
                              AverageMetricCalculator[timedelta]):
    """Measure average tag release age."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_AGE)
class BranchReleaseAgeCalculator(ReleaseAgeMixin,
                                 BranchReleaseMetricCalculatorMixin[timedelta],
                                 AverageMetricCalculator[timedelta]):
    """Measure average branch release age."""
