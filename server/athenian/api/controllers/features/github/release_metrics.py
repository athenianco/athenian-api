from datetime import timedelta
from typing import Dict, Generic, Sequence, Type, TypeVar

import numpy as np
import pandas as pd

from athenian.api.controllers.features.metric_calculator import AverageMetricCalculator, \
    BinnedMetricCalculator, MetricCalculator, MetricCalculatorEnsemble, SumMetricCalculator
from athenian.api.controllers.miners.github.released_pr import matched_by_column
from athenian.api.controllers.miners.types import ReleaseFacts
from athenian.api.controllers.settings import ReleaseMatch
from athenian.api.models.metadata.github import PullRequest
from athenian.api.models.web import ReleaseMetricID


metric_calculators: Dict[str, Type[MetricCalculator]] = {}
T = TypeVar("T")


def register_metric(name: str):
    """Keep track of the release metric calculators."""
    assert isinstance(name, str)

    def register_with_name(cls: Type[MetricCalculator]):
        metric_calculators[name] = cls
        return cls

    return register_with_name


class ReleaseMetricCalculatorEnsemble(MetricCalculatorEnsemble):
    """MetricCalculatorEnsemble adapted for releases."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of ReleaseMetricCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=metric_calculators)


class ReleaseBinnedMetricCalculator(BinnedMetricCalculator):
    """BinnedMetricCalculator adapted for releases."""

    ensemble_class = ReleaseMetricCalculatorEnsemble


class ReleaseMetricCalculatorMixin(Generic[T]):
    """
    Split _analyze() to _check() and _extract().

    _check() decides whether _extract() must be called and thus deals with Optional[T].
    """

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.array:
        result = np.full((len(min_times), len(facts)), None, object)
        checked_mask = self._check(facts, min_times, max_times)
        extracted = np.repeat(self._extract(facts)[None, :], len(min_times), axis=0)
        result[checked_mask] = extracted[checked_mask]
        return result

    def _check(self,
               facts: pd.DataFrame,
               min_times: np.ndarray,
               max_times: np.ndarray) -> np.ndarray:
        published = facts[ReleaseFacts.published.__name__].values
        return (min_times[:, None] <= published) & (published < max_times[:, None])

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class TagReleaseMetricCalculatorMixin(ReleaseMetricCalculatorMixin[T]):
    """Augment _check() to pass tag releases only."""

    def _check(self,
               facts: pd.DataFrame,
               min_times: np.ndarray,
               max_times: np.ndarray) -> np.ndarray:
        return super()._check(facts, min_times, max_times) & \
            (facts[matched_by_column].values == ReleaseMatch.tag)


class BranchReleaseMetricCalculatorMixin(ReleaseMetricCalculatorMixin[T]):
    """Augment _check() to pass branch releases only."""

    def _check(self,
               facts: pd.DataFrame,
               min_times: np.ndarray,
               max_times: np.ndarray) -> np.ndarray:
        return super()._check(facts, min_times, max_times) & \
            (facts[matched_by_column].values == ReleaseMatch.branch)


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
        return np.array([len(prs[PullRequest.number.key])
                         for prs in facts[ReleaseFacts.prs.__name__].values])


class ReleaseCommitsMixin:
    """Extract the number of commits belonging to the matched release."""

    may_have_negative_values = False
    dtype = int

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        return facts[ReleaseFacts.commits_count.__name__].values


class ReleaseLinesMixin:
    """Extract the sum of added + deleted lines in the commits belonging to the matched release."""

    may_have_negative_values = False
    dtype = int

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        return (facts[ReleaseFacts.additions.__name__].values +
                facts[ReleaseFacts.deletions.__name__].values)


class ReleaseAgeMixin:
    """Extract the age of the matched release."""

    may_have_negative_values = False
    dtype = "timedelta64[s]"

    def _extract(self, facts: pd.DataFrame) -> np.ndarray:
        return facts[ReleaseFacts.age.__name__].values.astype(self.dtype).view(int)


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
