from datetime import datetime, timedelta
from typing import Dict, Generic, Optional, Sequence, Type

from athenian.api.controllers.features.metric_calculator import AverageMetricCalculator, \
    BinnedMetricCalculator, \
    MetricCalculator, \
    MetricCalculatorEnsemble, SumMetricCalculator
from athenian.api.controllers.miners.types import ReleaseFacts, T
from athenian.api.controllers.settings import ReleaseMatch
from athenian.api.models.web import ReleaseMetricID

metric_calculators: Dict[str, Type[MetricCalculator]] = {}


def register_metric(name: str):
    """Keep track of the release metric calculators."""
    assert isinstance(name, str)

    def register_with_name(cls: Type[MetricCalculator]):
        metric_calculators[name] = cls
        return cls

    return register_with_name


class ReleaseMetricCalculatorEnsemble(MetricCalculatorEnsemble[T]):
    """MetricCalculatorEnsemble adapted for pull requests."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of ReleaseMetricCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=metric_calculators)


class ReleaseBinnedMetricCalculator(BinnedMetricCalculator[T]):
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

    def _analyze(self, facts: ReleaseFacts, min_time: datetime, max_time: datetime,
                 **kwargs) -> Optional[T]:
        if not self._check(facts, min_time, max_time):
            return None
        return self._extract(facts)

    def _check(self, facts: ReleaseFacts, min_time: datetime, max_time: datetime) -> bool:
        return min_time <= facts.published < max_time

    def _extract(self, facts: ReleaseFacts) -> T:
        raise NotImplementedError


class TagReleaseMetricCalculatorMixin(ReleaseMetricCalculatorMixin[T]):
    """Augment _check() to pass tag releases only."""

    def _check(self, facts: ReleaseFacts, min_time: datetime, max_time: datetime) -> Optional[T]:
        return super()._check(facts, min_time, max_time) and facts.matched_by == ReleaseMatch.tag


class BranchReleaseMetricCalculatorMixin(ReleaseMetricCalculatorMixin[T]):
    """Augment _check() to pass branch releases only."""

    def _check(self, facts: ReleaseFacts, min_time: datetime, max_time: datetime) -> Optional[T]:
        return super()._check(facts, min_time, max_time) and \
            facts.matched_by == ReleaseMatch.branch


class ReleaseCounterMixin:
    """Count the number of matched release."""

    may_have_negative_values = False

    def _extract(self, facts: ReleaseFacts) -> int:
        return 1


class ReleasePRsMixin:
    """Extract the number of PRs belonging to the matched release."""

    may_have_negative_values = False

    def _extract(self, facts: ReleaseFacts) -> int:
        return facts.prs_count


class ReleaseCommitsMixin:
    """Extract the number of commits belonging to the matched release."""

    may_have_negative_values = False

    def _extract(self, facts: ReleaseFacts) -> int:
        return facts.commits_count


class ReleaseLinesMixin:
    """Extract the sum of added + deleted lines in the commits belonging to the matched release."""

    may_have_negative_values = False

    def _extract(self, facts: ReleaseFacts) -> int:
        return facts.additions + facts.deletions


class ReleaseAgeMixin:
    """Extract the age of the matched release."""

    may_have_negative_values = False

    def _extract(self, facts: ReleaseFacts) -> timedelta:
        return facts.age


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
