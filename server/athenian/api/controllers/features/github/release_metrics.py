from datetime import datetime, timedelta
from typing import Dict, Optional, Sequence, Type

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
                         start_time_getter=lambda r: r.published_at,
                         finish_time_getter=lambda r: r.published_at)


class ReleaseMetricCalculator(MetricCalculator[T]):
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


class TagReleaseMetricCalculator(ReleaseMetricCalculator[T]):
    """Augment _check() to pass tag releases only."""

    def _check(self, facts: ReleaseFacts, min_time: datetime, max_time: datetime) -> Optional[T]:
        return super()._check(facts, min_time, max_time) and facts.matched_by == ReleaseMatch.tag


class BranchReleaseMetricCalculator(ReleaseMetricCalculator[T]):
    """Augment _check() to pass branch releases only."""

    def _check(self, facts: ReleaseFacts, min_time: datetime, max_time: datetime) -> Optional[T]:
        return super()._check(facts, min_time, max_time) and \
            facts.matched_by == ReleaseMatch.branch


class ReleaseCounterMixin:
    """Count the number of matched release."""

    def _extract(self, facts: ReleaseFacts) -> int:
        return 1


class ReleasePRsMixin:
    """Extract the number of PRs belonging to the matched release."""

    def _extract(self, facts: ReleaseFacts) -> int:
        return facts.prs_count


class ReleaseCommitsMixin:
    """Extract the number of commits belonging to the matched release."""

    def _extract(self, facts: ReleaseFacts) -> int:
        return facts.commits_count


class ReleaseLinesMixin:
    """Extract the sum of added + deleted lines in the commits belonging to the matched release."""

    def _extract(self, facts: ReleaseFacts) -> int:
        return facts.additions + facts.deletions


class ReleaseAgeMixin:
    """Extract the age of the matched release."""

    def _extract(self, facts: ReleaseFacts) -> timedelta:
        return facts.age


@register_metric(ReleaseMetricID.RELEASE_COUNT)
class ReleaseCounter(SumMetricCalculator[int],
                     ReleaseMetricCalculator[int],
                     ReleaseCounterMixin):
    """Count releases."""


@register_metric(ReleaseMetricID.TAG_RELEASE_COUNT)
class TagReleaseCounter(SumMetricCalculator[int],
                        TagReleaseMetricCalculator[int],
                        ReleaseCounterMixin):
    """Count tag releases."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_COUNT)
class BranchReleaseCounter(SumMetricCalculator[int],
                           BranchReleaseMetricCalculator[int],
                           ReleaseCounterMixin):
    """Count branch releases."""


@register_metric(ReleaseMetricID.RELEASE_PRS)
class ReleasePRsCounter(SumMetricCalculator[int],
                        ReleaseMetricCalculator[int],
                        ReleasePRsMixin):
    """Count released PRs."""


@register_metric(ReleaseMetricID.TAG_RELEASE_PRS)
class TagReleasePRsCounter(SumMetricCalculator[int],
                           TagReleaseMetricCalculator[int],
                           ReleasePRsMixin):
    """Count PRs released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_PRS)
class BranchReleasePRsCounter(SumMetricCalculator[int],
                              BranchReleaseMetricCalculator[int],
                              ReleasePRsMixin):
    """Count PRs released by branch."""


@register_metric(ReleaseMetricID.RELEASE_COMMITS)
class ReleaseCommitsCounter(SumMetricCalculator[int],
                            ReleaseMetricCalculator[int],
                            ReleaseCommitsMixin):
    """Count released commits."""


@register_metric(ReleaseMetricID.TAG_RELEASE_COMMITS)
class TagReleaseCommitsCounter(SumMetricCalculator[int],
                               TagReleaseMetricCalculator[int],
                               ReleaseCommitsMixin):
    """Count commits released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_COMMITS)
class BranchReleaseCommitsCounter(SumMetricCalculator[int],
                                  BranchReleaseMetricCalculator[int],
                                  ReleaseCommitsMixin):
    """Count commits released by branch."""


@register_metric(ReleaseMetricID.RELEASE_LINES)
class ReleaseLinesCounter(SumMetricCalculator[int],
                          ReleaseMetricCalculator[int],
                          ReleaseLinesMixin):
    """Count changed lines belonging to released commits."""


@register_metric(ReleaseMetricID.TAG_RELEASE_LINES)
class TagReleaseLinesCounter(SumMetricCalculator[int],
                             TagReleaseMetricCalculator[int],
                             ReleaseLinesMixin):
    """Count changed lines belonging to commits released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_LINES)
class BranchReleaseLinesCounter(SumMetricCalculator[int],
                                BranchReleaseMetricCalculator[int],
                                ReleaseLinesMixin):
    """Count changed lines belonging to commits released by branch."""


@register_metric(ReleaseMetricID.RELEASE_AVG_PRS)
class ReleasePRsCalculator(AverageMetricCalculator[float],
                           ReleaseMetricCalculator[float],
                           ReleasePRsMixin):
    """Measure average number of released PRs."""


@register_metric(ReleaseMetricID.TAG_RELEASE_AVG_PRS)
class TagReleasePRsCalculator(AverageMetricCalculator[float],
                              TagReleaseMetricCalculator[float],
                              ReleasePRsMixin):
    """Measure average number of PRs released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_AVG_PRS)
class BranchReleasePRsCalculator(AverageMetricCalculator[float],
                                 BranchReleaseMetricCalculator[float],
                                 ReleasePRsMixin):
    """Measure average number of PRs released by branch."""


@register_metric(ReleaseMetricID.RELEASE_AVG_COMMITS)
class ReleaseCommitsCalculator(AverageMetricCalculator[float],
                               ReleaseMetricCalculator[float],
                               ReleaseCommitsMixin):
    """Measure average number of released commits."""


@register_metric(ReleaseMetricID.TAG_RELEASE_AVG_COMMITS)
class TagReleaseCommitsCalculator(AverageMetricCalculator[float],
                                  TagReleaseMetricCalculator[float],
                                  ReleaseCommitsMixin):
    """Measure average number of commits released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_AVG_COMMITS)
class BranchReleaseCommitsCalculator(AverageMetricCalculator[float],
                                     BranchReleaseMetricCalculator[float],
                                     ReleaseCommitsMixin):
    """Measure average number of commits released by branch."""


@register_metric(ReleaseMetricID.RELEASE_AVG_LINES)
class ReleaseLinesCalculator(AverageMetricCalculator[float],
                             ReleaseMetricCalculator[float],
                             ReleaseLinesMixin):
    """Measure average number of changed lines belonging to released commits."""


@register_metric(ReleaseMetricID.TAG_RELEASE_AVG_LINES)
class TagReleaseLinesCalculator(AverageMetricCalculator[float],
                                TagReleaseMetricCalculator[float],
                                ReleaseLinesMixin):
    """Measure average number of changed lines belonging to commits released by tag."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_AVG_LINES)
class BranchReleaseLinesCalculator(AverageMetricCalculator[float],
                                   BranchReleaseMetricCalculator[float],
                                   ReleaseLinesMixin):
    """Measure average number of changed lines belonging to commits released by branch."""


@register_metric(ReleaseMetricID.RELEASE_AGE)
class ReleaseAgeCalculator(AverageMetricCalculator[timedelta],
                           ReleaseMetricCalculator[timedelta],
                           ReleaseAgeMixin):
    """Measure average release age."""


@register_metric(ReleaseMetricID.TAG_RELEASE_AGE)
class TagReleaseAgeCalculator(AverageMetricCalculator[timedelta],
                              TagReleaseMetricCalculator[timedelta],
                              ReleaseAgeMixin):
    """Measure average tag release age."""


@register_metric(ReleaseMetricID.BRANCH_RELEASE_AGE)
class BranchReleaseAgeCalculator(AverageMetricCalculator[timedelta],
                                 BranchReleaseMetricCalculator[timedelta],
                                 ReleaseAgeMixin):
    """Measure average branch release age."""
