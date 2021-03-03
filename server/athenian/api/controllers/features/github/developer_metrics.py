from typing import Collection, Dict, List, Sequence, Type, TypeVar

import numpy as np
import pandas as pd

from athenian.api.controllers.features.metric_calculator import BinnedMetricCalculator, \
    MetricCalculator, \
    MetricCalculatorEnsemble, SumMetricCalculator
from athenian.api.controllers.miners.github.developer import developer_identity_column, \
    DeveloperTopic

metric_calculators: Dict[str, Type[MetricCalculator]] = {}
T = TypeVar("T")


def register_metric(name: str):
    """Keep track of the release metric calculators."""
    assert isinstance(name, str)

    def register_with_name(cls: Type[MetricCalculator]):
        metric_calculators[name] = cls
        return cls

    return register_with_name


class DeveloperMetricCalculatorEnsemble(MetricCalculatorEnsemble):
    """MetricCalculatorEnsemble adapted for developers."""

    def __init__(self, *metrics: str, quantiles: Sequence[float]):
        """Initialize a new instance of ReleaseMetricCalculatorEnsemble class."""
        super().__init__(*metrics, quantiles=quantiles, class_mapping=metric_calculators)


class DeveloperBinnedMetricCalculator(BinnedMetricCalculator):
    """BinnedMetricCalculator adapted for developers."""

    ensemble_class = DeveloperMetricCalculatorEnsemble


class DeveloperTopicCounter(SumMetricCalculator[int]):
    """Count all `topic` events in each time interval."""

    may_have_negative_values = False
    dtype = int
    topic: str

    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **kwargs) -> np.array:
        result = np.full((len(min_times), len(facts)), None, object)
        column = facts[self.topic].values
        column_in_range = (min_times[:, None] <= column) & (column < max_times[:, None])
        result[column_in_range] = 1
        return result


for developer_topic in DeveloperTopic:
    class SpecificDeveloperTopicCounter(DeveloperTopicCounter):
        """Calculate %s metric.""" % developer_topic.value

        topic = developer_topic.name

    SpecificDeveloperTopicCounter.__name__ = "%sCounter" % "".join(
        (s[0].upper() + s[1:]) for s in developer_topic.name.split("_"))

    register_metric(developer_topic.value)(SpecificDeveloperTopicCounter)


def group_actions_by_developers(devs: Sequence[Collection[str]],
                                df: pd.DataFrame,
                                ) -> List[np.ndarray]:
    """Group developer actions by developer groups."""
    indexes = []
    identities = df[developer_identity_column].values.astype("U")
    for group in devs:
        if len(group) == 1:
            dev = next(iter(group))
            indexes.append(np.nonzero(identities == dev)[0])
            continue
        if isinstance(group, set):
            group = list(group)
        indexes.append(np.nonzero(np.in1d(identities, np.array(group, dtype="U")))[0])
    return indexes
