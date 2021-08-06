from datetime import datetime, timedelta
import itertools

import numpy as np
import pandas as pd
import pytest

from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.features.metric_calculator import AverageMetricCalculator, \
    MedianMetricCalculator
from athenian.api.controllers.features.statistics import mean_confidence_interval, \
    median_confidence_interval
from athenian.api.typing_utils import df_from_structs


def dt64arr(dt: datetime) -> np.ndarray:
    return np.array([dt], dtype="datetime64[ns]")


class DummyAverageMetricCalculator(AverageMetricCalculator):
    def _analyze(self,
                 facts: pd.DataFrame,
                 min_times: np.ndarray,
                 max_times: np.ndarray,
                 **_) -> np.ndarray:
        raise AssertionError("this must be never called")


@pytest.fixture
def square_centered_samples():
    data = (10 - np.arange(0, 21, dtype=int)) ** 2
    data[11:] *= -1
    return data


def test_mean_confidence_interval_positive():
    np.random.seed(8)
    data = np.random.lognormal(1, 2, 1000).astype(np.float32)
    mean, conf_min, conf_max = mean_confidence_interval(data, False)
    assert isinstance(mean, float)
    assert isinstance(conf_min, float)
    assert isinstance(conf_max, float)
    assert 20.7 < mean < 20.8
    assert 17.5 < conf_min < 18.0
    assert 23.5 < conf_max < 23.75


def test_mean_confidence_interval_negative(square_centered_samples):
    mean, conf_min, conf_max = mean_confidence_interval(square_centered_samples, True)
    assert isinstance(mean, int)
    assert isinstance(conf_min, int)
    assert isinstance(conf_max, int)
    assert mean == 0
    assert conf_min == -14
    assert conf_max == 14


def test_mean_confidence_interval_timedelta_positive():
    np.random.seed(8)
    data = (np.random.lognormal(1, 2, 1000) * 3600).astype("timedelta64[s]")
    mean, conf_min, conf_max = mean_confidence_interval(data, False)
    assert isinstance(mean, timedelta)
    assert isinstance(conf_min, timedelta)
    assert isinstance(conf_max, timedelta)
    assert timedelta(hours=20) < mean < timedelta(hours=21)
    assert timedelta(hours=17) < conf_min < timedelta(hours=18)
    assert timedelta(hours=23) < conf_max < timedelta(hours=24)


def test_metric_zero_division():
    m = Metric(value=np.int64(0),
               confidence_min=np.int64(0),
               confidence_max=np.int64(0),
               exists=True)
    assert m.confidence_score() == 100


def test_mean_confidence_interval_timedelta_negative(square_centered_samples):
    data = square_centered_samples.astype("timedelta64[s]")
    mean, conf_min, conf_max = mean_confidence_interval(data, True)
    assert isinstance(mean, timedelta)
    assert isinstance(conf_min, timedelta)
    assert isinstance(conf_max, timedelta)
    assert mean == timedelta(0)
    assert abs((conf_min - timedelta(seconds=-14)).total_seconds()) < 1
    assert abs((conf_max - timedelta(seconds=14)).total_seconds()) < 1


def test_mean_confidence_interval_empty():
    with pytest.raises(AssertionError):
        mean_confidence_interval(np.array([]), True)


def test_mean_confidence_interval_negative_list(square_centered_samples):
    mean, conf_min, conf_max = mean_confidence_interval(
        np.array(list(square_centered_samples)), True)
    assert isinstance(mean, int)
    assert isinstance(conf_min, int)
    assert isinstance(conf_max, int)
    assert mean == 0
    assert conf_min == -14
    assert conf_max == 14


def test_median_confidence_interval_int(square_centered_samples):
    mean, conf_min, conf_max = median_confidence_interval(square_centered_samples)
    assert isinstance(mean, int)
    assert isinstance(conf_min, int)
    assert isinstance(conf_max, int)
    assert mean == 0
    assert conf_min == -4
    assert conf_max == 4


def test_median_confidence_interval_timedelta(square_centered_samples):
    data = square_centered_samples.astype("timedelta64[s]")
    mean, conf_min, conf_max = median_confidence_interval(data)
    assert isinstance(mean, timedelta)
    assert isinstance(conf_min, timedelta)
    assert isinstance(conf_max, timedelta)
    assert mean == timedelta(0)
    assert conf_min == timedelta(seconds=-4)
    assert conf_max == timedelta(seconds=4)


def test_median_confidence_interval_empty():
    with pytest.raises(AssertionError):
        median_confidence_interval(np.array([]))


@pytest.mark.parametrize(
    "cls, negative, dtype",
    ((*t[0], t[1]) for t in itertools.product(
        [(DummyAverageMetricCalculator, False),
         (DummyAverageMetricCalculator, True),
         (MedianMetricCalculator, False)],
        [datetime, pd.Timestamp])))
def test_metric_calculator(pr_samples, cls, negative, dtype):
    class LeadTimeCalculator(cls):
        may_have_negative_values = negative
        dtype = "timedelta64[s]"

        def _analyze(self, facts: np.ndarray, min_times: np.ndarray, max_time: np.ndarray,
                     ) -> np.ndarray:
            return np.repeat((facts["released"] - facts["work_began"]).values[None, :],
                             len(min_times), axis=0)

    calc = LeadTimeCalculator(quantiles=(0, 0.99))
    calc._representative_time_interval_indexes = [0]
    assert len(calc.values) == 0 and isinstance(calc.values, list)
    calc = LeadTimeCalculator(quantiles=(0, 1))
    calc(df_from_structs(pr_samples(100)),
         dt64arr(datetime.utcnow()),
         dt64arr(datetime.utcnow()),
         None,
         np.array([[True] * 100]))
    m = calc.values[0][0]
    assert m.exists
    assert isinstance(m.value, timedelta)
    assert isinstance(m.confidence_min, timedelta)
    assert isinstance(m.confidence_max, timedelta)
    assert m.confidence_score() > 50
    assert timedelta(0) < m.value < timedelta(days=365 * 3 + 32)
    assert m.confidence_min < m.value < m.confidence_max
    calc.reset()
    assert len(calc.values) == 0
    calc.reset()
    calc._samples = np.zeros((1, 1), dtype=object)
    calc._samples[0, 0] = np.array([0], dtype=LeadTimeCalculator.dtype)
    m = calc.values[0][0]
    assert m.exists
    assert m.value == timedelta(0)
    assert m.confidence_min == timedelta(0)
    assert m.confidence_max == timedelta(0)
    assert m.confidence_score() == 100


def test_average_metric_calculator_zeros_nonnegative():
    calc = DummyAverageMetricCalculator(quantiles=(0, 1))
    calc.may_have_negative_values = False
    calc._samples = np.zeros((1, 1), dtype=object)
    calc._samples[0, 0] = np.full(3, timedelta(0), "timedelta64[s]")
    calc._representative_time_interval_indexes = [0]
    m = calc.values[0][0]
    assert m.exists
    assert m.value == timedelta(0)


def test_mean_confidence_interval_nan_confidence_nonnegative():
    m, cmin, cmax = mean_confidence_interval(
        np.array([0, 1] * 2, dtype="timedelta64[s]"), False)
    assert m == timedelta(seconds=0)
    assert cmin == m
    assert cmax == m


def test_mean_confidence_interval_nan_confidence_negative():
    m, cmin, cmax = mean_confidence_interval(np.array([1] * 3, dtype="timedelta64[s]"), True)
    assert m == timedelta(seconds=1)
    assert cmin == m
    assert cmax == m


def test_mean_confidence_interval_timedelta_positive_zeros():
    np.random.seed(8)
    mean, conf_min, conf_max = mean_confidence_interval(
        np.array([0] * 10 + [10] * 20 + [20] * 10 + [30] * 5 + [40] * 3 + [50]), False)
    assert isinstance(mean, int)
    assert isinstance(conf_min, int)
    assert isinstance(conf_max, int)
    assert mean == 14
    assert conf_min == 12
    assert conf_max == 16
    mean, conf_min, conf_max = mean_confidence_interval(np.array([0] * 10), False)
    assert isinstance(mean, int)
    assert isinstance(conf_min, int)
    assert isinstance(conf_max, int)
    assert mean == 0
    assert conf_min == 0
    assert conf_max == 0
    mean, conf_min, conf_max = mean_confidence_interval(np.array([0.0] * 10 + [1.0]), False)
    assert mean > 0


@pytest.mark.flaky(reruns=3)
def test_mean_confidence_interval_nonnegative_overflow():
    arr = np.array([5689621, 5448983, 5596389, 5468130, 4722905, 5000224, 4723318,  # noqa
                    4063452, 4406564, 3728378, 4064774, 3874963, 3693545, 3618715,  # noqa
                    3208079, 3207821, 3116119, 2656753, 2424436, 2408454, 2058306,  # noqa
                    1884453, 1907901, 1221960, 1013571, 1012170,       0,       0,  # noqa
                          0,       0,       0,     659,       0,       0,       0,  # noqa
                        682,       0,       0,       0,     693,       0,       0,  # noqa
                          0,     666,       0,     719,       0,       0,       0,  # noqa
                        715,       0,   94176,       0,       0,       0,       0,  # noqa
                        742,       0,     683,       0,       0,       0,       0,  # noqa
                          0,       0,       0,       0,       0],                   # noqa
                   dtype="timedelta64[s]")
    m, conf_min, conf_max = mean_confidence_interval(arr, False)
    assert m.days == 15
    assert conf_min.days in (11, 12)
    assert conf_max.days == 18
