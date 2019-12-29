import numpy as np
import pandas as pd
import pytest

from athenian.api.controllers.features.github.pull_request import mean_confidence_interval, \
    median_confidence_interval


@pytest.fixture
def square_centered_samples():
    data = (10 - np.arange(0, 21, dtype=int)) ** 2
    data[11:] *= -1
    return data


def test_mean_confidence_interval_positive():
    np.random.seed(8)
    data = np.random.lognormal(1, 2, 1000).astype(np.float32)
    mean, conf_min, conf_max = mean_confidence_interval(data, False)
    assert isinstance(mean, np.float32)
    assert isinstance(conf_min, np.float32)
    assert isinstance(conf_max, np.float32)
    assert 20.7 < mean < 20.8
    assert 18.93 < conf_min < 18.94
    assert 29.5 < conf_max < 29.6


def test_mean_confidence_interval_negative(square_centered_samples):
    mean, conf_min, conf_max = mean_confidence_interval(square_centered_samples, True)
    assert isinstance(mean, np.int64)
    assert isinstance(conf_min, np.int64)
    assert isinstance(conf_max, np.int64)
    assert mean == 0
    assert conf_min == -22
    assert conf_max == 22


def test_mean_confidence_interval_timedelta_positive():
    np.random.seed(8)
    data = pd.Series(
        (np.random.lognormal(1, 2, 1000) * 1_000_000_000 * 3600).astype(np.timedelta64))
    mean, conf_min, conf_max = mean_confidence_interval(data, False)
    assert isinstance(mean, pd.Timedelta)
    assert isinstance(conf_min, pd.Timedelta)
    assert isinstance(conf_max, pd.Timedelta)
    assert pd.Timedelta(hours=20) < mean < pd.Timedelta(hours=21)
    assert pd.Timedelta(hours=18) < conf_min < pd.Timedelta(hours=19)
    assert pd.Timedelta(hours=29) < conf_max < pd.Timedelta(hours=30)


def test_mean_confidence_interval_timedelta_negative(square_centered_samples):
    data = pd.Series((square_centered_samples * 1_000_000_000).astype(np.timedelta64))
    mean, conf_min, conf_max = mean_confidence_interval(data, True)
    assert isinstance(mean, pd.Timedelta)
    assert isinstance(conf_min, pd.Timedelta)
    assert isinstance(conf_max, pd.Timedelta)
    assert mean == pd.Timedelta(0)
    assert abs((conf_min - pd.Timedelta(seconds=-22)).total_seconds()) < 1
    assert abs((conf_max - pd.Timedelta(seconds=22)).total_seconds()) < 1


def test_mean_confidence_interval_empty():
    mean, conf_min, conf_max = mean_confidence_interval([], True)
    assert mean is None
    assert conf_min is None
    assert conf_max is None


def test_mean_confidence_interval_negative_list(square_centered_samples):
    mean, conf_min, conf_max = mean_confidence_interval(list(square_centered_samples), True)
    assert isinstance(mean, np.int64)
    assert isinstance(conf_min, np.int64)
    assert isinstance(conf_max, np.int64)
    assert mean == 0
    assert conf_min == -22
    assert conf_max == 22


def test_median_confidence_interval_int(square_centered_samples):
    mean, conf_min, conf_max = median_confidence_interval(square_centered_samples)
    assert isinstance(mean, np.int64)
    assert isinstance(conf_min, np.int64)
    assert isinstance(conf_max, np.int64)
    assert mean == 0
    assert conf_min == -16
    assert conf_max == 16


def test_median_confidence_interval_timedelta(square_centered_samples):
    data = pd.Series((square_centered_samples * 1_000_000_000).astype(np.timedelta64))
    mean, conf_min, conf_max = median_confidence_interval(data)
    assert isinstance(mean, pd.Timedelta)
    assert isinstance(conf_min, pd.Timedelta)
    assert isinstance(conf_max, pd.Timedelta)
    assert mean == pd.Timedelta(0)
    assert conf_min == pd.Timedelta(seconds=-16)
    assert conf_max == pd.Timedelta(seconds=16)


def test_median_confidence_interval_empty():
    mean, conf_min, conf_max = median_confidence_interval([])
    assert mean is None
    assert conf_min is None
    assert conf_max is None
