from datetime import timedelta

import numpy as np
import pytest

from athenian.api.internal.features.histogram import calculate_histogram, Scale


def test_calculate_histogram_empty_log():
    h = calculate_histogram(np.array([]), Scale.LOG, 10, None)
    assert h.bins == 10
    assert h.scale == Scale.LOG
    assert h.ticks == []
    assert h.frequencies == []
    assert h.interquartile == (0, 0)


def test_calculate_histogram_empty_ticks():
    h = calculate_histogram(np.array([]), None, None, [1, 2, 3])
    assert h.bins is None
    assert h.scale == Scale.LINEAR
    assert h.ticks == [1, 2, 3]
    assert h.frequencies == [0, 0]
    assert h.interquartile == (0, 0)


def test_calculate_histogram_timedelta_linear_fixed():
    h = calculate_histogram(
        np.array([timedelta(seconds=n) for n in range(1000)], dtype="timedelta64[s]"),
        Scale.LINEAR, 10, None)
    assert h.bins == 10
    assert h.scale == Scale.LINEAR
    assert h.ticks == [timedelta(seconds=60), timedelta(seconds=153), timedelta(seconds=247),
                       timedelta(seconds=341), timedelta(seconds=435), timedelta(seconds=529),
                       timedelta(seconds=623), timedelta(seconds=717), timedelta(seconds=811),
                       timedelta(seconds=905), timedelta(seconds=999)]
    assert h.frequencies == [154, 94, 94, 94, 94, 94, 94, 94, 94, 94]
    assert h.interquartile == (timedelta(seconds=249), timedelta(seconds=749))


def test_calculate_histogram_timedelta_linear_auto():
    h = calculate_histogram(
        np.array([timedelta(seconds=n) for n in range(1000)], dtype="timedelta64[s]"),
        Scale.LINEAR, 0, None)
    assert h.bins == 12
    assert h.scale == Scale.LINEAR
    assert h.ticks == [timedelta(seconds=60), timedelta(seconds=138), timedelta(seconds=216),
                       timedelta(seconds=294), timedelta(seconds=373), timedelta(seconds=451),
                       timedelta(seconds=529), timedelta(seconds=607), timedelta(seconds=686),
                       timedelta(seconds=764), timedelta(seconds=842), timedelta(seconds=920),
                       timedelta(seconds=999)]
    assert h.frequencies == [139, 78, 78, 78, 79, 78, 78, 78, 79, 78, 78, 79]
    assert h.interquartile == (timedelta(seconds=249), timedelta(seconds=749))


def test_calculate_histogram_timedelta_log_fixed():
    h = calculate_histogram(
        np.array([timedelta(seconds=n) for n in range(1000)], dtype="timedelta64[s]"),
        Scale.LOG, 10, None)
    assert h.bins == 10
    assert h.scale == Scale.LOG
    assert h.ticks[:-1] == [
        timedelta(seconds=60), timedelta(seconds=79), timedelta(seconds=105),
        timedelta(seconds=139), timedelta(seconds=184), timedelta(seconds=244),
        timedelta(seconds=324), timedelta(seconds=429), timedelta(seconds=569),
        timedelta(seconds=754)]
    # depends on the version of numpy
    assert h.ticks[-1] in (timedelta(seconds=999), timedelta(seconds=998))
    assert h.frequencies == [80, 26, 34, 45, 60, 80, 105, 140, 185, 245]
    assert h.interquartile == (timedelta(seconds=249), timedelta(seconds=749))


def test_calculate_histogram_timedelta_log_auto():
    h = calculate_histogram(
        np.array([timedelta(seconds=n) for n in range(1000)], dtype="timedelta64[s]"),
        Scale.LOG, 0, None)
    assert h.bins == 15
    assert h.scale == Scale.LOG
    assert h.ticks[:-1] == [
        timedelta(seconds=60), timedelta(seconds=72), timedelta(seconds=87),
        timedelta(seconds=105), timedelta(seconds=127), timedelta(seconds=153),
        timedelta(seconds=184), timedelta(seconds=222), timedelta(seconds=268),
        timedelta(seconds=324), timedelta(seconds=391), timedelta(seconds=471),
        timedelta(seconds=569), timedelta(seconds=686), timedelta(seconds=828)]
    # depends on the version of numpy
    assert h.ticks[-1] in (timedelta(seconds=999), timedelta(seconds=998))
    assert h.frequencies == [73, 15, 18, 22, 26, 31, 38, 46, 56, 67, 80, 98, 117, 142, 171]
    assert h.interquartile == (timedelta(seconds=249), timedelta(seconds=749))


def test_calculate_histogram_negative_log_error():
    with pytest.raises(ValueError):
        calculate_histogram(np.array([-10, 10]), Scale.LOG, 10, None)


def test_calculate_histogram_int_linear_fixed():
    h = calculate_histogram(np.arange(1000), Scale.LINEAR, 9, None)
    assert h.bins == 9
    assert h.scale == Scale.LINEAR
    assert h.ticks == [0.0, 111.0, 222.0, 333.0, 444.0, 555.0, 666.0, 777.0, 888.0, 999.0]
    assert h.frequencies == [111, 111, 111, 111, 111, 111, 111, 111, 112]
    assert h.interquartile == (249.75, 749.25)


def test_calculate_histogram_int_log_fixed():
    h = calculate_histogram(np.arange(1, 1000), Scale.LOG, 10, None)
    assert h.bins == 10
    assert h.scale == Scale.LOG
    baseline_ticks = [1.0, 1.9950626988936724, 3.980275172516904, 7.9408985280210524,
                      15.842590448954471, 31.606961258558222, 63.05786943232691,
                      125.80440317614288, 250.98767213330342, 500.7361425553087,
                      999.0000000000003]
    for my_tick, baseline_tick in zip(h.ticks, baseline_ticks):
        assert my_tick == pytest.approx(baseline_tick)
    assert h.frequencies == [1, 2, 4, 8, 16, 32, 62, 125, 250, 499]
    assert h.interquartile == (250.5, 749.5)


def test_calculate_histogram_timedelta_ticks():
    h = calculate_histogram(
        np.array([timedelta(seconds=n) for n in range(1000)], dtype="timedelta64[s]"),
        None, None, [timedelta(seconds=240), timedelta(seconds=700)])
    assert h.bins == 3
    assert h.scale == Scale.LINEAR
    assert h.ticks == [timedelta(seconds=60), timedelta(seconds=240), timedelta(seconds=700),
                       timedelta(seconds=999)]
    assert h.frequencies == [240, 460, 300]
    assert h.interquartile == (timedelta(seconds=249), timedelta(seconds=749))
