import sys

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest

from athenian.api.controllers.features.entries import \
    CalculatorNotReadyException, make_calculator, \
    MetricEntriesCalculator as OriginalMetricEntriesCalculator
from athenian.api.controllers.features.metric_calculator import group_by_repo


@pytest.fixture
def current_module():
    return sys.modules[__name__].__name__


@pytest.fixture
def base_testing_module(current_module):
    return current_module[: current_module.rfind(".")]


class MetricEntriesCalculator:
    """Fake calculator for different metrics."""

    def __init__(self, *args) -> "MetricEntriesCalculator":
        """Create a `MetricEntriesCalculator`."""
        pass

    def is_ready_for(self, account, meta_ids) -> bool:
        """Check whether the calculator is ready for the given account and meta ids."""
        return account == 1


def test_get_calculator_no_variation(base_testing_module, mdb, pdb, rdb, cache):
    calc = make_calculator(
        None, 1, (1, ), mdb, pdb, rdb, cache, base_module=base_testing_module,
    )
    assert isinstance(calc, OriginalMetricEntriesCalculator)


def test_get_calculator_missing_module_no_error(mdb, pdb, rdb, cache):
    calc = make_calculator(
        "test_entries", 1, (1, ), mdb, pdb, rdb, cache, base_module="missing_module",
    )
    assert isinstance(calc, OriginalMetricEntriesCalculator)


def test_get_calculator_missing_implementation_no_error(
    base_testing_module, mdb, pdb, rdb, cache,
):
    calc = make_calculator(
        "api", 1, (1, ), mdb, pdb, rdb, cache, base_module="athenian",
    )
    assert isinstance(calc, OriginalMetricEntriesCalculator)


def test_get_calculator_variation_found(
    base_testing_module, current_module, mdb, pdb, rdb, cache,
):
    calc = make_calculator(
        "test_entries", 1, (1, ), mdb, pdb, rdb, cache, base_module=base_testing_module,
    )
    assert isinstance(calc, MetricEntriesCalculator)

    with pytest.raises(CalculatorNotReadyException):
        make_calculator(
            "test_entries", 2, (1, ), mdb, pdb, rdb, cache, base_module=base_testing_module,
        )


def test_group_by_repo_single_repos():
    df = pd.DataFrame({"repo": [
        "one",
        "two",
        "one",
        "one",
        "one",
        "two",
    ]})
    groups = group_by_repo("repo", [["one"], ["two"], ["one", "two"]], df)
    assert len(groups) == 3
    assert_array_equal(groups[0], [0, 2, 3, 4])
    assert_array_equal(groups[1], [1, 5])
    assert_array_equal(groups[2], np.arange(len(df)))


def test_group_by_repo_few_groups():
    df = pd.DataFrame({"repo": [
        "one",
        "two",
        "one",
        "one",
        "one",
        "two",
    ]})
    groups = group_by_repo("repo", [["one"]], df)
    assert len(groups) == 1
    assert_array_equal(groups[0], [0, 2, 3, 4])
