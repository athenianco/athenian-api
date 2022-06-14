import sys

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest

from athenian.api.internal.features.entries import (
    MetricEntriesCalculator as OriginalMetricEntriesCalculator,
    make_calculator,
)
from athenian.api.internal.features.metric_calculator import group_by_repo


@pytest.fixture
def current_module():
    return sys.modules[__name__].__name__


@pytest.fixture
def base_testing_module(current_module):
    return current_module[: current_module.rfind(".")]


class MetricEntriesCalculator:
    """Fake calculator for different metrics."""

    def __init__(self, *args):
        """Create a `MetricEntriesCalculator`."""
        pass

    def is_ready_for(self, account, meta_ids) -> bool:
        """Check whether the calculator is ready for the given account and meta ids."""
        return account == 1


async def test_get_calculator(base_testing_module, mdb, pdb, rdb, cache):
    calc = make_calculator(1, (1,), mdb, pdb, rdb, cache)
    assert isinstance(calc, OriginalMetricEntriesCalculator)


def test_group_by_repo_single_repos():
    df = pd.DataFrame(
        {
            "repo": [
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
            ]
        }
    )
    groups = group_by_repo("repo", [["one"], ["two"], ["one", "two"]], df)
    assert len(groups) == 3
    assert_array_equal(groups[0], [0, 2, 3, 4])
    assert_array_equal(groups[1], [1, 5])
    assert_array_equal(groups[2], np.arange(len(df)))


def test_group_by_repo_few_groups():
    df = pd.DataFrame(
        {
            "repo": [
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
            ]
        }
    )
    groups = group_by_repo("repo", [["one"]], df)
    assert len(groups) == 1
    assert_array_equal(groups[0], [0, 2, 3, 4])
