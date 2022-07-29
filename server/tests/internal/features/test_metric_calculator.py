import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

from athenian.api.internal.features.metric_calculator import group_by_repo


def test_group_by_repo_single_repos():
    df = pd.DataFrame({"repo": ["one", "two", "one", "one", "one", "two"]})
    groups = group_by_repo("repo", [["one"], ["two"], ["one", "two"]], df)
    assert len(groups) == 3
    assert_array_equal(groups[0], [0, 2, 3, 4])
    assert_array_equal(groups[1], [1, 5])
    assert_array_equal(groups[2], np.arange(len(df)))


def test_group_by_repo_few_groups():
    df = pd.DataFrame({"repo": ["one", "two", "one", "one", "one", "two"]})
    groups = group_by_repo("repo", [["one"]], df)
    assert len(groups) == 1
    assert_array_equal(groups[0], [0, 2, 3, 4])
