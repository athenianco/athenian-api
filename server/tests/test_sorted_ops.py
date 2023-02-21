import numpy as np
from numpy.testing import assert_array_equal
import pytest

from athenian.api.sorted_ops import sorted_union1d


@pytest.mark.parametrize(
    "arr1, arr2, result",
    [
        ([1, 2], [2, 3], [1, 2, 3]),
        ([1, 2], [], [1, 2]),
        ([], [2, 3], [2, 3]),
        ([1, 2], [1, 2], [1, 2]),
        ([1, 3], [1, 2], [1, 2, 3]),
    ],
)
def test_sorted_union1d_general(arr1, arr2, result):
    assert_array_equal(
        sorted_union1d(np.asarray(arr1, dtype=int), np.asarray(arr2, dtype=int)),
        np.asarray(result, dtype=int),
    )
