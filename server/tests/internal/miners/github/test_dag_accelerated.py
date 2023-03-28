import numpy as np
from numpy.random import choice, randint, seed
from numpy.testing import assert_array_equal
import pytest

from athenian.api.internal.miners.github.dag_accelerated import sorted_set_difference


def test_sorted_set_difference_edges():
    arr1 = np.array([], dtype=np.uint32)
    arr2 = np.array([1], dtype=np.uint32)
    res = sorted_set_difference(arr1, arr2)
    assert len(res) == 0
    assert res.dtype == np.uint32

    arr2 = np.array([], dtype=np.uint32)
    res = sorted_set_difference(arr1, arr2)
    assert len(res) == 0
    assert res.dtype == np.uint32

    arr1 = np.array([1], dtype=np.uint32)
    res = sorted_set_difference(arr1, arr2)
    assert len(res) == 1
    assert res.dtype == np.uint32
    assert res[0] == 1


@pytest.mark.parametrize("size", [4, 8, 16, 32, 64, 128])
def test_sorted_set_difference_torture(size):
    seed(7)
    for i in range(1000):
        len1 = randint(1, size + 1)
        len2 = randint(1, size + 1)
        arr1 = np.sort(choice(np.arange(size, dtype=np.uint32), len1, replace=False))
        arr2 = np.sort(choice(np.arange(size, dtype=np.uint32), len2, replace=False))
        res = sorted_set_difference(arr1, arr2)
        truth = np.setdiff1d(arr1, arr2)
        assert_array_equal(res, truth, err_msg=f"{i}: {arr1} {arr2}")
