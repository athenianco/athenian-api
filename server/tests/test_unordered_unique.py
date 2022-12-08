import numpy as np
from numpy.testing import assert_array_equal
import pytest

from athenian.api.unordered_unique import in1d_str, map_array_values


@pytest.mark.parametrize("dtype", ["S", "U"])
def test_in1d_str(dtype):
    arr1 = np.array(["vadim", "vadim", "markovtsev", "whatever", "vadim"], dtype=f"{dtype}10")
    arr2 = np.array(["vadim"], dtype=f"{dtype}5")
    mask = in1d_str(arr1, arr2)
    assert mask.tolist() == [True, True, False, False, True]
    mask = in1d_str(arr1, arr2[:0])
    assert mask.tolist() == [False, False, False, False, False]
    mask = in1d_str(arr1[:0], arr2)
    assert mask.tolist() == []


class TestMapArrayValues:
    def test_empty(self) -> None:
        mapped = map_array_values(
            np.array([]), np.array([1]), np.array(["a"], dtype="S"), "MISSING",
        )
        assert_array_equal(mapped, np.array([], dtype="S"))

    def test_only_matches(self) -> None:
        ar = np.array([4, 3, 2])
        keys = np.array([2, 3, 4, 5])
        values = np.array([20, 30, 40, 50])
        mapped = map_array_values(ar, keys, values, 0)
        assert_array_equal(mapped, np.array([40, 30, 20]))

    def test_only_misses(self) -> None:
        ar = np.array([4, 3])
        keys = np.array([2, 5])
        values = np.array([20, 50])
        mapped = map_array_values(ar, keys, values, -1)
        assert_array_equal(mapped, np.array([-1, -1]))

    def test_some_matches(self) -> None:
        ar = np.array([4, 7, 3, -1, 1, 0, -1])
        keys = np.array([0, 1, 3, 4, 5])
        values = np.array(["a", "b", "c", "d", "e", "f"], dtype="U10")
        mapped = map_array_values(ar, keys, values, "MISS")
        assert_array_equal(
            mapped, np.array(["d", "MISS", "c", "MISS", "b", "a", "MISS"], dtype="U10"),
        )
