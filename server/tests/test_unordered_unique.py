import numpy as np
import pytest

from athenian.api.unordered_unique import in1d_str


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
