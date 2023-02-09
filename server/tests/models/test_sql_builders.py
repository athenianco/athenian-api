import numpy as np
import pytest

from athenian.api.models import in_any_values_inline
from athenian.api.models.sql_builders import in_inline


@pytest.mark.parametrize(
    "values, dtype, result",
    [
        (["aaa", "bbb"], "S3", "VALUES ('aaa'),('bbb')"),
        (["aaa", "bbb"], "U3", "VALUES ('aaa'),('bbb')"),
        (["a", "bbb"], "S3", "VALUES ('a'  ),('bbb')"),
        (["a", "bbb"], "U3", "VALUES ('a'  ),('bbb')"),
        ([1, 2], int, "VALUES (1),(2)"),
        ([1, 203], int, "VALUES (  1),(203)"),
    ],
)
def test_in_any_values_inline_array(values, dtype, result):
    assert in_any_values_inline(np.array(values, dtype=dtype)) == result


@pytest.mark.parametrize(
    "values, result",
    [
        (["aaa", "bbb"], "VALUES ('aaa'),('bbb')"),
        ([b"aaa", b"bbb"], "VALUES ('aaa'),('bbb')"),
        ([None, b"aaa", None, b"bbb"], "VALUES ('aaa'),('bbb')"),
        (["a", "bbb", None], "VALUES ('a'),('bbb')"),
        ([b"a", b"bbb"], "VALUES ('a'),('bbb')"),
        (["a", "вадим"], "VALUES ('a'),('вадим')"),
        (["a", "вадим", "👍"], "VALUES ('a'),('вадим'),('👍')"),
        ([1, 2], "VALUES (1),(2)"),
        ([1, None, 2], "VALUES (1),(2)"),
        ([1, 203], "VALUES (  1),(203)"),
    ],
)
def test_in_any_values_inline_list(values, result):
    assert in_any_values_inline(values) == result


@pytest.mark.parametrize(
    "dtype",
    [
        "S1",
        "U1",
        int,
    ],
)
def test_in_any_values_inline_empty(dtype):
    with pytest.raises(AssertionError):
        in_any_values_inline(np.array([], dtype=dtype))


def test_in_any_values_inline_null():
    assert in_any_values_inline([None, None]) == "VALUES (null)"


@pytest.mark.parametrize(
    "values, dtype, result",
    [
        (["aaa", "bbb"], "S3", "'aaa','bbb'"),
        (["aaa", "bbb"], "U3", "'aaa','bbb'"),
        ([], "S1", "null"),
        ([], "U1", "null"),
        ([], int, "null"),
        ([1, 2], int, "1,2"),
        ([1, 203], int, "  1,203"),
    ],
)
def test_in_inline(values, dtype, result):
    assert in_inline(np.array(values, dtype=dtype)) == result
