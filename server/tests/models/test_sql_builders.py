import numpy as np
import pandas as pd
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
    with pytest.raises(ValueError):
        in_any_values_inline(np.array([], dtype=dtype))


def test_in_any_values_inline_null():
    assert in_any_values_inline([None, None]) == "VALUES (null)"


@pytest.mark.parametrize(
    "hack",
    [["normal", "oops')); drop table users;"], [b"normal", b"oops')); drop table users;"]],
)
@pytest.mark.parametrize("numpy", [False, True])
@pytest.mark.parametrize("method", [in_any_values_inline, in_inline])
def test_inline_injection(hack, numpy, method):
    values = hack if not numpy else np.array(hack)
    with pytest.raises(NotImplementedError):
        method(values)


@pytest.mark.parametrize(
    "values, dtype, result",
    [
        (["aaa", "bbb"], "S3", "'aaa','bbb'"),
        (["aaa", "b"], "S3", "'aaa','b'  "),
        (["aaa", "bbb"], "U3", "'aaa','bbb'"),
        (["aaa", "b"], "U3", "'aaa','b'  "),
        (["aaa", "ц"], "U3", "'aaa','ц'  "),
        (["aaa", "ц", "👍"], "U3", "'aaa','ц'  ,'👍'  "),
        (["aaa", "👍" * 1000], "U1000", "'aaa'" + " " * (1000 - 3) + ",'" + "👍" * 1000 + "'"),
        ([], "S1", "null"),
        ([], "U1", "null"),
        ([], int, "null"),
        ([1, 2], int, "1,2"),
        ([1, 203], int, "  1,203"),
    ],
)
def test_in_inline_array(values, dtype, result):
    assert in_inline(np.array(values, dtype=dtype)) == result


@pytest.mark.parametrize(
    "values, result",
    [
        (["aaa", "bbb"], "'aaa','bbb'"),
        (["aaa", "b"], "'aaa','b'"),
        (["aaa", "ц"], "'aaa','ц'"),
        ([b"aaa", b"bbb"], "'aaa','bbb'"),
        ([b"aaa", b"b"], "'aaa','b'"),
        ([None], "null"),
        ([], "null"),
        ([1, 2], "1,2"),
        ([1, 203], "  1,203"),
    ],
)
def test_in_inline_list(values, result):
    assert in_inline(values) == result


@pytest.mark.parametrize("method", [in_inline, in_any_values_inline])
@pytest.mark.parametrize(
    "container",
    [{"a", "b"}, {"a": 1, "b": 2}, pd.Index([1, 2, 3]), np.array(["a", "b"], dtype=object)],
)
def test_in_wrong_type(method, container):
    with pytest.raises(ValueError):
        method(container)
