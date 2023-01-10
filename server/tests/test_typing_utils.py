import dataclasses
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from athenian.api.typing_utils import (
    dataclass_asdict,
    df_from_structs,
    is_optional,
    is_union,
    numpy_struct,
)


class TestIsUnion:
    def test_typing_union(self) -> None:
        assert is_union(Union[int, str])

    def test_new_style_union(self) -> None:
        assert is_union(int | str)

    def test_negative(self) -> None:
        for obj in (1, None, int, type("_SimpleStruct", (), {})):
            assert not is_union(obj)


class TestIsOptional:
    @pytest.mark.xfail
    def test_optional_new_style_union(self) -> None:
        assert is_optional(Optional[str | int])
        assert is_optional(Optional[str | int | list])

    @pytest.mark.xfail
    def test_optional_union(self) -> None:
        assert is_optional(Optional[Union[str, int]])
        assert is_optional(Optional[Union[str, int, list]])


class TestDataclassAsDict:
    def test_base(self) -> None:
        @dataclass
        class A:
            a: str
            b: int
            c: list[int]

        a = A("a", 2, [1])

        a_dct = dataclass_asdict(a)
        assert a_dct == {"a": "a", "b": 2, "c": [1]}
        assert list(a_dct.keys()) == ["a", "b", "c"]

    def test_nested_subclass(self) -> None:
        @dataclass
        class A:
            a: str

        @dataclass
        class B:
            a: A

        b = B(A("a"))

        b_dct = dataclass_asdict(b)
        assert b_dct == {"a": b.a}
        assert b_dct["a"] is b.a


@numpy_struct
class _SimpleStruct:
    class Immutable:
        i: np.uint32


@dataclasses.dataclass(frozen=True, slots=True)
class _NestedDC:
    f: int


@numpy_struct
class _WithNestedDC:
    class Immutable:
        i: np.uint32

    class Optional:
        n: _NestedDC


@numpy_struct
class _ArrayStruct:
    class Immutable:
        i: [np.uint32]


@numpy_struct
class _NestedStruct:
    class Immutable:
        f: np.uint32
        f2: np.bool_


@numpy_struct
class _WithNestedStruct:
    class Immutable:
        i: np.uint32

    class Optional:
        n: _NestedStruct


class TestDFFromStructs:
    def test_base(self) -> None:
        a = _SimpleStruct.from_fields(i=3)
        df = df_from_structs([a])

        assert list(df.columns) == ["i"]
        assert list(df.i.values) == [3]

    def test_base_from_iterable(self) -> None:
        a = _SimpleStruct.from_fields(i=3)
        b = _SimpleStruct.from_fields(i=5)
        df = df_from_structs(iter([a, b]))

        assert list(df.columns) == ["i"]
        assert list(df.i.values) == [3, 5]

    def test_nested_dataclass(self) -> None:
        a = _WithNestedDC.from_fields(i=1)
        a.n = _NestedDC(2)
        a1 = _WithNestedDC.from_fields(i=10)
        a1.n = _NestedDC(20)
        df = df_from_structs([a, a1])

        assert list(df.i.values) == [1, 10]
        assert list(df.n_f.values) == [2, 20]

    def test_nested_dataclass_from_iterable(self) -> None:
        a = _WithNestedDC.from_fields(i=1)
        a.n = _NestedDC(2)
        a1 = _WithNestedDC.from_fields(i=10)
        a1.n = _NestedDC(20)
        df = df_from_structs(iter([a, a1]))
        assert sorted(df.columns) == ["i", "n_f"]
        assert list(df.i.values) == [1, 10]
        assert list(df.n_f.values) == [2, 20]


def test_numpy_struct_vectorize_scalar_field():
    structs = [_WithNestedDC.from_fields(i=i) for i in range(100)]
    assert_array_equal(
        _WithNestedDC.vectorize_field(structs, "i"),
        np.arange(100, dtype=_WithNestedDC.dtype["i"]),
    )


def test_numpy_struct_vectorize_array_field():
    structs = [_ArrayStruct.from_fields(i=np.full(i, i, np.uint32)) for i in range(100)]
    arr, offsets = _ArrayStruct.vectorize_field(structs, "i")
    assert offsets[0] == 0
    assert_array_equal(
        offsets[1:],
        np.cumsum(np.arange(100, dtype=int)),
    )
    assert_array_equal(
        arr,
        np.repeat(np.arange(100), np.arange(100)).astype(np.uint32),
    )
