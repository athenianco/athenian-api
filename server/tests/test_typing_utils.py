from dataclasses import dataclass
from typing import Union

from athenian.api.typing_utils import dataclass_asdict, is_union


class TestIsUnion:
    def test_typing_union(self) -> None:
        assert is_union(Union[int, str])

    def test_new_style_union(self) -> None:
        assert is_union(int | str)

    def test_negative(self) -> None:
        for obj in (1, None, int, type("A", (), {})):
            assert not is_union(obj)


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
