from dataclasses import dataclass

from athenian.api.typing_utils import dataclass_asdict


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
