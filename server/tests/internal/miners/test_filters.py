from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter


class TestJIRAFilter:
    def test_compatible_with_priorities(self) -> None:
        f0 = JIRAFilter(1, [], LabelFilter.empty(), set(), set(), {"high", "medium"}, False, False)

        for priorities in (set(), {"low", "high"}):
            f1 = JIRAFilter(1, [], LabelFilter.empty(), set(), set(), priorities, False, False)
            assert not f0.compatible_with(f1)

        for priorities in ({"high"}, {"medium", "high"}):
            f1 = JIRAFilter(1, [], LabelFilter.empty(), set(), set(), priorities, False, False)
            assert f0.compatible_with(f1)

    def test_compatible_with_epics(self) -> None:
        f0 = JIRAFilter(0, [], LabelFilter(set(), set()), set(), set(), set(), False, False)
        for epics in (frozenset(), frozenset(["E0", "E1", "E2"])):
            f1 = JIRAFilter(
                1, [], LabelFilter({"enh."}, set()), epics, {"task"}, set(), False, False,
            )
            assert f0.compatible_with(f1)
