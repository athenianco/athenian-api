from typing import Any

import pytest

from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter


class TestJIRAFilter:
    def test_compatible_with_priorities(self) -> None:
        f0 = JIRAFilter(1, (), LabelFilter.empty(), set(), set(), {"high", "medium"}, False, False)

        for priorities in (set(), {"low", "high"}):
            f1 = JIRAFilter(1, (), LabelFilter.empty(), set(), set(), priorities, False, False)
            assert not f0.compatible_with(f1)

        for priorities in ({"high"}, {"medium", "high"}):
            f1 = JIRAFilter(1, (), LabelFilter.empty(), set(), set(), priorities, False, False)
            assert f0.compatible_with(f1)

    def test_compatible_with_epics(self) -> None:
        f0 = JIRAFilter(0, [], LabelFilter(set(), set()), set(), set(), set(), False, False)
        for epics in (frozenset(), frozenset(["E0", "E1", "E2"])):
            f1 = JIRAFilter(
                1, [], LabelFilter({"enh."}, set()), epics, {"task"}, set(), False, False,
            )
            assert f0.compatible_with(f1)

    def test_str_varies_with_projects(self) -> None:
        f0 = JIRAFilter(1, ["foo"], LabelFilter.empty(), set(), set(), set(), True, False)
        f1 = JIRAFilter(1, ["bar"], LabelFilter.empty(), set(), set(), set(), True, False)
        assert str(f0) != str(f1)

    def test_str_no_custom_projects(self) -> None:
        f0 = JIRAFilter(1, ["foo"], LabelFilter.empty(), set(), set(), set(), False, False)
        f1 = JIRAFilter(1, ["bar", "foo"], LabelFilter.empty(), set(), set(), set(), False, False)
        assert str(f0) == str(f1)

    def test_str_unmapped(self) -> None:
        f0 = JIRAFilter(1, [], LabelFilter.empty(), {"a"}, set(), set(), False, True)
        f1 = JIRAFilter(1, [], LabelFilter.empty(), {"b"}, set(), set(), False, True)
        assert str(f0) == str(f1)


class TestJIRAFilterUnion:
    def test_different_account(self) -> None:
        f0 = _mk_jira_filter(1)
        f1 = _mk_jira_filter(2)
        with pytest.raises(ValueError):
            f0 | f1

    def test_base(self) -> None:
        f0 = _mk_jira_filter(
            1,
            projects=frozenset(["proj0"]),
            epics=frozenset(["epic0"]),
            priorities=frozenset(["high", "medium"]),
        )
        f1 = _mk_jira_filter(
            1, issue_types=frozenset(["bug"]), priorities=frozenset(["low", "medium"]),
        )
        expected = _mk_jira_filter(1, priorities=frozenset(["low", "high", "medium"]))
        assert f0 | f1 == expected

    def test_zero_account(self) -> None:
        f0 = _mk_jira_filter(1, projects=frozenset(["pr0"]), custom_projects=True)
        f1 = _mk_jira_filter(0)
        expected = _mk_jira_filter(0)
        assert f0 | f1 == expected


def _mk_jira_filter(*args: Any, **kwargs: Any) -> JIRAFilter:
    for field in ("projects", "epics", "issue_types", "priorities"):
        kwargs.setdefault(field, frozenset())
    kwargs.setdefault("labels", LabelFilter.empty())
    kwargs.setdefault("custom_projects", False)
    kwargs.setdefault("unmapped", False)

    return JIRAFilter(*args, **kwargs)
