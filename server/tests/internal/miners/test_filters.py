from typing import Any

import pytest

from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.models.web import JIRAFilter as WebJIRAFilter


class TestJIRAFilter:
    def test_compatible_with_priorities(self) -> None:
        f0 = _mk_jira_filter(1, priorities=frozenset(["high", "medium"]))

        for priorities in ((), ("low", "high")):
            f1 = _mk_jira_filter(1, priorities=frozenset(priorities))
            assert not f0.compatible_with(f1)

        for priorities in (("high",), ("medium", "high")):
            f1 = _mk_jira_filter(1, priorities=frozenset(priorities))
            assert f0.compatible_with(f1)

    def test_compatible_with_epics(self) -> None:
        f0 = _mk_jira_filter(0)
        for epics in (frozenset(), frozenset(["E0", "E1", "E2"])):
            f1 = _mk_jira_filter(
                1, labels=LabelFilter({"enh."}, ()), epics=epics, priorities=frozenset(["task"]),
            )
            assert f0.compatible_with(f1)

    def test_str_varies_with_projects(self) -> None:
        f0 = _mk_jira_filter(1, projects=frozenset(["foo"]), custom_projects=True)
        f1 = _mk_jira_filter(1, projects=frozenset(["bar"]), custom_projects=True)
        assert str(f0) != str(f1)

    def test_str_no_custom_projects(self) -> None:
        f0 = _mk_jira_filter(1, projects=frozenset(["foo"]), custom_projects=False)
        f1 = _mk_jira_filter(1, projects=frozenset(["foo", "bar"]), custom_projects=False)
        assert str(f0) == str(f1)

    def test_str_unmapped(self) -> None:
        f0 = _mk_jira_filter(1, epics=frozenset(["a"]), unmapped=True)
        f1 = _mk_jira_filter(1, epics=frozenset(["b"]), unmapped=True)
        assert str(f0) == str(f1)

    def test_epics_bool_value_validation(self) -> None:
        with pytest.raises(ValueError):
            _mk_jira_filter(1, epics=True)
        _mk_jira_filter(1, epics=False)


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
            1,
            projects=frozenset(["proj1"]),
            issue_types=frozenset(["bug"]),
            priorities=frozenset(["low", "medium"]),
        )
        expected = _mk_jira_filter(
            1,
            projects=frozenset(["proj1", "proj0"]),
            priorities=frozenset(["low", "high", "medium"]),
        )
        assert f0 | f1 == expected

    def test_zero_account(self) -> None:
        f0 = _mk_jira_filter(1, projects=frozenset(["pr0"]), custom_projects=True)
        f1 = _mk_jira_filter(0)
        expected = _mk_jira_filter(0)
        assert f0 | f1 == expected


class TestJIRAFilterCombine:
    def test_two_filters(self) -> None:
        f0 = _mk_jira_filter(1, projects=frozenset(["proj0"]))
        f1 = _mk_jira_filter(1, projects=frozenset(["proj0", "proj1"]))
        expected = _mk_jira_filter(1, projects=frozenset(["proj0", "proj1"]))
        assert JIRAFilter.combine(f0, f1) == expected

    def test_more_filters(self) -> None:
        f0 = _mk_jira_filter(1, projects=frozenset(["proj0"]))
        f1 = _mk_jira_filter(1, projects=frozenset(["proj1"]))
        f2 = _mk_jira_filter(0)
        expected = _mk_jira_filter(0)
        assert JIRAFilter.combine(f0, f1, f2) == expected

    def test_two_filters_custom_projects(self) -> None:
        f0 = JIRAFilter(account=1, projects=frozenset(["p0", "p1"]), custom_projects=False)
        f1 = JIRAFilter(account=1, projects=frozenset(["p0"]), custom_projects=True)
        expected = JIRAFilter(account=1, projects=frozenset(["p0", "p1"]), custom_projects=False)
        assert JIRAFilter.combine(f0, f1) == expected

    def test_one_filter(self) -> None:
        f = JIRAFilter(account=1, projects=frozenset(), custom_projects=False)
        f1 = JIRAFilter.combine(f)
        assert f == f1

    def test_status_categories(self) -> None:
        f0 = JIRAFilter(account=1)
        f1 = JIRAFilter(account=1, status_categories=frozenset(["Foo"]))
        f2 = JIRAFilter(account=1, status_categories=frozenset(["Bar", "Ba Z"]))

        assert JIRAFilter.combine(f0, f1) == f0
        assert JIRAFilter.combine(f1, f2) == JIRAFilter(
            account=1, status_categories=frozenset(["Foo", "Bar", "Ba Z"]),
        )


class TestJIRAFilterFromWeb:
    def test_empty_web_filter(self) -> None:
        web_filter = WebJIRAFilter()
        f = JIRAFilter.from_web(web_filter, 1)
        assert not f
        assert f == JIRAFilter.empty()


def _mk_jira_filter(account: int, **kwargs: Any) -> JIRAFilter:
    kwargs.setdefault("projects", frozenset(["1"] if account else []))
    kwargs.setdefault("custom_projects", False)
    return JIRAFilter(account, **kwargs)
