from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Optional

from athenian.api.internal.jira import JIRAConfig, normalize_issue_type
from athenian.api.models.web.jira_filter import JIRAFilter as WebJIRAFilter


@dataclasses.dataclass(slots=True, frozen=True)
class LabelFilter:
    """Pull Request labels: must/must not contain."""

    include: set[str]
    exclude: set[str]

    @classmethod
    def from_iterables(
        cls,
        include: Optional[Iterable[str]],
        exclude: Optional[Iterable[str]],
    ) -> "LabelFilter":
        """Initialize a new instance of LabelFilter from two iterables."""
        return cls(
            include={s.lower().strip(" \t,") for s in (include or [])},
            exclude={s.lower() for s in (exclude or [])},
        )

    @classmethod
    def empty(cls) -> LabelFilter:
        """Initialize an empty LabelFilter."""
        return cls(set(), set())

    def __bool__(self) -> bool:
        """Return value indicating whether there is at least one included or excluded label."""
        return bool(self.include) or bool(self.exclude)

    def __or__(self, other: LabelFilter) -> LabelFilter:
        """Return a new LabelFilter which is logical union of this filter with another."""
        return LabelFilter(
            _join_filter_sets(self.include, other.include),
            set() if (not self.exclude or not other.exclude) else self.exclude & other.exclude,
        )

    def __str__(self) -> str:
        """Implement str()."""
        return "[%s, %s]" % (sorted(self.include), sorted(self.exclude))

    def __repr__(self) -> str:
        """Implement repr()."""
        return "LabelFilter(%r, %r)" % (self.include, self.exclude)

    def compatible_with(self, other: LabelFilter) -> bool:
        """Check whether the `other` filter can be applied to the items filtered by `self`."""
        return (
            (not self.include) or (bool(other.include) and self.include.issuperset(other.include))
        ) and (
            (not self.exclude) or (bool(other.exclude) and self.exclude.issubset(other.exclude))
        )

    def match(self, labels: Iterable[str]) -> bool:
        """Check whether a set of labels satisfies the filter."""
        assert not isinstance(labels, str)
        labels = set(labels)
        if self.include:
            if len(labels) == 1:
                if not self.include.intersection(labels):
                    return False
            else:
                for iset in self.include:
                    if labels.issuperset({p.strip() for p in iset.split(",")}):
                        break
                else:
                    return False
        return not (self.exclude and self.exclude.intersection(labels))

    @classmethod
    def split(cls, labels: set[str]) -> tuple[list[str], list[list[str]]]:
        """Split labels by comma "," and divide into two groups: singles and multiples."""
        singles = []
        multiples = []
        for label in labels:
            parts = label.split(",")
            if len(parts) == 1:
                singles.append(parts[0])
                continue
            multiples.append([p.strip() for p in parts])
        return singles, multiples


@dataclasses.dataclass(slots=True, frozen=True)
class JIRAFilter:
    """JIRA traits to select assigned PRs."""

    account: int
    projects: list[str]
    labels: LabelFilter
    epics: set[str] | bool
    issue_types: set[str]
    priorities: set[str]
    custom_projects: bool  # PRs must be mapped to any issue in `projects`
    unmapped: bool  # select everything but the mapped PRs

    def __post_init(self) -> None:
        if self.epics is True:
            raise ValueError("epics must be a set of strings or `False`")

    @classmethod
    def empty(cls) -> JIRAFilter:
        """Initialize an empty JIRAFilter."""
        return cls(0, [], LabelFilter.empty(), set(), set(), set(), False, False)

    def __bool__(self) -> bool:
        """Return value indicating whether this filter is not an identity."""
        return self.account > 0 and any(
            [
                self.labels,
                self.epics,
                self.issue_types,
                self.priorities,
                self.unmapped,
                self.custom_projects,
            ],
        )

    def __or__(self, other: JIRAFilter) -> JIRAFilter:
        """Return a new JIRAFilter which is logical union of this filter with another."""
        if (
            ((self.account != other.account) and self.account and other.account)
            or self.unmapped != other.unmapped
            or isinstance(self.epics, bool) != isinstance(other.epics, bool)
        ):
            raise ValueError("Cannot union JIRAFilter with different accounts, unmapped or epics")

        if not self.account or not other.account:
            return self.empty()

        if isinstance(self.epics, bool):
            epics: bool | set[str] = False
        else:
            epics = _join_filter_sets(self.epics, other.epics)

        projects = list(_join_filter_sets(set(self.projects), set(other.projects)))

        return JIRAFilter(
            self.account,
            projects,
            self.labels | other.labels,
            epics,
            _join_filter_sets(self.issue_types, other.issue_types),
            _join_filter_sets(self.priorities, other.priorities),
            bool(projects),
            self.unmapped,
        )

    def __str__(self) -> str:
        """Implement str()."""
        if self.unmapped:
            return "[%s, unmapped=True]" % self.account

        projects = sorted(self.projects) if self.custom_projects else ["<all>"]
        return "[%s, %s, %s, %s, %s, projects=%s]" % (
            self.account,
            self.labels,
            self.epics if isinstance(self.epics, bool) else sorted(self.epics),
            sorted(self.issue_types),
            sorted(self.priorities),
            projects,
        )

    def __repr__(self) -> str:
        """Implement repr()."""
        return "JIRAFilter(%r, %r, %r, %r, %r, %r, %r)" % (
            self.account,
            self.labels,
            self.epics,
            self.issue_types,
            self.priorities,
            self.custom_projects,
            self.unmapped,
        )

    def __sentry_repr__(self) -> str:
        """Override {}.__repr__() in Sentry."""
        return repr(self)

    def compatible_with(self, other: JIRAFilter) -> bool:
        """Check whether the `other` filter can be applied to the items filtered by `self`."""
        if self.unmapped != other.unmapped:
            return False
        elif self.unmapped:
            return True
        if not self.labels.compatible_with(other.labels):
            return False
        if (self.epics is False) != (other.epics is False):
            return False
        if not isinstance(self.epics, bool):
            assert not isinstance(other.epics, bool)
            if self.epics and (not other.epics or not self.epics.issuperset(other.epics)):
                return False
        if self.issue_types and (  # noqa: PIE801
            not other.issue_types or not self.issue_types.issuperset(other.issue_types)
        ):
            return False
        if self.priorities and (  # noqa: PIE801
            not other.priorities or not self.priorities.issuperset(other.priorities)
        ):
            return False
        return True

    @classmethod
    def from_web(cls, model: Optional[WebJIRAFilter], ids: Optional[JIRAConfig]) -> JIRAFilter:
        """Initialize a new JIRAFilter from the corresponding web model."""
        if model is None or ids is None:
            return cls.empty()
        labels = LabelFilter.from_iterables(model.labels_include, model.labels_exclude)
        if not (custom_projects := bool(model.projects)):
            projects = sorted(ids.projects)
        else:
            reverse_map = {v: k for k, v in ids[1].items()}
            projects = sorted(reverse_map[k] for k in model.projects if k in reverse_map)
        return JIRAFilter(
            account=ids.acc_id,
            projects=projects,
            labels=labels,
            epics={s.upper() for s in (model.epics or [])},
            issue_types={normalize_issue_type(s) for s in (model.issue_types or [])},
            priorities=set(),  # not present in web model
            custom_projects=custom_projects,
            unmapped=bool(model.unmapped),
        )

    @classmethod
    def from_jira_config(cls, jira_config: JIRAConfig) -> JIRAFilter:
        """Initialize a new JIRAFilter selecting everything belonging to the jira account."""
        return cls.empty().replace(
            account=jira_config.acc_id,
            projects=list(jira_config.projects),
            custom_projects=True,
        )

    def replace(self, **kwargs: Any) -> JIRAFilter:
        """Return a new  JIRAFilter with some fields replaced."""
        return dataclasses.replace(self, **kwargs)


def _join_filter_sets(set_a: set, set_b) -> set:
    if (not set_a) or (not set_b):
        return set()
    return set_a | set_b
