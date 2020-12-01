from typing import Iterable, List, Optional, Set, Tuple

from athenian.api.models.web.jira_filter import JIRAFilter as WebJIRAFilter
from athenian.api.typing_utils import dataclass


@dataclass(slots=True, frozen=True)
class LabelFilter:
    """Pull Request labels: must/must not contain."""

    include: Set[str]
    exclude: Set[str]

    @classmethod
    def from_iterables(cls,
                       include: Optional[Iterable[str]],
                       exclude: Optional[Iterable[str]],
                       ) -> "LabelFilter":
        """Initialize a new instance of LabelFilter from two iterables."""
        return cls(include=set(s.lower() for s in (include or [])),
                   exclude=set(s.lower() for s in (exclude or [])))

    @classmethod
    def empty(cls) -> "LabelFilter":
        """Initialize an empty LabelFilter."""
        return cls(set(), set())

    def __bool__(self) -> bool:
        """Return value indicating whether there is at least one included or excluded label."""
        return bool(self.include) or bool(self.exclude)

    def __str__(self) -> str:
        """Implement str()."""
        return "[%s, %s]" % (sorted(self.include), sorted(self.exclude))

    def __repr__(self) -> str:
        """Implement repr()."""
        return "LabelFilter(%r, %r)" % (self.include, self.exclude)

    def compatible_with(self, other: "LabelFilter") -> bool:
        """Check whether the `other` filter can be applied to the items filtered by `self`."""
        return (
            ((not self.include) or (other.include and self.include.issuperset(other.include)))
            and
            ((not self.exclude) or (other.exclude and self.exclude.issubset(other.exclude)))
        )

    @classmethod
    def split(cls, labels: Set[str]) -> Tuple[List[str], List[List[str]]]:
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


@dataclass(frozen=True)
class JIRAFilter:
    """JIRA traits to select assigned PRs."""

    account: int
    projects: List[str]
    labels: LabelFilter
    epics: Set[str]
    issue_types: Set[str]
    unmapped: bool

    @classmethod
    def empty(cls) -> "JIRAFilter":
        """Initialize an empty JIRAFilter."""
        return cls(0, [], LabelFilter.empty(), set(), set(), False)

    def __bool__(self) -> bool:
        """Return value indicating whether this filter is not an identity."""
        return self.account > 0 and (
            bool(self.labels) or bool(self.epics) or bool(self.issue_types) or self.unmapped)

    def __str__(self) -> str:
        """Implement str()."""
        if not self.unmapped:
            return "[%s, %s, %s]" % (self.labels, sorted(self.epics), sorted(self.issue_types))
        return "<unmapped>"

    def __repr__(self) -> str:
        """Implement repr()."""
        return "JIRAFilter(%r, %r, %r, %r, %r)" % (
            self.account, self.labels, self.epics, self.issue_types, self.unmapped)

    def compatible_with(self, other: "JIRAFilter") -> bool:
        """Check whether the `other` filter can be applied to the items filtered by `self`."""
        if self.unmapped != other.unmapped:
            return False
        elif self.unmapped:
            return True
        if not self.labels.compatible_with(other.labels):
            return False
        if self.epics and (not other.epics or
                           not self.epics.issuperset(other.epics)):
            return False
        if self.issue_types and (not other.issue_types or
                                 not self.issue_types.issuperset(other.issue_types)):
            return False
        return True

    @classmethod
    def from_web(cls, model: Optional[WebJIRAFilter], ids: Tuple[int, List[str]]) -> "JIRAFilter":
        """Initialize a new JIRAFilter from the corresponding web model."""
        if model is None:
            return cls.empty()
        labels = LabelFilter.from_iterables(model.labels_include, model.labels_exclude)
        return JIRAFilter(account=ids[0],
                          projects=ids[1],
                          labels=labels,
                          epics={s.upper() for s in (model.epics or [])},
                          issue_types={s.lower() for s in (model.issue_types or [])},
                          unmapped=bool(model.unmapped))
