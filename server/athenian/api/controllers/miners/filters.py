from dataclasses import dataclass
from typing import Iterable, Optional, Set


@dataclass(frozen=True)
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
        return cls(include=set(include or []), exclude=set(exclude or []))

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


@dataclass(frozen=True)
class JIRAFilter:
    """JIRA traits to select assigned PRs."""

    labels: LabelFilter
    epics: Set[str]
    issue_types: Set[str]
