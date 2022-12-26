from typing import Optional, Type, TypeVar

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_filter import JIRAFilter

ForSetLike = TypeVar("ForSetLike", bound=Model)


class RepositoryGroupsMixin:
    """Mixin to add support for `repositories` and `repogroups`."""

    def validate_repositories(self, repositories: Optional[list[str]]) -> Optional[list[str]]:
        """Sets the repositories of this ForSetPullRequests.

        :param repositories: The repositories of this ForSetPullRequests.
        """
        self._validate_repogroups_and_repos(getattr(self, "_repogroups", None), repositories)
        return repositories

    def validate_repogroups(
        self,
        repogroups: Optional[list[list[int]]],
    ) -> Optional[list[list[int]]]:
        """Sets the repogroups of this ForSetPullRequests.

        :param repogroups: The repogroups of this ForSetPullRequests.
        """
        self._validate_repogroups_and_repos(repogroups, getattr(self, "_repositories", None))
        return repogroups

    def select_repogroup(self: ForSetLike, index: int) -> ForSetLike:
        """Change `repositories` to point at the specified group and clear `repogroups`."""
        fs = self.copy()
        if not self.repogroups:
            if index > 0:
                raise IndexError("%d is out of range (no repogroups)" % index)
            return fs
        if index >= len(self.repogroups):
            raise IndexError("%d is out of range (max is %d)" % (index, len(self.repogroups)))
        fs.repogroups = None
        fs.repositories = [self.repositories[i] for i in self.repogroups[index]]
        return fs

    @classmethod
    def _validate_repogroups_and_repos(
        self,
        repogroups: Optional[list[list[int]]],
        repositories: Optional[list[str]],
    ) -> None:
        if repositories is not None and len(repositories) == 0:
            raise ValueError("Invalid value for `repositories`, must not be an empty list")
        if repogroups is not None and len(repogroups) == 0:
            raise ValueError("`repogroups` must contain at least one list")

        if repogroups is None:
            return

        if repositories is None:
            raise ValueError("`repogroups` cannot be used with missing `repositories`")

        for i, group in enumerate(repogroups):
            if len(group) == 0:
                raise ValueError("`repogroups[%d]` must contain at least one element" % i)
            for j, v in enumerate(group):
                if v < 0:
                    raise ValueError(f"`repogroups[{i}][{j}]` = {v} must not be negative")
                if repositories is not None and v >= len(repositories):
                    raise ValueError(
                        f"`repogroups[{i}][{j}]` = {v} must be less than the number of "
                        f"repositories ({len(repositories)})",
                    )
            if len(set(group)) < len(group):
                raise ValueError("`repogroups[%d]` has duplicate items" % i)


def make_common_pull_request_filters(prefix_labels: str) -> Type[Model]:
    """Generate CommonPullRequestFilters class with the specified label properties name prefix."""

    class CommonPullRequestFilters(Model, sealed=False):
        """A few filters that are specific to filtering PR-related entities."""

        attribute_types = {
            prefix_labels + "labels_include": Optional[list[str]],
            prefix_labels + "labels_exclude": Optional[list[str]],
            "jira": Optional[JIRAFilter],
        }

    # we cannot do this at once because it crashes the ast module
    CommonPullRequestFilters.__init__.__doc__ = f"""
    Initialize a new instance of CommonPullRequestFilters.

    :param {prefix_labels}labels_include: The labels_include of this CommonPullRequestFilters.
    :param {prefix_labels}labels_exclude: The labels_exclude of this CommonPullRequestFilters.
    :param jira: The jira of this CommonPullRequestFilters.
    """

    return CommonPullRequestFilters


CommonPullRequestFilters = make_common_pull_request_filters("")


class ForSetLines(Model, RepositoryGroupsMixin, sealed=False):
    """Support for splitting metrics by the number of changed lines."""

    repositories: Optional[list[str]]
    repogroups: Optional[list[list[int]]]
    lines: Optional[list[int]]

    def validate_lines(self, lines: Optional[list[int]]) -> Optional[list[int]]:
        """Sets the lines of this ForSetPullRequests.

        :param lines: The lines of this ForSetPullRequests.
        """
        if lines is not None:
            if len(lines) < 2:
                raise ValueError("`lines` must contain at least 2 elements")
            if lines[0] < 0:
                raise ValueError("all elements of `lines` must be non-negative")
            for i, val in enumerate(lines[:-1]):
                if val >= lines[i + 1]:
                    raise ValueError("`lines` must monotonically increase")
        return lines

    def select_lines(self, index: int) -> "ForSetLines":
        """Change `lines` to point at the specified line range."""
        fs = self.copy()
        if self.lines is None:
            if index > 0:
                raise IndexError("%d is out of range (no lines)" % index)
            return fs
        if index >= len(self.lines) - 1:
            raise IndexError("%d is out of range (max is %d)" % (index, len(self.lines) - 1))
        fs.lines = [fs.lines[index], fs.lines[index + 1]]
        return fs
