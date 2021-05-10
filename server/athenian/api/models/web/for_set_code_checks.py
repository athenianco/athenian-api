from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.for_set import RepositoryGroupsMixin
from athenian.api.models.web.jira_filter import JIRAFilter


class ForSetCodeChecks(Model, RepositoryGroupsMixin):
    """Filter for `/metrics/code_checks`."""

    openapi_types = {
        "repositories": List[str],
        "repogroups": Optional[List[List[int]]],
        "commit_authors": Optional[List[str]],
        "commit_author_groups": Optional[List[List[str]]],
        "jira": Optional[JIRAFilter],
    }

    attribute_map = {
        "repositories": "repositories",
        "repogroups": "repogroups",
        "commit_authors": "commit_authors",
        "commit_author_groups": "commit_author_groups",
        "jira": "jira",
    }

    def __init__(
        self,
        repositories: Optional[List[str]] = None,
        repogroups: Optional[List[List[int]]] = None,
        commit_authors: Optional[List[str]] = None,
        commit_author_groups: Optional[List[List[str]]] = None,
        jira: Optional[JIRAFilter] = None,
    ):
        """ForSetCodeChecks - a model defined in OpenAPI

        :param repositories: The repositories of this ForSetCodeChecks.
        :param repogroups: The repogroups of this ForSetCodeChecks.
        :param commit_authors: The commit_authors of this ForSetCodeChecks.
        :param commit_author_groups: The commit_author_groups of this ForSetCodeChecks.
        :param jira: The jira of this ForSetCodeChecks.
        """
        self._repositories = repositories
        self._repogroups = repogroups
        self._commit_authors = commit_authors
        self._commit_author_groups = commit_author_groups
        self._jira = jira

    @property
    def commit_authors(self) -> Optional[List[str]]:
        """Gets the commit_authors of this ForSetCodeChecks.

        Check runs must be triggered by commits authored by these people.

        :return: The commit_authors of this ForSetCodeChecks.
        """
        return self._commit_authors

    @commit_authors.setter
    def commit_authors(self, commit_authors: Optional[List[str]]):
        """Sets the commit_authors of this ForSetCodeChecks.

        Check runs must be triggered by commits authored by these people.

        :param commit_authors: The commit_authors of this ForSetCodeChecks.
        """
        self._commit_authors = commit_authors

    @property
    def commit_author_groups(self) -> Optional[List[List[str]]]:
        """Gets the commit_author_groups of this ForSetCodeChecks.

        Check runs must be triggered by commits authored by these people. We aggregate by each
        group so that you can request metrics of several teams at once. We treat `commit_authors`
        as another group, if specified.

        :return: The commit_author_groups of this ForSetCodeChecks.
        """
        return self._commit_author_groups

    @commit_author_groups.setter
    def commit_author_groups(self, commit_author_groups: Optional[List[List[str]]]):
        """Sets the commit_author_groups of this ForSetCodeChecks.

        Check runs must be triggered by commits authored by these people. We aggregate by each
        group so that you can request metrics of several teams at once. We treat `commit_authors`
        as another group, if specified.

        :param commit_author_groups: The commit_author_groups of this ForSetCodeChecks.
        """
        self._commit_author_groups = commit_author_groups

    @property
    def jira(self) -> Optional[JIRAFilter]:
        """Gets the jira of this ForSetCodeChecks.

        :return: The jira of this ForSetCodeChecks.
        """
        return self._jira

    @jira.setter
    def jira(self, jira: Optional[JIRAFilter]):
        """Sets the jira of this ForSetCodeChecks.

        :param jira: The jira of this ForSetCodeChecks.
        """
        self._jira = jira

    def select_commit_authors_group(self, index: int) -> "ForSetCodeChecks":
        """Change `commit_authors` to point at the specified `commit_authors_group`."""
        fs = self.copy()
        if self.commit_author_groups is None:
            if index > 0:
                raise IndexError("%d is out of range (no commit_author_groups)" % index)
            return fs
        if index >= len(self.commit_author_groups):
            raise IndexError("%d is out of range (max is %d)" % (
                index, len(self.withcommit_author_groupsgroups) - 1))
        fs.commit_authors = self.commit_author_groups[index]
        fs.commit_author_groups = None
        return fs
