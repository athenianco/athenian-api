from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.for_set import RepositoryGroupsMixin
from athenian.api.models.web.jira_filter import JIRAFilter


class ForSetCodeChecks(Model, RepositoryGroupsMixin):
    """Filter for `/metrics/code_checks`."""

    openapi_types = {
        "repositories": List[str],
        "repogroups": Optional[List[List[int]]],
        "pushers": Optional[List[str]],
        "pusher_groups": Optional[List[List[str]]],
        "jira": Optional[JIRAFilter],
    }

    attribute_map = {
        "repositories": "repositories",
        "repogroups": "repogroups",
        "pushers": "pushers",
        "pusher_groups": "pusher_groups",
        "jira": "jira",
    }

    def __init__(
        self,
        repositories: Optional[List[str]] = None,
        repogroups: Optional[List[List[int]]] = None,
        pushers: Optional[List[str]] = None,
        pusher_groups: Optional[List[List[str]]] = None,
        jira: Optional[JIRAFilter] = None,
    ):
        """ForSetCodeChecks - a model defined in OpenAPI

        :param repositories: The repositories of this ForSetCodeChecks.
        :param repogroups: The repogroups of this ForSetCodeChecks.
        :param pushers: The pushers of this ForSetCodeChecks.
        :param pusher_groups: The pusher_groups of this ForSetCodeChecks.
        :param jira: The jira of this ForSetCodeChecks.
        """
        self._repositories = repositories
        self._repogroups = repogroups
        self._pushers = pushers
        self._pusher_groups = pusher_groups
        self._jira = jira

    @property
    def pushers(self) -> Optional[List[str]]:
        """Gets the pushers of this ForSetCodeChecks.

        Check runs must be triggered by commits pushed by these people.

        :return: The pushers of this ForSetCodeChecks.
        """
        return self._pushers

    @pushers.setter
    def pushers(self, pushers: Optional[List[str]]):
        """Sets the pushers of this ForSetCodeChecks.

        Check runs must be triggered by commits pushed by these people.

        :param pushers: The pushers of this ForSetCodeChecks.
        """
        self._pushers = pushers

    @property
    def pusher_groups(self) -> Optional[List[List[str]]]:
        """Gets the pusher_groups of this ForSetCodeChecks.

        Check runs must be triggered by commits authored by these people. We aggregate by each
        group so that you can request metrics of several teams at once. We treat `pushers`
        as another group, if specified.

        :return: The pusher_groups of this ForSetCodeChecks.
        """
        return self._pusher_groups

    @pusher_groups.setter
    def pusher_groups(self, pusher_groups: Optional[List[List[str]]]):
        """Sets the pusher_groups of this ForSetCodeChecks.

        Check runs must be triggered by commits authored by these people. We aggregate by each
        group so that you can request metrics of several teams at once. We treat `pushers`
        as another group, if specified.

        :param pusher_groups: The pusher_groups of this ForSetCodeChecks.
        """
        self._pusher_groups = pusher_groups

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

    def select_pushers_group(self, index: int) -> "ForSetCodeChecks":
        """Change `pushers` to point at the specified `pushers_group`."""
        fs = self.copy()
        if self.pusher_groups is None:
            if index > 0:
                raise IndexError("%d is out of range (no pusher_groups)" % index)
            return fs
        if index >= len(self.pusher_groups):
            raise IndexError("%d is out of range (max is %d)" % (
                index, len(self.withpusher_groupsgroups) - 1))
        fs.pushers = self.pusher_groups[index]
        fs.pusher_groups = None
        return fs
