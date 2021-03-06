from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.for_set import RepositoryGroupsMixin
from athenian.api.models.web.jira_filter import JIRAFilter


class ForSetDevelopers(Model, RepositoryGroupsMixin):
    """Filter for `/metrics/developers`."""

    openapi_types = {
        "repositories": List[str],
        "repogroups": Optional[List[List[int]]],
        "developers": List[str],
        "labels_include": List[str],
        "labels_exclude": List[str],
        "jira": JIRAFilter,
    }

    attribute_map = {
        "repositories": "repositories",
        "repogroups": "repogroups",
        "developers": "developers",
        "labels_include": "labels_include",
        "labels_exclude": "labels_exclude",
        "jira": "jira",
    }

    def __init__(
        self,
        repositories: Optional[List[str]] = None,
        repogroups: Optional[List[List[int]]] = None,
        developers: Optional[List[str]] = None,
        labels_include: Optional[List[str]] = None,
        labels_exclude: Optional[List[str]] = None,
        jira: Optional[JIRAFilter] = None,
    ):
        """ForSet - a model defined in OpenAPI

        :param repositories: The repositories of this ForSetDevelopers.
        :param repogroups: The repogroups of this ForSetDevelopers.
        :param developers: The developers of this ForSetDevelopers.
        :param labels_include: The labels_include of this ForSetDevelopers.
        :param labels_exclude: The labels_exclude of this ForSetDevelopers.
        :param jira: The jira of this ForSetDevelopers.
        """
        self._repositories = repositories
        self._repogroups = repogroups
        self._developers = developers
        self._labels_include = labels_include
        self._labels_exclude = labels_exclude
        self._jira = jira

    @property
    def repositories(self) -> List[str]:
        """Gets the repositories of this ForSetDevelopers.

        :return: The repositories of this ForSetDevelopers.
        """
        return self._repositories

    @repositories.setter
    def repositories(self, repositories: List[str]):
        """Sets the repositories of this ForSetDevelopers.

        :param repositories: The repositories of this ForSetDevelopers.
        """
        if repositories is None:
            raise ValueError("Invalid value for `repositories`, must not be `None`")

        self._repositories = repositories

    @property
    def developers(self) -> List[str]:
        """Gets the developers of this ForSetDevelopers.

        :return: The developers of this ForSetDevelopers.
        """
        return self._developers

    @developers.setter
    def developers(self, developers: List[str]):
        """Sets the developers of this ForSetDevelopers.

        :param developers: The developers of this ForSetDevelopers.
        """
        if developers is None:
            raise ValueError("Invalid value for `developers`, must not be `None`")

        self._developers = developers

    @property
    def labels_include(self) -> List[str]:
        """Gets the labels_include of this ForSetDevelopers.

        :return: The labels_include of this ForSetDevelopers.
        """
        return self._labels_include

    @labels_include.setter
    def labels_include(self, labels_include: List[str]):
        """Sets the labels_include of this ForSetDevelopers.

        :param labels_include: The labels_include of this ForSetDevelopers.
        """
        self._labels_include = labels_include

    @property
    def labels_exclude(self) -> List[str]:
        """Gets the labels_exclude of this ForSetDevelopers.

        :return: The labels_exclude of this ForSetDevelopers.
        """
        return self._labels_exclude

    @labels_exclude.setter
    def labels_exclude(self, labels_exclude: List[str]):
        """Sets the labels_exclude of this ForSetDevelopers.

        :param labels_exclude: The labels_exclude of this ForSetDevelopers.
        """
        self._labels_exclude = labels_exclude

    @property
    def jira(self) -> Optional[JIRAFilter]:
        """Gets the jira of this ForSetDevelopers.

        :return: The jira of this ForSetDevelopers.
        """
        return self._jira

    @jira.setter
    def jira(self, jira: Optional[JIRAFilter]):
        """Sets the jira of this ForSetDevelopers.

        :param jira: The jira of this ForSetDevelopers.
        """
        self._jira = jira
