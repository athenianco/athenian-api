from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class JIRAFilter(Model):
    """PR filters related to JIRA traits. The aggregation between each filter type is *AND*."""

    openapi_types = {
        "epics": List[str],
        "labels_include": List[str],
        "labels_exclude": List[str],
        "issue_types": List[str],
    }

    attribute_map = {
        "epics": "epics",
        "labels_include": "labels_include",
        "labels_exclude": "labels_exclude",
        "issue_types": "issue_types",
    }

    __slots__ = ["_" + k for k in openapi_types]

    def __init__(
        self,
        epics: Optional[List[str]] = None,
        labels_include: Optional[List[str]] = None,
        labels_exclude: Optional[List[str]] = None,
        issue_types: Optional[List[str]] = None,
    ):
        """JIRAFilter - a model defined in OpenAPI

        :param epics: The epics of this JIRAFilter.
        :param labels_include: The labels_include of this JIRAFilter.
        :param labels_exclude: The labels_exclude of this JIRAFilter.
        :param issue_types: The issue_types of this JIRAFilter.
        """
        self._epics = epics
        self._labels_include = labels_include
        self._labels_exclude = labels_exclude
        self._issue_types = issue_types

    @property
    def epics(self) -> List[str]:
        """Gets the epics of this JIRAFilter.

        PRs must be linked to at least one JIRA epic from the list.

        :return: The epics of this JIRAFilter.
        """
        return self._epics

    @epics.setter
    def epics(self, epics: List[str]):
        """Sets the epics of this JIRAFilter.

        PRs must be linked to at least one JIRA epic from the list.

        :param epics: The epics of this JIRAFilter.
        """
        self._epics = epics

    @property
    def labels_include(self) -> List[str]:
        """Gets the labels_include of this JIRAFilter.

        PRs must relate to at least one JIRA issue label from the list. Several labels may be
        concatenated by a comma `,` so that all of them are required.

        :return: The labels_include of this JIRAFilter.
        """
        return self._labels_include

    @labels_include.setter
    def labels_include(self, labels_include: List[str]):
        """Sets the labels_include of this JIRAFilter.

        PRs must relate to at least one JIRA issue label from the list. Several labels may be
        concatenated by a comma `,` so that all of them are required.

        :param labels_include: The labels_include of this JIRAFilter.
        """
        self._labels_include = labels_include

    @property
    def labels_exclude(self) -> List[str]:
        """Gets the labels_exclude of this JIRAFilter.

        PRs cannot relate to JIRA issue labels from the list.

        :return: The labels_exclude of this JIRAFilter.
        """
        return self._labels_exclude

    @labels_exclude.setter
    def labels_exclude(self, labels_exclude: List[str]):
        """Sets the labels_exclude of this JIRAFilter.

        PRs cannot relate to JIRA issue labels from the list.

        :param labels_exclude: The labels_exclude of this JIRAFilter.
        """
        self._labels_exclude = labels_exclude

    @property
    def issue_types(self) -> List[str]:
        """Gets the issue_types of this JIRAFilter.

        PRs must be linked to certain JIRA issue types, e.g. Bug, Task, Design Document, etc.

        :return: The issue_types of this JIRAFilter.
        """
        return self._issue_types

    @issue_types.setter
    def issue_types(self, issue_types: List[str]):
        """Sets the issue_types of this JIRAFilter.

        PRs must be linked to certain JIRA issue types, e.g. Bug, Task, Design Document, etc.

        :param issue_types: The issue_types of this JIRAFilter.
        """
        self._issue_types = issue_types
