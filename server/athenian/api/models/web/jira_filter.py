from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class JIRAFilter(Model):
    """PR filters related to JIRA traits. The aggregation between each filter type is *AND*."""

    openapi_types = {
        "epics": Optional[List[str]],
        "labels_include": Optional[List[str]],
        "labels_exclude": Optional[List[str]],
        "issue_types": Optional[List[str]],
        "projects": Optional[List[str]],
        "unmapped": Optional[bool],
    }

    attribute_map = {
        "epics": "epics",
        "labels_include": "labels_include",
        "labels_exclude": "labels_exclude",
        "issue_types": "issue_types",
        "projects": "projects",
        "unmapped": "unmapped",
    }

    def __init__(
        self,
        epics: Optional[List[str]] = None,
        labels_include: Optional[List[str]] = None,
        labels_exclude: Optional[List[str]] = None,
        issue_types: Optional[List[str]] = None,
        projects: Optional[List[str]] = None,
        unmapped: Optional[bool] = None,
    ):
        """JIRAFilter - a model defined in OpenAPI

        :param epics: The epics of this JIRAFilter.
        :param labels_include: The labels_include of this JIRAFilter.
        :param labels_exclude: The labels_exclude of this JIRAFilter.
        :param issue_types: The issue_types of this JIRAFilter.
        :param projects: The projects of this JIRAFilter.
        :param unmapped: The unmapped of this JIRAFilter.
        """
        self._epics = epics
        self._labels_include = labels_include
        self._labels_exclude = labels_exclude
        self._issue_types = issue_types
        self._projects = projects
        self._unmapped = unmapped

    @property
    def epics(self) -> Optional[List[str]]:
        """Gets the epics of this JIRAFilter.

        PRs must be linked to at least one JIRA epic from the list.

        :return: The epics of this JIRAFilter.
        """
        return self._epics

    @epics.setter
    def epics(self, epics: Optional[List[str]]):
        """Sets the epics of this JIRAFilter.

        PRs must be linked to at least one JIRA epic from the list.

        :param epics: The epics of this JIRAFilter.
        """
        if epics and self._unmapped:
            raise ValueError("`unmapped` may not be mixed with anything else")
        self._epics = epics

    @property
    def labels_include(self) -> Optional[List[str]]:
        """Gets the labels_include of this JIRAFilter.

        PRs must relate to at least one JIRA issue label from the list. Several labels may be
        concatenated by a comma `,` so that all of them are required.

        :return: The labels_include of this JIRAFilter.
        """
        return self._labels_include

    @labels_include.setter
    def labels_include(self, labels_include: Optional[List[str]]):
        """Sets the labels_include of this JIRAFilter.

        PRs must relate to at least one JIRA issue label from the list. Several labels may be
        concatenated by a comma `,` so that all of them are required.

        :param labels_include: The labels_include of this JIRAFilter.
        """
        if labels_include and self._unmapped:
            raise ValueError("`unmapped` may not be mixed with anything else")
        self._labels_include = labels_include

    @property
    def labels_exclude(self) -> Optional[List[str]]:
        """Gets the labels_exclude of this JIRAFilter.

        PRs cannot relate to JIRA issue labels from the list.

        :return: The labels_exclude of this JIRAFilter.
        """
        return self._labels_exclude

    @labels_exclude.setter
    def labels_exclude(self, labels_exclude: Optional[List[str]]):
        """Sets the labels_exclude of this JIRAFilter.

        PRs cannot relate to JIRA issue labels from the list.

        :param labels_exclude: The labels_exclude of this JIRAFilter.
        """
        if labels_exclude and self._unmapped:
            raise ValueError("`unmapped` may not be mixed with anything else")
        self._labels_exclude = labels_exclude

    @property
    def issue_types(self) -> Optional[List[str]]:
        """Gets the issue_types of this JIRAFilter.

        PRs must be linked to certain JIRA issue types, e.g. Bug, Task, Design Document, etc.

        :return: The issue_types of this JIRAFilter.
        """
        return self._issue_types

    @issue_types.setter
    def issue_types(self, issue_types: Optional[List[str]]):
        """Sets the issue_types of this JIRAFilter.

        PRs must be linked to certain JIRA issue types, e.g. Bug, Task, Design Document, etc.

        :param issue_types: The issue_types of this JIRAFilter.
        """
        if issue_types and self._unmapped:
            raise ValueError("`unmapped` may not be mixed with anything else")
        self._issue_types = issue_types

    @property
    def projects(self) -> Optional[List[str]]:
        """Gets the projects of this JIRAFilter.

        PRs must be linked to any JIRA issue in the given project keys.

        :return: The projects of this JIRAFilter.
        """
        return self._projects

    @projects.setter
    def projects(self, projects: Optional[List[str]]):
        """Sets the projects of this JIRAFilter.

        PRs must be linked to any JIRA issue in the given project keys.

        :param projects: The projects of this JIRAFilter.
        """
        if projects and self._unmapped:
            raise ValueError("`unmapped` may not be mixed with anything else")
        self._projects = projects

    @property
    def unmapped(self) -> Optional[bool]:
        """Gets the unmapped of this JIRAFilter.

        Select PRs that are not mapped to any JIRA issue. May not be specified with anything else.

        :return: The unmapped of this JIRAFilter.
        """
        return self._unmapped

    @unmapped.setter
    def unmapped(self, unmapped: Optional[bool]):
        """Sets the unmapped of this JIRAFilter.

        Select PRs that are not mapped to any JIRA issue. May not be specified with anything else.

        :param unmapped: The unmapped of this JIRAFilter.
        """
        if unmapped and (self._epics or self._labels_include or
                         self._labels_exclude or self._issue_types):
            raise ValueError("`unmapped` may not be mixed with anything else")
        self._unmapped = unmapped
