from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRAFilter(Model):
    """PR filters related to JIRA traits. The aggregation between each filter type is *AND*."""

    epics: Optional[list[str]]
    labels_include: Optional[list[str]]
    labels_exclude: Optional[list[str]]
    issue_types: Optional[list[str]]
    projects: Optional[list[str]]
    unmapped: Optional[bool]

    def validate_epics(self, epics: Optional[list[str]]) -> Optional[list[str]]:
        """Sets the epics of this JIRAFilter.

        PRs must be linked to at least one JIRA epic from the list.

        :param epics: The epics of this JIRAFilter.
        """
        if epics and getattr(self, "_unmapped", False):
            raise ValueError("`unmapped` may not be mixed with anything else")
        return epics

    def validate_labels_include(self, labels_include: Optional[list[str]]) -> Optional[list[str]]:
        """Sets the labels_include of this JIRAFilter.

        PRs must relate to at least one JIRA issue label from the list. Several labels may be
        concatenated by a comma `,` so that all of them are required.

        :param labels_include: The labels_include of this JIRAFilter.
        """
        if labels_include and getattr(self, "_unmapped", False):
            raise ValueError("`unmapped` may not be mixed with anything else")
        return labels_include

    def validate_labels_exclude(self, labels_exclude: Optional[list[str]]) -> Optional[list[str]]:
        """Sets the labels_exclude of this JIRAFilter.

        PRs cannot relate to JIRA issue labels from the list.

        :param labels_exclude: The labels_exclude of this JIRAFilter.
        """
        if labels_exclude and getattr(self, "_unmapped", False):
            raise ValueError("`unmapped` may not be mixed with anything else")
        return labels_exclude

    def validate_issue_types(self, issue_types: Optional[list[str]]) -> Optional[list[str]]:
        """Sets the issue_types of this JIRAFilter.

        PRs must be linked to certain JIRA issue types, e.g. Bug, Task, Design Document, etc.

        :param issue_types: The issue_types of this JIRAFilter.
        """
        if issue_types and getattr(self, "_unmapped", False):
            raise ValueError("`unmapped` may not be mixed with anything else")
        return issue_types

    def validate_unmapped(self, unmapped: Optional[bool]) -> Optional[bool]:
        """Sets the unmapped of this JIRAFilter.

        Select PRs that are not mapped to any JIRA issue. May not be specified with anything else.

        :param unmapped: The unmapped of this JIRAFilter.
        """
        if unmapped and (
            getattr(self, "_epics", False)
            or getattr(self, "_labels_include", False)
            or getattr(self, "_labels_exclude", False)
            or getattr(self, "_issue_types", False)
        ):
            raise ValueError("`unmapped` may not be mixed with anything else")
        return unmapped
