from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class JIRAMetricsRequestWith(Model):
    """Group of JIRA issue participant names split by role."""

    openapi_types = {
        "assignees": Optional[List[Optional[str]]],
        "reporters": Optional[List[str]],
        "commenters": Optional[List[str]],
    }
    attribute_map = {
        "assignees": "assignees",
        "reporters": "reporters",
        "commenters": "commenters",
    }

    def __init__(
        self,
        assignees: Optional[List[Optional[str]]] = None,
        reporters: Optional[List[str]] = None,
        commenters: Optional[List[str]] = None,
    ):
        """JIRAMetricsRequestWith - a model defined in OpenAPI

        :param assignees: The assignees of this JIRAMetricsRequestWith.
        :param reporters: The reporters of this JIRAMetricsRequestWith.
        :param commenters: The commenters of this JIRAMetricsRequestWith.
        """
        self._assignees = assignees
        self._reporters = reporters
        self._commenters = commenters

    @property
    def assignees(self) -> Optional[List[Optional[str]]]:
        """Gets the assignees of this JIRAMetricsRequestWith.

        Selected issue assignee users. `null` means unassigned.

        :return: The assignees of this JIRAMetricsRequestWith.
        """
        return self._assignees

    @assignees.setter
    def assignees(self, assignees: Optional[List[Optional[str]]]):
        """Sets the assignees of this JIRAMetricsRequestWith.

        Selected issue assignee users. `null` means unassigned.

        :param assignees: The assignees of this JIRAMetricsRequestWith.
        """
        self._assignees = assignees

    @property
    def reporters(self) -> Optional[List[str]]:
        """Gets the reporters of this JIRAMetricsRequestWith.

        Selected issue reporter users.

        :return: The reporters of this JIRAMetricsRequestWith.
        """
        return self._reporters

    @reporters.setter
    def reporters(self, reporters: Optional[List[str]]):
        """Sets the reporters of this JIRAMetricsRequestWith.

        Selected issue reporter users.

        :param reporters: The reporters of this JIRAMetricsRequestWith.
        """
        for i, reporter in enumerate(reporters):
            if reporter is None:
                raise ValueError("`reporters[%d]` cannot be null" % i)
        self._reporters = reporters

    @property
    def commenters(self) -> Optional[List[str]]:
        """Gets the commenters of this JIRAMetricsRequestWith.

        Selected issue commenter users.

        :return: The commenters of this JIRAMetricsRequestWith.
        """
        return self._commenters

    @commenters.setter
    def commenters(self, commenters: Optional[List[str]]):
        """Sets the commenters of this JIRAMetricsRequestWith.

        Selected issue commenter users.

        :param commenters: The commenters of this JIRAMetricsRequestWith.
        """
        for i, commenter in enumerate(commenters):
            if commenter is None:
                raise ValueError("`commenters[%d]` cannot be null" % i)
        self._commenters = commenters
