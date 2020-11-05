from typing import List

from athenian.api.models.web.base_model_ import Model


class JIRAMetricsRequestWith(Model):
    """Group of JIRA issue participant names split by role."""

    openapi_types = {
        "assignees": List[str],
        "reporters": List[str],
        "commenters": List[str],
    }
    attribute_map = {
        "assignees": "assignees",
        "reporters": "reporters",
        "commenters": "commenters",
    }

    def __init__(
        self,
        assignees: List[str] = None,
        reporters: List[str] = None,
        commenters: List[str] = None,
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
    def assignees(self) -> List[str]:
        """Gets the assignees of this JIRAMetricsRequestWith.

        Selected issue assignee users.

        :return: The assignees of this JIRAMetricsRequestWith.
        """
        return self._assignees

    @assignees.setter
    def assignees(self, assignees: List[str]):
        """Sets the assignees of this JIRAMetricsRequestWith.

        Selected issue assignee users.

        :param assignees: The assignees of this JIRAMetricsRequestWith.
        """
        self._assignees = assignees

    @property
    def reporters(self) -> List[str]:
        """Gets the reporters of this JIRAMetricsRequestWith.

        Selected issue reporter users.

        :return: The reporters of this JIRAMetricsRequestWith.
        """
        return self._reporters

    @reporters.setter
    def reporters(self, reporters: List[str]):
        """Sets the reporters of this JIRAMetricsRequestWith.

        Selected issue reporter users.

        :param reporters: The reporters of this JIRAMetricsRequestWith.
        """
        self._reporters = reporters

    @property
    def commenters(self) -> List[str]:
        """Gets the commenters of this JIRAMetricsRequestWith.

        Selected issue commenter users.

        :return: The commenters of this JIRAMetricsRequestWith.
        """
        return self._commenters

    @commenters.setter
    def commenters(self, commenters: List[str]):
        """Sets the commenters of this JIRAMetricsRequestWith.

        Selected issue commenter users.

        :param commenters: The commenters of this JIRAMetricsRequestWith.
        """
        self._commenters = commenters
