from typing import Dict

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_issue import JIRAIssue


class _IncludedJIRAIssues(Model):
    """Mentioned JIRA issues."""

    openapi_types = {"jira": Dict[str, JIRAIssue]}
    attribute_map = {"jira": "jira"}
    __enable_slots__ = False

    def __init__(self, jira: Dict[str, JIRAIssue] = None):
        """IncludedJIRAIssues - a model defined in OpenAPI

        :param jira: The jira of this IncludedJIRAIssues.
        """
        self._jira = jira

    @property
    def jira(self) -> Dict[str, JIRAIssue]:
        """Gets the jira of this IncludedJIRAIssues.

        Mapping JIRA issue ID -> details.

        :return: The jira of this IncludedJIRAIssues.
        """
        return self._jira

    @jira.setter
    def jira(self, jira: Dict[str, JIRAIssue]):
        """Sets the jira of this IncludedJIRAIssues.

        Mapping JIRA issue ID -> details.

        :param jira: The jira of this IncludedJIRAIssues.
        """
        if jira is None:
            raise ValueError("Invalid value for `jira`, must not be `None`")

        self._jira = jira


class IncludedJIRAIssues(_IncludedJIRAIssues):
    """Mentioned JIRA issues."""

    __enable_slots__ = True
