from typing import Dict, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.linked_jira_issue import LinkedJIRAIssue


class _IncludedJIRAIssues(Model, sealed=False):
    """Mentioned JIRA issues."""

    openapi_types = {"jira": Optional[Dict[str, LinkedJIRAIssue]]}
    attribute_map = {"jira": "jira"}

    def __init__(self, jira: Optional[Dict[str, LinkedJIRAIssue]] = None):
        """IncludedJIRAIssues - a model defined in OpenAPI

        :param jira: The jira of this IncludedJIRAIssues.
        """
        self._jira = jira

    @property
    def jira(self) -> Optional[Dict[str, LinkedJIRAIssue]]:
        """Gets the jira of this IncludedJIRAIssues.

        Mapping JIRA issue ID -> details.

        :return: The jira of this IncludedJIRAIssues.
        """
        return self._jira

    @jira.setter
    def jira(self, jira: Optional[Dict[str, LinkedJIRAIssue]]):
        """Sets the jira of this IncludedJIRAIssues.

        Mapping JIRA issue ID -> details.

        :param jira: The jira of this IncludedJIRAIssues.
        """
        self._jira = jira


class IncludedJIRAIssues(_IncludedJIRAIssues):
    """Mentioned JIRA issues."""
