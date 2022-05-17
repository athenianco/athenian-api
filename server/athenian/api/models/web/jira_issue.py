from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.jira_epic_issue_common import JIRAEpicIssueCommon
from athenian.api.models.web.pull_request import PullRequest


class _JIRAIssueSpecials(Model, sealed=False):
    """Details specific to JIRA issues."""

    attribute_types = {
        "project": str,
        "prs": Optional[List[PullRequest]],
    }
    attribute_map = {
        "project": "project",
        "prs": "prs",
    }

    def __init__(self,
                 project: Optional[str] = None,
                 prs: Optional[List[PullRequest]] = None):
        """JIRAIssue - a model defined in OpenAPI

        :param project: The project of this JIRAIssue.
        :param prs: The prs of this JIRAIssue.
        """
        self._project = project
        self._prs = prs

    @property
    def project(self) -> str:
        """Gets the project of this JIRAIssue.

        Identifier of the project where this issue exists.

        :return: The project of this JIRAIssue.
        """
        return self._project

    @project.setter
    def project(self, project: str):
        """Sets the project of this JIRAIssue.

        Identifier of the project where this issue exists.

        :param project: The project of this JIRAIssue.
        """
        if project is None:
            raise ValueError("Invalid value for `project`, must not be `None`")

        self._project = project

    @property
    def prs(self) -> Optional[List[PullRequest]]:
        """Gets the prs of this JIRAIssue.

        Details about the mapped PRs. `jira` field is unfilled.

        :return: The prs of this JIRAIssue.
        """
        return self._prs

    @prs.setter
    def prs(self, prs: Optional[List[PullRequest]]):
        """Sets the prs of this JIRAIssue.

        Details about the mapped PRs. `jira` field is unfilled.

        :param prs: The prs of this JIRAIssue.
        """
        self._prs = prs


JIRAIssue = AllOf(JIRAEpicIssueCommon, _JIRAIssueSpecials, name="JIRAIssue", module=__name__)
