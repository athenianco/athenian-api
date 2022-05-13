from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.jira_epic_child import JIRAEpicChild
from athenian.api.models.web.jira_epic_issue_common import JIRAEpicIssueCommon


class _JIRAEpicSpecials(Model, sealed=False):
    """Details specific to JIRA epics."""

    attribute_types = {
        "project": str,
        "children": List[JIRAEpicChild],
        "prs": int,
    }

    attribute_map = {
        "project": "project",
        "children": "children",
        "prs": "prs",
    }

    def __init__(
        self,
        project: Optional[str] = None,
        children: Optional[List[JIRAEpicChild]] = None,
        prs: Optional[int] = None,
    ):
        """_JIRAEpicSpecials - a model defined in OpenAPI

        :param project: The id of this _JIRAEpicSpecials.
        :param children: The children of this _JIRAEpicSpecials.
        :param prs: The prs of this _JIRAEpicSpecials.
        """
        self._project = project
        self._children = children
        self._prs = prs

    @property
    def project(self) -> str:
        """Gets the project of this _JIRAEpicSpecials.

        Identifier of the project where this epic exists.

        :return: The project of this _JIRAEpicSpecials.
        """
        return self._project

    @project.setter
    def project(self, project: str):
        """Sets the project of this _JIRAEpicSpecials.

        Identifier of the project where this epic exists.

        :param project: The project of this _JIRAEpicSpecials.
        """
        if project is None:
            raise ValueError("Invalid value for `project`, must not be `None`")

        self._project = project

    @property
    def children(self) -> Optional[List[JIRAEpicChild]]:
        """Gets the children of this _JIRAEpicSpecials.

        Details about the child issues.

        :return: The children of this _JIRAEpicSpecials.
        """
        return self._children

    @children.setter
    def children(self, children: Optional[List[JIRAEpicChild]]):
        """Sets the children of this _JIRAEpicSpecials.

        Details about the child issues.

        :param children: The children of this _JIRAEpicSpecials.
        """
        self._children = children

    @property
    def prs(self) -> int:
        """Gets the prs of this _JIRAEpicSpecials.

        Number of mapped pull requests.

        :return: The prs of this _JIRAEpicSpecials.
        """
        return self._prs

    @prs.setter
    def prs(self, prs: int):
        """Sets the prs of this _JIRAEpicSpecials.

        Number of mapped pull requests.

        :param prs: The prs of this _JIRAEpicSpecials.
        """
        if prs is None:
            raise ValueError("Invalid value for `prs`, must not be `None`")

        self._prs = prs


JIRAEpic = AllOf(JIRAEpicIssueCommon, _JIRAEpicSpecials, name="JIRAEpic", module=__name__)
