from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.jira_epic_child import JIRAEpicChild
from athenian.api.models.web.jira_epic_issue_common import JIRAEpicIssueCommon


class _JIRAEpicSpecials(Model):
    """Details specific to JIRA epics."""

    openapi_types = {
        "project": str,
        "children": List[JIRAEpicChild],
    }

    attribute_map = {
        "project": "project",
        "children": "children",
    }

    __enable_slots__ = False

    def __init__(
        self,
        project: Optional[str] = None,
        children: Optional[List[JIRAEpicChild]] = None,
    ):
        """_JIRAEpicSpecials - a model defined in OpenAPI

        :param project: The id of this _JIRAEpicSpecials.
        :param children: The children of this _JIRAEpicSpecials.
        """
        self._project = project
        self._children = children

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


JIRAEpic = AllOf(JIRAEpicIssueCommon, _JIRAEpicSpecials, module=__name__)
