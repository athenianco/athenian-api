from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.jira_epic_issue_common import JIRAEpicIssueCommon


class _JIRAEpicChildSpecials(Model):
    """Details specific to JIRA issues."""

    openapi_types = {
        "type": str,
        "subtasks": int,
    }
    attribute_map = {
        "type": "type",
        "subtasks": "subtasks",
    }

    __enable_slots__ = False

    def __init__(self,
                 type: Optional[str] = None,
                 subtasks: Optional[int] = None):
        """JIRAEpicChild - a model defined in OpenAPI

        :param type: The type of this JIRAEpicChild.
        :param subtasks: The subtasks of this JIRAEpicChild.
        """
        self._type = type
        self._subtasks = subtasks

    @property
    def type(self) -> str:
        """Gets the type of this JIRAEpicChild.

        :return: The type of this JIRAEpicChild.
        """
        return self._type

    @type.setter
    def type(self, type: str):
        """Sets the type of this JIRAEpicChild.

        :param type: The type of this JIRAEpicChild.
        """
        if type is None:
            raise ValueError("Invalid value for `type`, must not be `None`")

        self._type = type

    @property
    def subtasks(self) -> int:
        """Gets the subtasks of this JIRAEpicChild.

        Number of sub-tasks.

        :return: The subtasks of this JIRAEpicChild.
        """
        return self._subtasks

    @subtasks.setter
    def subtasks(self, subtasks: int):
        """Sets the subtasks of this JIRAEpicChild.

        Number of sub-tasks.

        :param subtasks: The subtasks of this JIRAEpicChild.
        """
        if subtasks is None:
            raise ValueError("Invalid value for `subtasks`, must not be `None`")

        self._subtasks = subtasks


JIRAEpicChild = AllOf(JIRAEpicIssueCommon, _JIRAEpicChildSpecials, module=__name__)
