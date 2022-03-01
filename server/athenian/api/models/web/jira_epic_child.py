from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.jira_epic_issue_common import JIRAEpicIssueCommon


class _JIRAEpicChildSpecials(Model, sealed=False):
    """Details specific to JIRA issues which are children of epics."""

    openapi_types = {
        "type": str,
        "subtasks": int,
        "prs": int,
    }
    attribute_map = {
        "type": "type",
        "subtasks": "subtasks",
        "prs": "prs",
    }

    def __init__(self,
                 type: Optional[str] = None,
                 subtasks: Optional[int] = None,
                 prs: Optional[int] = None):
        """_JIRAEpicChildSpecials - a model defined in OpenAPI

        :param type: The type of this _JIRAEpicChildSpecials.
        :param subtasks: The subtasks of this _JIRAEpicChildSpecials.
        :param prs: The prs of this _JIRAEpicChildSpecials.
        """
        self._type = type
        self._subtasks = subtasks
        self._prs = prs

    @property
    def type(self) -> str:
        """Gets the type of this _JIRAEpicChildSpecials.

        :return: The type of this _JIRAEpicChildSpecials.
        """
        return self._type

    @type.setter
    def type(self, type: str):
        """Sets the type of this _JIRAEpicChildSpecials.

        :param type: The type of this _JIRAEpicChildSpecials.
        """
        if type is None:
            raise ValueError("Invalid value for `type`, must not be `None`")

        self._type = type

    @property
    def subtasks(self) -> int:
        """Gets the subtasks of this _JIRAEpicChildSpecials.

        Number of sub-tasks.

        :return: The subtasks of this _JIRAEpicChildSpecials.
        """
        return self._subtasks

    @subtasks.setter
    def subtasks(self, subtasks: int):
        """Sets the subtasks of this _JIRAEpicChildSpecials.

        Number of sub-tasks.

        :param subtasks: The subtasks of this _JIRAEpicChildSpecials.
        """
        if subtasks is None:
            raise ValueError("Invalid value for `subtasks`, must not be `None`")

        self._subtasks = subtasks

    @property
    def prs(self) -> int:
        """Gets the prs of this _JIRAEpicChildSpecials.

        Number of mapped pull requests.

        :return: The prs of this _JIRAEpicChildSpecials.
        """
        return self._prs

    @prs.setter
    def prs(self, prs: int):
        """Sets the prs of this _JIRAEpicChildSpecials.

        Number of mapped pull requests.

        :param prs: The prs of this _JIRAEpicChildSpecials.
        """
        if prs is None:
            raise ValueError("Invalid value for `prs`, must not be `None`")

        self._prs = prs


JIRAEpicChild = AllOf(JIRAEpicIssueCommon, _JIRAEpicChildSpecials,
                      name="JIRAEpicChild", module=__name__)
