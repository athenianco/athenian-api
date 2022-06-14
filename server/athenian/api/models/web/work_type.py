import re
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.work_type_rule import WorkTypeRule


class WorkType(Model):
    """Definition of a work type - a set of rules to group PRs, releases, etc. together."""

    attribute_types = {"name": str, "color": str, "rules": List[WorkTypeRule]}

    attribute_map = {"name": "name", "color": "color", "rules": "rules"}

    def __init__(
        self,
        name: Optional[str] = None,
        color: Optional[str] = None,
        rules: Optional[List[WorkTypeRule]] = None,
    ):
        """WorkType - a model defined in OpenAPI

        :param name: The name of this WorkType.
        :param color: The color of this WorkType.
        :param rules: The rules of this WorkType.
        """
        self._name = name
        self._color = color
        self._rules = rules

    @property
    def name(self) -> str:
        """Gets the name of this WorkType.

        Work type name. It is unique within the account scope.

        :return: The name of this WorkType.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this WorkType.

        Work type name. It is unique within the account scope.

        :param name: The name of this WorkType.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")
        if len(name) < 1:
            raise ValueError(
                "Invalid value for `name`, length must be greater than or equal to `1`",
            )

        self._name = name

    @property
    def color(self) -> str:
        """Gets the color of this WorkType.

        RGB 24-bit color in hex.

        :return: The color of this WorkType.
        """
        return self._color

    @color.setter
    def color(self, color: str):
        """Sets the color of this WorkType.

        RGB 24-bit color in hex.

        :param color: The color of this WorkType.
        """
        if color is None:
            raise ValueError("Invalid value for `color`, must not be `None`")
        if not re.fullmatch(r"[0-9a-fA-F]{6}", color):
            raise ValueError(
                "Invalid value for `color`, must be a follow pattern or equal to "
                "`/^[0-9a-fA-F]{6}$/`",
            )

        self._color = color

    @property
    def rules(self) -> List[WorkTypeRule]:
        """Gets the rules of this WorkType.

        :return: The rules of this WorkType.
        """
        return self._rules

    @rules.setter
    def rules(self, rules: List[WorkTypeRule]):
        """Sets the rules of this WorkType.

        :param rules: The rules of this WorkType.
        """
        if rules is None:
            raise ValueError("Invalid value for `rules`, must not be `None`")

        self._rules = rules
