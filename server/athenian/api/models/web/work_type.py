import re

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.work_type_rule import WorkTypeRule


class WorkType(Model):
    """Definition of a work type - a set of rules to group PRs, releases, etc. together."""

    name: str
    color: str
    rules: list[WorkTypeRule]

    def validate_name(self, name: str) -> str:
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

        return name

    def validate_color(self, color: str) -> str:
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

        return color
