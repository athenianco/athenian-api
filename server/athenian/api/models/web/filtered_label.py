from typing import Optional

from athenian.api.models.web.pull_request_label import PullRequestLabel


class FilteredLabel(PullRequestLabel):
    """Details about a label and some basic stats."""

    attribute_types = PullRequestLabel.attribute_types.copy()
    attribute_types["used_prs"] = int
    attribute_map = PullRequestLabel.attribute_map.copy()
    attribute_map["used_prs"] = "used_prs"

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        color: Optional[str] = None,
        used_prs: Optional[int] = None,
    ):
        """FilteredLabel - a model defined in OpenAPI

        :param name: The name of this FilteredLabel.
        :param description: The description of this FilteredLabel.
        :param color: The color of this FilteredLabel.
        :param used_prs: The used_prs of this FilteredLabel.
        """
        super().__init__(name=name, description=description, color=color)
        self._used_prs = used_prs

    @property
    def used_prs(self) -> int:
        """Gets the used_prs of this FilteredLabel.

        :return: The used_prs of this FilteredLabel.
        """
        return self._used_prs

    @used_prs.setter
    def used_prs(self, used_prs: int):
        """Sets the used_prs of this FilteredLabel.

        :param used_prs: The used_prs of this FilteredLabel.
        """
        if used_prs is None:
            raise ValueError("Invalid value for `used_prs`, must not be `None`")

        self._used_prs = used_prs
