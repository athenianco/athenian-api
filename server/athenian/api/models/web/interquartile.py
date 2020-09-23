from typing import Optional

from athenian.api.models.web.base_model_ import Model


class Interquartile(Model):
    """Middle 50% range."""

    openapi_types = {"left": float, "right": float}
    attribute_map = {"left": "left", "right": "right"}

    def __init__(self,
                 left: Optional[float] = None,
                 right: Optional[float] = None):
        """Interquartile - a model defined in OpenAPI

        :param left: The left of this Interquartile.
        :param right: The right of this Interquartile.
        """
        self._left = left
        self._right = right

    @property
    def left(self) -> float:
        """Gets the left of this Interquartile.

        :return: The left of this Interquartile.
        """
        return self._left

    @left.setter
    def left(self, left: float):
        """Sets the left of this Interquartile.

        :param left: The left of this Interquartile.
        """
        if left is None:
            raise ValueError("Invalid value for `left`, must not be `None`")

        self._left = left

    @property
    def right(self) -> float:
        """Gets the right of this Interquartile.

        :return: The right of this Interquartile.
        """
        return self._right

    @right.setter
    def right(self, right: float):
        """Sets the right of this Interquartile.

        :param right: The right of this Interquartile.
        """
        if right is None:
            raise ValueError("Invalid value for `right`, must not be `None`")

        self._right = right
