from datetime import timedelta
from typing import Optional, Union

from athenian.api.models.web.base_model_ import Model


class Interquartile(Model):
    """Middle 50% range."""

    attribute_types = {"left": Union[float, timedelta], "right": Union[float, timedelta]}
    attribute_map = {"left": "left", "right": "right"}

    def __init__(self,
                 left: Optional[Union[float, timedelta]] = None,
                 right: Optional[Union[float, timedelta]] = None):
        """Interquartile - a model defined in OpenAPI

        :param left: The left of this Interquartile.
        :param right: The right of this Interquartile.
        """
        self._left = left
        self._right = right

    @property
    def left(self) -> Union[float, timedelta]:
        """Gets the left of this Interquartile.

        :return: The left of this Interquartile.
        """
        return self._left

    @left.setter
    def left(self, left: Union[float, timedelta]):
        """Sets the left of this Interquartile.

        :param left: The left of this Interquartile.
        """
        if left is None:
            raise ValueError("Invalid value for `left`, must not be `None`")

        self._left = left

    @property
    def right(self) -> Union[float, timedelta]:
        """Gets the right of this Interquartile.

        :return: The right of this Interquartile.
        """
        return self._right

    @right.setter
    def right(self, right: Union[float, timedelta]):
        """Sets the right of this Interquartile.

        :param right: The right of this Interquartile.
        """
        if right is None:
            raise ValueError("Invalid value for `right`, must not be `None`")

        self._right = right
