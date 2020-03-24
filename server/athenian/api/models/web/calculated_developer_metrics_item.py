from typing import List, Optional, Union

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.for_set import ForSet


class CalculatedDeveloperMetricsItem(Model):
    """Measured developer metrics for each `DeveloperMetricsRequest.for`."""

    def __init__(self,
                 for_: Optional[ForSet] = None,
                 values: Optional[List[List[Union[int, float]]]] = None):
        """CalculatedDeveloperMetricsItem - a model defined in OpenAPI

        :param for_: The for_ of this CalculatedDeveloperMetricsItem.
        :param values: The values of this CalculatedDeveloperMetricsItem.
        """
        self.openapi_types = {"for_": ForSet, "values": List[List[Union[int, float]]]}

        self.attribute_map = {"for_": "for", "values": "values"}

        self._for_ = for_
        self._values = values

    @property
    def for_(self) -> ForSet:
        """Gets the for_ of this CalculatedDeveloperMetricsItem.

        :return: The for_ of this CalculatedDeveloperMetricsItem.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: ForSet):
        """Sets the for_ of this CalculatedDeveloperMetricsItem.

        :param for_: The for_ of this CalculatedDeveloperMetricsItem.
        """
        if for_ is None:
            raise ValueError("Invalid value for `for_`, must not be `None`")

        self._for_ = for_

    @property
    def values(self) -> List[List[Union[int, float]]]:
        """Gets the values of this CalculatedDeveloperMetricsItem.

        The sequence matches `CalculatedDeveloperMetricsItem.for.developers`.

        :return: The values of this CalculatedDeveloperMetricsItem.
        """
        return self._values

    @values.setter
    def values(self, values: List[List[Union[int, float]]]):
        """Sets the values of this CalculatedDeveloperMetricsItem.

        The sequence matches `CalculatedDeveloperMetricsItem.for.developers`.

        :param values: The values of this CalculatedDeveloperMetricsItem.
        """
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")

        self._values = values
