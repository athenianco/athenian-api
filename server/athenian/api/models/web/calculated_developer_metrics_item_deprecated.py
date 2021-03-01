from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.for_set_developers import ForSetDevelopers


class CalculatedDeveloperMetricsItemDeprecated(Model):
    """Measured developer metrics for each `DeveloperMetricsRequest.for`."""

    openapi_types = {"for_": ForSetDevelopers, "values": List[List[object]]}
    attribute_map = {"for_": "for", "values": "values"}

    def __init__(self,
                 for_: Optional[ForSetDevelopers] = None,
                 values: Optional[List[List[object]]] = None):
        """CalculatedDeveloperMetricsItemDeprecated - a model defined in OpenAPI

        :param for_: The for_ of this CalculatedDeveloperMetricsItemDeprecated.
        :param values: The values of this CalculatedDeveloperMetricsItemDeprecated.
        """
        self._for_ = for_
        self._values = values

    @property
    def for_(self) -> ForSetDevelopers:
        """Gets the for_ of this CalculatedDeveloperMetricsItemDeprecated.

        :return: The for_ of this CalculatedDeveloperMetricsItemDeprecated.
        """
        return self._for_

    @for_.setter
    def for_(self, for_: ForSetDevelopers):
        """Sets the for_ of this CalculatedDeveloperMetricsItemDeprecated.

        :param for_: The for_ of this CalculatedDeveloperMetricsItemDeprecated.
        """
        if for_ is None:
            raise ValueError("Invalid value for `for_`, must not be `None`")

        self._for_ = for_

    @property
    def values(self) -> List[List[object]]:
        """Gets the values of this CalculatedDeveloperMetricsItemDeprecated.

        The sequence matches `CalculatedDeveloperMetricsItemDeprecated.for.developers`.

        :return: The values of this CalculatedDeveloperMetricsItemDeprecated.
        """
        return self._values

    @values.setter
    def values(self, values: List[List[object]]):
        """Sets the values of this CalculatedDeveloperMetricsItemDeprecated.

        The sequence matches `CalculatedDeveloperMetricsItemDeprecated.for.developers`.

        :param values: The values of this CalculatedDeveloperMetricsItemDeprecated.
        """
        if values is None:
            raise ValueError("Invalid value for `values`, must not be `None`")

        self._values = values
