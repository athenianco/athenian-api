from typing import Optional

from athenian.api.models.web.base_model_ import Model


class ProductFeature(Model):
    """Client-side product feature definition."""

    attribute_types = {"name": str, "parameters": object}
    attribute_map = {"name": "name", "parameters": "parameters"}

    def __init__(self, name: Optional[str] = None, parameters: Optional[object] = None):
        """ProductFeature - a model defined in OpenAPI

        :param name: The name of this ProductFeature.
        :param parameters: The parameters of this ProductFeature.
        """
        self._name = name
        self._parameters = parameters

    @property
    def name(self) -> str:
        """Gets the name of this ProductFeature.

        :return: The name of this ProductFeature.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this ProductFeature.

        :param name: The name of this ProductFeature.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def parameters(self) -> object:
        """Gets the parameters of this ProductFeature.

        :return: The parameters of this ProductFeature.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: object):
        """Sets the parameters of this ProductFeature.

        :param parameters: The parameters of this ProductFeature.
        """
        if parameters is None:
            raise ValueError("Invalid value for `parameters`, must not be `None`")

        self._parameters = parameters
