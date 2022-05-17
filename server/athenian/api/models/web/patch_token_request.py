from typing import Optional

from athenian.api.models.web.base_model_ import Model


class PatchTokenRequest(Model):
    """Request body of `/token/{id}` PATCH. Allows changing the token name."""

    attribute_types = {"name": str}
    attribute_map = {"name": "name"}

    def __init__(self, name: Optional[str] = None):
        """PatchTokenRequest - a model defined in OpenAPI

        :param name: The name of this PatchTokenRequest.
        """
        self._name = name

    @property
    def name(self) -> Optional[str]:
        """Gets the name of this PatchTokenRequest.

        New name of the token.

        :return: The name of this PatchTokenRequest.
        """
        return self._name

    @name.setter
    def name(self, name: Optional[str]):
        """Sets the name of this PatchTokenRequest.

        New name of the token.

        :param name: The name of this PatchTokenRequest.
        """
        self._name = name
