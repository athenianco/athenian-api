from typing import Optional

from athenian.api.models.web.base_model_ import Model


class Versions(Model):
    """Versions of the backend components."""

    attribute_types = {"api": str, "metadata": str}
    attribute_map = {"api": "api", "metadata": "metadata"}

    def __init__(self, api: Optional[str] = None, metadata: Optional[str] = None):
        """Versions - a model defined in OpenAPI

        :param api: The api of this Versions.
        :param metadata: The metadata of this Versions.
        """
        self._api = api
        self._metadata = metadata

    @property
    def api(self) -> str:
        """Gets the api of this Versions.

        :return: The api of this Versions.
        """
        return self._api

    @api.setter
    def api(self, api: str):
        """Sets the api of this Versions.

        :param api: The api of this Versions.
        """
        if api is None:
            raise ValueError("Invalid value for `api`, must not be `None`")

        self._api = api

    @property
    def metadata(self) -> str:
        """Gets the metadata of this Versions.

        :return: The metadata of this Versions.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: str):
        """Sets the metadata of this Versions.

        :param metadata: The metadata of this Versions.
        """
        if metadata is None:
            raise ValueError("Invalid value for `metadata`, must not be `None`")

        self._metadata = metadata
