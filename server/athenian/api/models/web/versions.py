from athenian.api.models.web.base_model_ import Model


class Versions(Model):
    """Versions of the backend components."""

    openapi_types = {"api": str}
    attribute_map = {"api": "api"}

    def __init__(self, api: str = None):
        """Versions - a model defined in OpenAPI

        :param api: The api of this Versions.
        """
        self._api = api

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
