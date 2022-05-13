from typing import Optional

from athenian.api.models.web.base_model_ import Model


class CreatedToken(Model):
    """Value and ID of the generated Personal Access Token."""

    attribute_types = {"id": int, "token": str}
    attribute_map = {"id": "id", "token": "token"}

    def __init__(self, id: Optional[int] = None, token: Optional[str] = None):
        """CreatedToken - a model defined in OpenAPI

        :param id: The id of this CreatedToken.
        :param token: The token of this CreatedToken.
        """
        self._id = id
        self._token = token

    @property
    def id(self) -> int:
        """Gets the id of this CreatedToken.

        Token identifier - can be used in `/token/{id}` DELETE.

        :return: The id of this CreatedToken.
        """
        return self._id

    @id.setter
    def id(self, id: int):
        """Sets the id of this CreatedToken.

        Token identifier - can be used in `/token/{id}` DELETE.

        :param id: The id of this CreatedToken.
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")

        self._id = id

    @property
    def token(self) -> str:
        """Gets the token of this CreatedToken.

        Secret token - not stored server-side!

        :return: The token of this CreatedToken.
        """
        return self._token

    @token.setter
    def token(self, token: str):
        """Sets the token of this CreatedToken.

        Secret token - not stored server-side!

        :param token: The token of this CreatedToken.
        """
        if token is None:
            raise ValueError("Invalid value for `token`, must not be `None`")

        self._token = token
