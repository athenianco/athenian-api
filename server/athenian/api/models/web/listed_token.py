from datetime import datetime
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class ListedToken(Model):
    """Details about a token - without the token itself, which is not stored."""

    attribute_types = {"id": int, "name": str, "last_used": datetime}
    attribute_map = {"id": "id", "name": "name", "last_used": "last_used"}

    def __init__(
        self,
        id: Optional[int] = None,
        name: Optional[str] = None,
        last_used: Optional[datetime] = None,
    ):
        """ListedToken - a model defined in OpenAPI

        :param id: The id of this ListedToken.
        :param name: The name of this ListedToken.
        :param last_used: The last_used of this ListedToken.
        """
        self._id = id
        self._name = name
        self._last_used = last_used

    @property
    def id(self) -> int:
        """Gets the id of this ListedToken.

        Token identifier - can be used in `/token/{id}` DELETE.

        :return: The id of this ListedToken.
        """
        return self._id

    @id.setter
    def id(self, id: int):
        """Sets the id of this ListedToken.

        Token identifier - can be used in `/token/{id}` DELETE.

        :param id: The id of this ListedToken.
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")

        self._id = id

    @property
    def name(self) -> str:
        """Gets the name of this ListedToken.

        Name of the token (see `/token/create`).

        :return: The name of this ListedToken.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this ListedToken.

        Name of the token (see `/token/create`).

        :param name: The name of this ListedToken.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def last_used(self) -> datetime:
        """Gets the last_used of this ListedToken.

        When this token was used last time.

        :return: The last_used of this ListedToken.
        """
        return self._last_used

    @last_used.setter
    def last_used(self, last_used: datetime):
        """Sets the last_used of this ListedToken.

        When this token was used last time.

        :param last_used: The last_used of this ListedToken.
        """
        if last_used is None:
            raise ValueError("Invalid value for `last_used`, must not be `None`")

        self._last_used = last_used
