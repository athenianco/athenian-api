from datetime import datetime
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class Share(Model):
    """Saved UI views state with metadata."""

    openapi_types = {"author": str, "created": datetime, "data": object}
    attribute_map = {"author": "author", "created": "created", "data": "data"}

    def __init__(
        self,
        author: Optional[str] = None,
        created: Optional[datetime] = None,
        data: Optional[object] = None,
    ):
        """Share - a model defined in OpenAPI

        :param author: The author of this Share.
        :param created: The created of this Share.
        :param data: The data of this Share.
        """
        self._author = author
        self._created = created
        self._data = data

    @property
    def author(self) -> str:
        """Gets the author of this Share.

        User name who submitted.

        :return: The author of this Share.
        """
        return self._author

    @author.setter
    def author(self, author: str):
        """Sets the author of this Share.

        User name who submitted.

        :param author: The author of this Share.
        """
        if author is None:
            raise ValueError("Invalid value for `author`, must not be `None`")

        self._author = author

    @property
    def created(self) -> datetime:
        """Gets the created of this Share.

        Submission timestamp.

        :return: The created of this Share.
        """
        return self._created

    @created.setter
    def created(self, created: datetime):
        """Sets the created of this Share.

        Submission timestamp.

        :param created: The created of this Share.
        """
        if created is None:
            raise ValueError("Invalid value for `created`, must not be `None`")

        self._created = created

    @property
    def data(self) -> object:
        """Gets the data of this Share.

        Saved object.

        :return: The data of this Share.
        """
        return self._data

    @data.setter
    def data(self, data: object):
        """Sets the data of this Share.

        Saved object.

        :param data: The data of this Share.
        """
        if data is None:
            raise ValueError("Invalid value for `data`, must not be `None`")

        self._data = data
