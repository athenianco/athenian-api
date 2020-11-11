from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRAEpicChild(Model):
    """Brief details about a JIRA issue in an epic."""

    openapi_types = {
        "id": str,
        "status": str,
        "type": str,
    }
    attribute_map = {
        "id": "id",
        "status": "status",
        "type": "type",
    }

    def __init__(self,
                 id: Optional[str] = None,
                 status: Optional[str] = None,
                 type: Optional[str] = None):
        """JIRAEpicChild - a model defined in OpenAPI

        :param id: The id of this JIRAEpicChild.
        :param status: The status of this JIRAEpicChild.
        :param type: The type of this JIRAEpicChild.
        """
        self._id = id
        self._status = status
        self._type = type

    @property
    def id(self) -> str:
        """Gets the id of this JIRAEpicChild.

        :return: The id of this JIRAEpicChild.
        """
        return self._id

    @id.setter
    def id(self, id: str):
        """Sets the id of this JIRAEpicChild.

        :param id: The id of this JIRAEpicChild.
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")

        self._id = id

    @property
    def status(self) -> str:
        """Gets the status of this JIRAEpicChild.

        :return: The status of this JIRAEpicChild.
        """
        return self._status

    @status.setter
    def status(self, status: str):
        """Sets the status of this JIRAEpicChild.

        :param status: The status of this JIRAEpicChild.
        """
        if status is None:
            raise ValueError("Invalid value for `status`, must not be `None`")

        self._status = status

    @property
    def type(self) -> str:
        """Gets the type of this JIRAEpicChild.

        :return: The type of this JIRAEpicChild.
        """
        return self._type

    @type.setter
    def type(self, type: str):
        """Sets the type of this JIRAEpicChild.

        :param type: The type of this JIRAEpicChild.
        """
        if type is None:
            raise ValueError("Invalid value for `type`, must not be `None`")

        self._type = type
