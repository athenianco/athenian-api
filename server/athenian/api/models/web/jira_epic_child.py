from datetime import datetime
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRAEpicChild(Model):
    """Brief details about a JIRA issue in an epic."""

    openapi_types = {
        "id": str,
        "status": str,
        "type": str,
        "work_began": Optional[datetime],
        "resolved": Optional[datetime],
    }
    attribute_map = {
        "id": "id",
        "status": "status",
        "type": "type",
        "work_began": "work_began",
        "resolved": "resolved",
    }

    def __init__(self,
                 id: Optional[str] = None,
                 status: Optional[str] = None,
                 type: Optional[str] = None,
                 work_began: Optional[datetime] = None,
                 resolved: Optional[datetime] = None):
        """JIRAEpicChild - a model defined in OpenAPI

        :param id: The id of this JIRAEpicChild.
        :param status: The status of this JIRAEpicChild.
        :param type: The type of this JIRAEpicChild.
        :param work_began: The work_began of this JIRAEpicChild.
        :param resolved: The resolved of this JIRAEpicChild.
        """
        self._id = id
        self._status = status
        self._type = type
        self._work_began = work_began
        self._resolved = resolved

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

    @property
    def work_began(self) -> Optional[datetime]:
        """Gets the work_began of this JIRAEpicChild.

        When the issue entered the "In Progress" stage. This timestamp can be missing and is always
        less than or equal to `resolved`.

        :return: The work_began of this JIRAEpicChild.
        """
        return self._work_began

    @work_began.setter
    def work_began(self, work_began: Optional[datetime]):
        """Sets the work_began of this JIRAEpicChild.

        When the issue entered the "In Progress" stage. This timestamp can be missing and is always
        less than or equal to `resolved`.

        :param work_began: The work_began of this JIRAEpicChild.
        """
        self._work_began = work_began

    @property
    def resolved(self) -> Optional[datetime]:
        """Gets the resolved of this JIRAEpicChild.

        When the issue was marked as completed. This timestamp can be missing and is always greater
        than or equal to `work_began`.

        :return: The resolved of this JIRAEpicChild.
        """
        return self._resolved

    @resolved.setter
    def resolved(self, resolved: Optional[datetime]):
        """Sets the resolved of this JIRAEpicChild.

        When the issue was marked as completed. This timestamp can be missing and is always greater
        than or equal to `work_began`.

        :param resolved: The resolved of this JIRAEpicChild.
        """
        self._resolved = resolved
