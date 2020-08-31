from datetime import datetime
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class JIRAEpic(Model):
    """Details about a JIRA epic - a "big" issue with sub-issues."""

    openapi_types = {
        "id": str,
        "title": str,
        "updated": datetime,
        "children": List[str],
    }

    attribute_map = {
        "id": "id",
        "title": "title",
        "updated": "updated",
        "children": "children",
    }

    def __init__(
        self,
        id: Optional[str] = None,
        title: Optional[str] = None,
        updated: Optional[datetime] = None,
        children: Optional[List[str]] = None,
    ):
        """JIRAEpic - a model defined in OpenAPI

        :param id: The id of this JIRAEpic.
        :param title: The title of this JIRAEpic.
        :param updated: The updated of this JIRAEpic.
        :param children: The children of this JIRAEpic.
        """
        self._id = id
        self._title = title
        self._updated = updated
        self._children = children

    @property
    def id(self) -> str:
        """Gets the id of this JIRAEpic.

        :return: The id of this JIRAEpic.
        """
        return self._id

    @id.setter
    def id(self, id: str):
        """Sets the id of this JIRAEpic.

        :param id: The id of this JIRAEpic.
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")

        self._id = id

    @property
    def title(self) -> str:
        """Gets the title of this JIRAEpic.

        :return: The title of this JIRAEpic.
        """
        return self._title

    @title.setter
    def title(self, title: str):
        """Sets the title of this JIRAEpic.

        :param title: The title of this JIRAEpic.
        """
        if title is None:
            raise ValueError("Invalid value for `title`, must not be `None`")

        self._title = title

    @property
    def updated(self) -> datetime:
        """Gets the updated of this JIRAEpic.

        When this epic was last updated.

        :return: The updated of this JIRAEpic.
        """
        return self._updated

    @updated.setter
    def updated(self, updated: datetime):
        """Sets the updated of this JIRAEpic.

        When this epic was last updated.

        :param updated: The updated of this JIRAEpic.
        """
        if updated is None:
            raise ValueError("Invalid value for `updated`, must not be `None`")

        self._updated = updated

    @property
    def children(self) -> List[str]:
        """Gets the children of this JIRAEpic.

        IDs of the owned sub-issues.

        :return: The children of this JIRAEpic.
        """
        return self._children

    @children.setter
    def children(self, children: List[str]):
        """Sets the children of this JIRAEpic.

        IDs of the owned sub-issues.

        :param children: The children of this JIRAEpic.
        """
        if children is None:
            raise ValueError("Invalid value for `children`, must not be `None`")

        self._children = children
