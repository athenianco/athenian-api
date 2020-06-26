from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class JIRAIssue(Model):
    """JIRA issue details"""

    openapi_types = {
        "id": str,
        "title": str,
        "epic": str,
        "parent": str,
        "children": List[str],
        "labels": List[str],
        "type": str,
    }

    attribute_map = {
        "id": "id",
        "title": "title",
        "epic": "epic",
        "parent": "parent",
        "children": "children",
        "labels": "labels",
        "type": "type",
    }

    __slots__ = ["_" + k for k in openapi_types]

    def __init__(
        self,
        id: Optional[str] = None,
        title: Optional[str] = None,
        epic: Optional[str] = None,
        parent: Optional[str] = None,
        children: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        type: Optional[str] = None,
    ):
        """JIRAIssue - a model defined in OpenAPI

        :param id: The id of this JIRAIssue.
        :param title: The title of this JIRAIssue.
        :param epic: The epic of this JIRAIssue.
        :param parent: The parent of this JIRAIssue.
        :param children: The children of this JIRAIssue.
        :param labels: The labels of this JIRAIssue.
        :param type: The type of this JIRAIssue.
        """
        self._id = id
        self._title = title
        self._epic = epic
        self._parent = parent
        self._children = children
        self._labels = labels
        self._type = type

    @property
    def id(self) -> str:
        """Gets the id of this JIRAIssue.

        Identifier of this issue.

        :return: The id of this JIRAIssue.
        """
        return self._id

    @id.setter
    def id(self, id: str):
        """Sets the id of this JIRAIssue.

        Identifier of this issue.

        :param id: The id of this JIRAIssue.
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")

        self._id = id

    @property
    def title(self) -> str:
        """Gets the title of this JIRAIssue.

        Title of this issue.

        :return: The title of this JIRAIssue.
        """
        return self._title

    @title.setter
    def title(self, title: str):
        """Sets the title of this JIRAIssue.

        Title of this issue.

        :param title: The title of this JIRAIssue.
        """
        if title is None:
            raise ValueError("Invalid value for `title`, must not be `None`")

        self._title = title

    @property
    def epic(self) -> str:
        """Gets the epic of this JIRAIssue.

        Identifier of the epic that owns this issue.

        :return: The epic of this JIRAIssue.
        """
        return self._epic

    @epic.setter
    def epic(self, epic: str):
        """Sets the epic of this JIRAIssue.

        Identifier of the epic that owns this issue.

        :param epic: The epic of this JIRAIssue.
        """
        self._epic = epic

    @property
    def parent(self) -> str:
        """Gets the parent of this JIRAIssue.

        If this issue is a subissue, identifier of the higher level issue.

        :return: The parent of this JIRAIssue.
        """
        return self._parent

    @parent.setter
    def parent(self, parent: str):
        """Sets the parent of this JIRAIssue.

        If this issue is a subissue, identifier of the higher level issue.

        :param parent: The parent of this JIRAIssue.
        """
        self._parent = parent

    @property
    def children(self) -> List[str]:
        """Gets the children of this JIRAIssue.

        If this issue has subissues, identifiers of all the subissues.

        :return: The children of this JIRAIssue.
        """
        return self._children

    @children.setter
    def children(self, children: List[str]):
        """Sets the children of this JIRAIssue.

        If this issue has subissues, identifiers of all the subissues.

        :param children: The children of this JIRAIssue.
        """
        self._children = children

    @property
    def labels(self) -> List[str]:
        """Gets the labels of this JIRAIssue.

        List of JIRA labels in this issue.

        :return: The labels of this JIRAIssue.
        """
        return self._labels

    @labels.setter
    def labels(self, labels: List[str]):
        """Sets the labels of this JIRAIssue.

        List of JIRA labels in this issue.

        :param labels: The labels of this JIRAIssue.
        """
        self._labels = labels

    @property
    def type(self) -> str:
        """Gets the type of this JIRAIssue.

        Type of this issue.

        :return: The type of this JIRAIssue.
        """
        return self._type

    @type.setter
    def type(self, type: str):
        """Sets the type of this JIRAIssue.

        Type of this issue.

        :param type: The type of this JIRAIssue.
        """
        if type is None:
            raise ValueError("Invalid value for `type`, must not be `None`")

        self._type = type
