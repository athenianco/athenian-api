from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class LinkedJIRAIssue(Model):
    """Brief details about a JIRA issue."""

    attribute_types = {
        "id": str,
        "title": str,
        "epic": Optional[str],
        "labels": Optional[List[str]],
        "type": str,
    }

    attribute_map = {
        "id": "id",
        "title": "title",
        "epic": "epic",
        "labels": "labels",
        "type": "type",
    }

    __slots__ = ["_" + k for k in attribute_types]

    def __init__(
        self,
        id: Optional[str] = None,
        title: Optional[str] = None,
        epic: Optional[str] = None,
        labels: Optional[List[str]] = None,
        type: Optional[str] = None,
    ):
        """LinkedJIRAIssue - a model defined in OpenAPI

        :param id: The id of this LinkedJIRAIssue.
        :param title: The title of this LinkedJIRAIssue.
        :param epic: The epic of this LinkedJIRAIssue.
        :param parent: The parent of this LinkedJIRAIssue.
        :param children: The children of this LinkedJIRAIssue.
        :param labels: The labels of this LinkedJIRAIssue.
        :param type: The type of this LinkedJIRAIssue.
        """
        self._id = id
        self._title = title
        self._epic = epic
        self._labels = labels
        self._type = type

    @property
    def id(self) -> str:
        """Gets the id of this LinkedJIRAIssue.

        Identifier of this issue.

        :return: The id of this LinkedJIRAIssue.
        """
        return self._id

    @id.setter
    def id(self, id: str):
        """Sets the id of this LinkedJIRAIssue.

        Identifier of this issue.

        :param id: The id of this LinkedJIRAIssue.
        """
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")

        self._id = id

    @property
    def title(self) -> str:
        """Gets the title of this LinkedJIRAIssue.

        Title of this issue.

        :return: The title of this LinkedJIRAIssue.
        """
        return self._title

    @title.setter
    def title(self, title: str):
        """Sets the title of this LinkedJIRAIssue.

        Title of this issue.

        :param title: The title of this LinkedJIRAIssue.
        """
        if title is None:
            raise ValueError("Invalid value for `title`, must not be `None`")

        self._title = title

    @property
    def epic(self) -> Optional[str]:
        """Gets the epic of this LinkedJIRAIssue.

        Identifier of the epic that owns this issue.

        :return: The epic of this LinkedJIRAIssue.
        """
        return self._epic

    @epic.setter
    def epic(self, epic: Optional[str]):
        """Sets the epic of this LinkedJIRAIssue.

        Identifier of the epic that owns this issue.

        :param epic: The epic of this LinkedJIRAIssue.
        """
        self._epic = epic

    @property
    def labels(self) -> Optional[List[str]]:
        """Gets the labels of this LinkedJIRAIssue.

        List of JIRA labels in this issue.

        :return: The labels of this LinkedJIRAIssue.
        """
        return self._labels

    @labels.setter
    def labels(self, labels: Optional[List[str]]):
        """Sets the labels of this LinkedJIRAIssue.

        List of JIRA labels in this issue.

        :param labels: The labels of this LinkedJIRAIssue.
        """
        self._labels = labels

    @property
    def type(self) -> str:
        """Gets the type of this LinkedJIRAIssue.

        Type of this issue.

        :return: The type of this LinkedJIRAIssue.
        """
        return self._type

    @type.setter
    def type(self, type: str):
        """Sets the type of this LinkedJIRAIssue.

        Type of this issue.

        :param type: The type of this LinkedJIRAIssue.
        """
        if type is None:
            raise ValueError("Invalid value for `type`, must not be `None`")

        self._type = type
