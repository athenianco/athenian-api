from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRAIssueType(Model):
    """Details about a JIRA issue type."""

    attribute_types = {
        "name": str,
        "count": int,
        "image": str,
        "project": str,
        "is_subtask": bool,
        "is_epic": bool,
        "normalized_name": str,
    }
    attribute_map = {
        "name": "name",
        "count": "count",
        "image": "image",
        "project": "project",
        "is_subtask": "is_subtask",
        "is_epic": "is_epic",
        "normalized_name": "normalized_name",
    }

    def __init__(self,
                 name: Optional[str] = None,
                 count: Optional[int] = None,
                 image: Optional[str] = None,
                 project: Optional[str] = None,
                 is_subtask: Optional[bool] = None,
                 is_epic: Optional[bool] = None,
                 normalized_name: Optional[str] = None):
        """JIRAIssueType - a model defined in OpenAPI

        :param name: The name of this JIRAIssueType.
        :param count: The count of this JIRAIssueType.
        :param image: The image of this JIRAIssueType.
        :param is_subtask: The is_subtask of this JIRAIssueType.
        :param is_epic: The is_epic of this JIRAIssueType.
        :param normalized_name: The normalized_name of this JIRAIssueType.
        """
        self._name = name
        self._count = count
        self._image = image
        self._project = project
        self._is_subtask = is_subtask
        self._is_epic = is_epic
        self._normalized_name = normalized_name

    @property
    def name(self) -> str:
        """Gets the name of this JIRAIssueType.

        Name of the issue type.

        :return: The name of this JIRAIssueType.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this JIRAIssueType.

        Name of the issue type.

        :param name: The name of this JIRAIssueType.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def count(self) -> int:
        """Gets the count of this JIRAIssueType.

        Number of issues that satisfy the filters and belong to this type.

        :return: The count of this JIRAIssueType.
        """
        return self._count

    @count.setter
    def count(self, count: int):
        """Sets the count of this JIRAIssueType.

        Number of issues that satisfy the filters and belong to this type.

        :param count: The count of this JIRAIssueType.
        """
        if count is None:
            raise ValueError("Invalid value for `count`, must not be `None`")
        if count is not None and count < 1:
            raise ValueError(
                "Invalid value for `count`, must be a value greater than or equal to `1`")

        self._count = count

    @property
    def image(self) -> str:
        """Gets the image of this JIRAIssueType.

        Icon URL.

        :return: The image of this JIRAIssueType.
        """
        return self._image

    @image.setter
    def image(self, image: str):
        """Sets the image of this JIRAIssueType.

        Icon URL.

        :param image: The image of this JIRAIssueType.
        """
        if image is None:
            raise ValueError("Invalid value for `image`, must not be `None`")

        self._image = image

    @property
    def project(self) -> str:
        """Gets the project of this JIRAIssueType.

        Bound project identifier.

        :return: The project of this JIRAIssueType.
        """
        return self._project

    @project.setter
    def project(self, project: str):
        """Sets the project of this JIRAIssueType.

        Bound project identifier.

        :param project: The project of this JIRAIssueType.
        """
        if project is None:
            raise ValueError("Invalid value for `project`, must not be `None`")

        self._project = project

    @property
    def is_subtask(self) -> bool:
        """Gets the is_subtask of this JIRAIssueType.

        Value indicating whether this issue type refers to a sub-task.

        :return: The is_subtask of this JIRAIssueType.
        """
        return self._is_subtask

    @is_subtask.setter
    def is_subtask(self, is_subtask: bool):
        """Sets the is_subtask of this JIRAIssueType.

        Value indicating whether this issue type refers to a sub-task.

        :param is_subtask: The is_subtask of this JIRAIssueType.
        """
        if is_subtask is None:
            raise ValueError("Invalid value for `is_subtask`, must not be `None`")

        self._is_subtask = is_subtask

    @property
    def is_epic(self) -> bool:
        """Gets the is_epic of this JIRAIssueType.

        Value indicating whether this issue type refers to an epic.

        :return: The is_epic of this JIRAIssueType.
        """
        return self._is_epic

    @is_epic.setter
    def is_epic(self, is_epic: bool):
        """Sets the is_epic of this JIRAIssueType.

        Value indicating whether this issue type refers to an epic.

        :param is_epic: The is_epic of this JIRAIssueType.
        """
        if is_epic is None:
            raise ValueError("Invalid value for `is_epic`, must not be `None`")

        self._is_epic = is_epic

    @property
    def normalized_name(self) -> str:
        """Gets the normalized_name of this JIRAIssueType.

        Normalized name of the issue type. For example, "Sub-tásks" becomes "subtask".

        :return: The normalized_name of this JIRAIssueType.
        """
        return self._normalized_name

    @normalized_name.setter
    def normalized_name(self, normalized_name: str):
        """Sets the normalized_name of this JIRAIssueType.

        Normalized name of the issue type. For example, "Sub-tásks" becomes "subtask".

        :param normalized_name: The normalized_name of this JIRAIssueType.
        """
        if normalized_name is None:
            raise ValueError("Invalid value for `normalized_name`, must not be `None`")

        self._normalized_name = normalized_name
