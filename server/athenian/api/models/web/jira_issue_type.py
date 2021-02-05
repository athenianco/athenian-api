from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRAIssueType(Model):
    """Details about a JIRA issue type."""

    openapi_types = {
        "name": str,
        "count": int,
        "image": str,
        "project": str,
    }
    attribute_map = {
        "name": "name",
        "count": "count",
        "image": "image",
        "project": "project",
    }

    def __init__(self,
                 name: Optional[str] = None,
                 count: Optional[int] = None,
                 image: Optional[str] = None,
                 project: Optional[str] = None):
        """JIRAIssueType - a model defined in OpenAPI

        :param name: The name of this JIRAIssueType.
        :param count: The count of this JIRAIssueType.
        :param image: The image of this JIRAIssueType.
        """
        self._name = name
        self._count = count
        self._image = image
        self._project = project

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
