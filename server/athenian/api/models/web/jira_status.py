from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRAStatus(Model):
    """JIRA issue status details."""

    attribute_types = {
        "name": str,
        "stage": str,
        "project": str,
    }

    attribute_map = {
        "name": "name",
        "stage": "stage",
        "project": "project",
    }

    def __init__(
        self,
        name: Optional[str] = None,
        stage: Optional[str] = None,
        project: Optional[str] = None,
    ):
        """JIRAStatus - a model defined in OpenAPI

        :param name: The name of this JIRAStatus.
        :param stage: The stage of this JIRAStatus.
        :param project: The project of this JIRAStatus.
        """
        self._name = name
        self._stage = stage
        self._project = project

    def __lt__(self, other: "JIRAStatus") -> bool:
        """Support sorting."""
        return self._name < other._name

    def __hash__(self) -> int:
        """Support dict/set keys."""
        return hash((self._name, self._project))

    @property
    def name(self) -> str:
        """Gets the name of this JIRAStatus.

        Exact status name.

        :return: The name of this JIRAStatus.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this JIRAStatus.

        Exact status name.

        :param name: The name of this JIRAStatus.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def stage(self) -> str:
        """Gets the stage of this JIRAStatus.

        One of the three status categories.

        :return: The stage of this JIRAStatus.
        """
        return self._stage

    @stage.setter
    def stage(self, stage: str):
        """Sets the stage of this JIRAStatus.

        One of the three status categories.

        :param stage: The stage of this JIRAStatus.
        """
        allowed_values = {"To Do", "In Progress", "Done"}
        if stage not in allowed_values:
            raise ValueError(
                "Invalid value for `stage` (%s), must be one of %s" % (stage, allowed_values),
            )

        self._stage = stage

    @property
    def project(self) -> str:
        """Gets the project of this JIRAStatus.

        Identifier of the project where this status exists.

        :return: The project of this JIRAStatus.
        """
        return self._project

    @project.setter
    def project(self, project: str):
        """Sets the project of this JIRAStatus.

        Identifier of the project where this status exists.

        :param project: The project of this JIRAStatus.
        """
        if project is None:
            raise ValueError("Invalid value for `project`, must not be `None`")

        self._project = project
