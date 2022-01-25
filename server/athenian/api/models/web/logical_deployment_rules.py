from typing import Dict, List, Optional

from athenian.api.models.web.base_model_ import Model


class LogicalDeploymentRules(Model):
    """Rules to match deployments to logical repository."""

    openapi_types = {"title": str, "labels_include": Dict[str, List[str]]}

    attribute_map = {"title": "title", "labels_include": "labels_include"}

    def __init__(
        self,
        title: Optional[str] = None,
        labels_include: Optional[Dict[str, List[str]]] = None,
    ):
        """LogicalDeploymentRules - a model defined in OpenAPI

        :param title: The title of this LogicalDeploymentRules.
        :param labels_include: The labels_include of this LogicalDeploymentRules.
        """
        self._title = title
        self._labels_include = labels_include

    @property
    def title(self) -> Optional[str]:
        """Gets the title of this LogicalDeploymentRules.

        Regular expression to match the deployment name. It must be a match starting from
        the start of the string.

        :return: The title of this LogicalDeploymentRules.
        """
        return self._title

    @title.setter
    def title(self, title: Optional[str]):
        """Sets the title of this LogicalDeploymentRules.

        Regular expression to match the deployment name. It must be a match starting from
        the start of the string.

        :param title: The title of this LogicalDeploymentRules.
        """
        self._title = title

    @property
    def labels_include(self) -> Optional[Dict[str, List[str]]]:
        """Gets the labels_include of this LogicalDeploymentRules.

        Match deployments labeled by any key and having at least one value in the list.

        :return: The labels_include of this LogicalDeploymentRules.
        """
        return self._labels_include

    @labels_include.setter
    def labels_include(self, labels_include: Optional[Dict[str, List[str]]]):
        """Sets the labels_include of this LogicalDeploymentRules.

        Match deployments labeled by any key and having at least one value in the list.

        :param labels_include: The labels_include of this LogicalDeploymentRules.
        """
        self._labels_include = labels_include
