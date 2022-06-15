from typing import Any, Dict, Optional

from athenian.api.models.web.base_model_ import Model


class CommonDeploymentProperties(Model, sealed=False):
    """Define `with_labels` and `without_labels` properties."""

    attribute_types = {
        "with_labels": Optional[object],
        "without_labels": Optional[object],
    }

    attribute_map = {
        "with_labels": "with_labels",
        "without_labels": "without_labels",
    }

    def __init__(
        self,
        with_labels: Optional[Dict[str, Any]] = None,
        without_labels: Optional[Dict[str, Any]] = None,
    ):
        """CommonDeploymentProperties - a model defined in OpenAPI

        :param with_labels: The with_labels of this CommonDeploymentProperties.
        :param without_labels: The without_labels of this CommonDeploymentProperties.
        """
        self._with_labels = with_labels
        self._without_labels = without_labels

    @property
    def with_labels(self) -> Dict[str, Any]:
        """Gets the with_labels of this CommonDeploymentProperties.

        Deployments should contain at least one of the specified label values. `null` matches any
        label value and effectively checks the label existence.

        :return: The with_labels of this CommonDeploymentProperties.
        """
        return self._with_labels

    @with_labels.setter
    def with_labels(self, with_labels: Dict[str, Any]):
        """Sets the with_labels of this CommonDeploymentProperties.

        Deployments should contain at least one of the specified label values. `null` matches any
        label value and effectively checks the label existence.

        :param with_labels: The with_labels of this CommonDeploymentProperties.
        """
        self._with_labels = with_labels

    @property
    def without_labels(self) -> Dict[str, Any]:
        """Gets the without_labels of this CommonDeploymentProperties.

        Deployments may not contain the specified label values. `null` matches any label value and
        effectively ensures that the label does not exist.

        :return: The without_labels of this CommonDeploymentProperties.
        """
        return self._without_labels

    @without_labels.setter
    def without_labels(self, without_labels: Dict[str, Any]):
        """Sets the without_labels of this CommonDeploymentProperties.

        Deployments may not contain the specified label values. `null` matches any label value and
        effectively ensures that the label does not exist.

        :param without_labels: The without_labels of this CommonDeploymentProperties.
        """
        self._without_labels = without_labels
