from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.deployment_conclusion import DeploymentConclusion


class FilteredEnvironment(Model):
    """Details about a deployment environment."""

    openapi_types = {
        "name": str,
        "deployments_count": int,
        "last_conclusion": str,
    }

    attribute_map = {
        "name": "name",
        "deployments_count": "deployments_count",
        "last_conclusion": "last_conclusion",
    }

    def __init__(
        self,
        name: Optional[str] = None,
        deployments_count: Optional[int] = None,
        last_conclusion: Optional[str] = None,
    ):
        """FilteredEnvironment - a model defined in OpenAPI

        :param name: The name of this FilteredEnvironment.
        :param deployments_count: The deployments_count of this FilteredEnvironment.
        :param last_conclusion: The last_conclusion of this FilteredEnvironment.
        """
        self._name = name
        self._deployments_count = deployments_count
        self._last_conclusion = last_conclusion

    @property
    def name(self) -> str:
        """Gets the name of this FilteredEnvironment.

        Name of the environment.

        :return: The name of this FilteredEnvironment.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this FilteredEnvironment.

        Name of the environment.

        :param name: The name of this FilteredEnvironment.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")

        self._name = name

    @property
    def deployments_count(self) -> int:
        """Gets the deployments_count of this FilteredEnvironment.

        Number of deployments (successful or not) in the specified time interval.

        :return: The deployments_count of this FilteredEnvironment.
        """
        return self._deployments_count

    @deployments_count.setter
    def deployments_count(self, deployments_count: int):
        """Sets the deployments_count of this FilteredEnvironment.

        Number of deployments (successful or not) in the specified time interval.

        :param deployments_count: The deployments_count of this FilteredEnvironment.
        """
        if deployments_count is None:
            raise ValueError(
                "Invalid value for `deployments_count`, must not be `None`")
        if deployments_count is not None and deployments_count < 1:
            raise ValueError(
                "Invalid value for `deployments_count`, must be a value greater than or equal "
                "to `1`")

        self._deployments_count = deployments_count

    @property
    def last_conclusion(self) -> str:
        """Gets the last_conclusion of this FilteredEnvironment.

        The conclusion of the most recent deployment before `time_to`.

        :return: The last_conclusion of this FilteredEnvironment.
        """
        return self._last_conclusion

    @last_conclusion.setter
    def last_conclusion(self, last_conclusion: str):
        """Sets the last_conclusion of this FilteredEnvironment.

        The conclusion of the most recent deployment before `time_to`.

        :param last_conclusion: The last_conclusion of this FilteredEnvironment.
        """
        if last_conclusion is None:
            raise ValueError("Invalid value for `last_conclusion`, must not be `None`")

        if last_conclusion not in DeploymentConclusion:
            raise ValueError(f"Invalid value for `last_conclusion`, must be one of "
                             f"{list(DeploymentConclusion)}")

        self._last_conclusion = last_conclusion
