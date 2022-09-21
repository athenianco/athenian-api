from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.deployment_conclusion import DeploymentConclusion


class FilteredEnvironment(Model):
    """Details about a deployment environment."""

    name: str
    deployments_count: int
    last_conclusion: str
    repositories: list[str]

    def validate_deployments_count(self, deployments_count: int) -> int:
        """Sets the deployments_count of this FilteredEnvironment.

        Number of deployments (successful or not) in the specified time interval.

        :param deployments_count: The deployments_count of this FilteredEnvironment.
        """
        if deployments_count is None:
            raise ValueError("Invalid value for `deployments_count`, must not be `None`")
        if deployments_count is not None and deployments_count < 1:
            raise ValueError(
                "Invalid value for `deployments_count`, must be a value greater than or equal "
                "to `1`",
            )

        return deployments_count

    def validate_last_conclusion(self, last_conclusion: str) -> str:
        """Sets the last_conclusion of this FilteredEnvironment.

        The conclusion of the most recent deployment before `time_to`.

        :param last_conclusion: The last_conclusion of this FilteredEnvironment.
        """
        if last_conclusion is None:
            raise ValueError("Invalid value for `last_conclusion`, must not be `None`")

        if last_conclusion not in DeploymentConclusion:
            raise ValueError(
                "Invalid value for `last_conclusion`, must be one of"
                f" {list(DeploymentConclusion)}",
            )

        return last_conclusion
