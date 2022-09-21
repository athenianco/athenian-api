from datetime import datetime
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.deployed_component import DeployedComponent
from athenian.api.models.web.deployment_conclusion import DeploymentConclusion
from athenian.api.models.web.invalid_request_error import InvalidRequestError
from athenian.api.response import ResponseError


class DeploymentNotificationUnsealed(Model, sealed=False):
    """Push message about a deployment. We remove unresolved components after 24h."""

    components: List[DeployedComponent]
    environment: str
    name: Optional[str]
    url: Optional[str]
    date_started: datetime
    date_finished: datetime
    conclusion: str
    labels: Optional[dict]

    def validate_timestamps(self) -> None:
        """Post-check the request data."""
        for field in ("date_finished", "date_started"):
            if getattr(self, field).tzinfo is None:
                raise ResponseError(InvalidRequestError(f"`{field}` must include the timezone"))

        if self.date_finished < self.date_started:
            raise ResponseError(
                InvalidRequestError("`date_finished` must be later than `date_started`"),
            )

    def validate_environment(self, environment: str) -> str:
        """Sets the environment of this DeploymentNotification.

        Name of the environment where the deployment happened.

        :param environment: The environment of this DeploymentNotification.
        """
        if environment is None:
            raise ValueError("Invalid value for `environment`, must not be `None`")
        if not environment:
            raise ValueError(
                "Invalid value for `environment`, length must be greater than or equal to `1`",
            )

        return environment

    def validate_name(self, name: Optional[str]) -> Optional[str]:
        """Sets the name of this DeploymentNotification.

        Name of the deployment. If is not specified, we generate our own by the template
        `<environment shortcut>-<date>-<hash of the components>`.

        :param name: The name of this DeploymentNotification.
        """
        if name is not None:
            if not name:
                raise ValueError("`name` must be either null or at least 1 character long")
            if "\n" in name:
                raise ValueError("`name` may not contain new line characters")
            if len(name) > 100:
                raise ValueError("`name` may not be longer than 100 characters")
        return name

    def validate_conclusion(self, conclusion: str | bytes) -> str | bytes:
        """Sets the conclusion of this DeploymentNotification.

        :param conclusion: The conclusion of this DeploymentNotification.
        """
        if isinstance(conclusion, bytes):
            return conclusion  # performance: skip validation
        if conclusion not in DeploymentConclusion:
            raise ValueError(
                f"Invalid value for `conclusion`, must be one of {list(DeploymentConclusion)}",
            )
        return conclusion


class DeploymentNotification(DeploymentNotificationUnsealed):
    """Push message about a deployment. We remove unresolved components after 24h."""
