from datetime import datetime
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.deployed_component import DeployedComponent
from athenian.api.models.web.deployment_conclusion import DeploymentConclusion
from athenian.api.models.web.invalid_request_error import InvalidRequestError
from athenian.api.response import ResponseError


class DeploymentNotificationUnsealed(Model):
    """Push message about a deployment. We remove unresolved components after 24h."""

    openapi_types = {
        "components": List[DeployedComponent],
        "environment": str,
        "name": Optional[str],
        "url": Optional[str],
        "date_started": datetime,
        "date_finished": datetime,
        "conclusion": str,
        "labels": dict,
    }

    attribute_map = {
        "components": "components",
        "environment": "environment",
        "name": "name",
        "url": "url",
        "date_started": "date_started",
        "date_finished": "date_finished",
        "conclusion": "conclusion",
        "labels": "labels",
    }

    __enable_slots__ = False

    def __init__(
        self,
        components: Optional[List[DeployedComponent]] = None,
        environment: Optional[str] = None,
        name: Optional[str] = None,
        url: Optional[str] = None,
        date_started: Optional[datetime] = None,
        date_finished: Optional[datetime] = None,
        conclusion: Optional[DeploymentConclusion] = None,
        labels: Optional[dict] = None,
    ):
        """DeploymentNotification - a model defined in OpenAPI

        :param components: The components of this DeploymentNotification.
        :param environment: The environment of this DeploymentNotification.
        :param name: The name of this DeploymentNotification.
        :param url: The url of this DeploymentNotification.
        :param date_started: The date_started of this DeploymentNotification.
        :param date_finished: The date_finished of this DeploymentNotification.
        :param conclusion: The conclusion of this DeploymentNotification.
        :param labels: The labels of this DeploymentNotification.
        """
        self._components = components
        self._environment = environment
        self._name = name
        self._url = url
        self._date_started = date_started
        self._date_finished = date_finished
        self._conclusion = conclusion
        self._labels = labels

    def validate_timestamps(self) -> None:
        """Post-check the request data."""
        if self.date_finished < self.date_started:
            raise ResponseError(InvalidRequestError(
                "`date_finished` must be later than `date_started`"))

    @property
    def components(self) -> List[DeployedComponent]:
        """Gets the components of this DeploymentNotification.

        List of deployed software version. Each item identifies a Git reference in a repository,
        either a tag or a commit hash.

        :return: The components of this DeploymentNotification.
        """
        return self._components

    @components.setter
    def components(self, components: List[DeployedComponent]):
        """Sets the components of this DeploymentNotification.

        List of deployed software version. Each item identifies a Git reference in a repository,
        either a tag or a commit hash.

        :param components: The components of this DeploymentNotification.
        """
        if components is None:
            raise ValueError("Invalid value for `components`, must not be `None`")

        self._components = components

    @property
    def environment(self) -> str:
        """Gets the environment of this DeploymentNotification.

        Name of the environment where the deployment happened.

        :return: The environment of this DeploymentNotification.
        """
        return self._environment

    @environment.setter
    def environment(self, environment: str):
        """Sets the environment of this DeploymentNotification.

        Name of the environment where the deployment happened.

        :param environment: The environment of this DeploymentNotification.
        """
        if environment is None:
            raise ValueError("Invalid value for `environment`, must not be `None`")
        if not environment:
            raise ValueError(
                "Invalid value for `environment`, length must be greater than or equal to `1`")

        self._environment = environment

    @property
    def name(self) -> Optional[str]:
        """Gets the name of this DeploymentNotification.

        Name of the deployment. If is not specified, we generate our own by the template
        `<environment shortcut>-<date>-<hash of the components>`.

        :return: The name of this DeploymentNotification.
        """
        return self._name

    @name.setter
    def name(self, name: Optional[str]):
        """Sets the name of this DeploymentNotification.

        Name of the deployment. If is not specified, we generate our own by the template
        `<environment shortcut>-<date>-<hash of the components>`.

        :param name: The name of this DeploymentNotification.
        """
        if name is not None and not name:
            raise ValueError("`name` must be either null or at least 1 character long")
        self._name = name

    @property
    def url(self) -> Optional[str]:
        """Gets the url of this DeploymentNotification.

        URL pointing at the internal details of the deployment.

        :return: The url of this DeploymentNotification.
        """
        return self._url

    @url.setter
    def url(self, url: Optional[str]):
        """Sets the url of this DeploymentNotification.

        URL pointing at the internal details of the deployment.

        :param url: The url of this DeploymentNotification.
        """
        self._url = url

    @property
    def date_started(self) -> datetime:
        """Gets the date_started of this DeploymentNotification.

        Timestamp of when the deployment procedure launched.

        :return: The date_started of this DeploymentNotification.
        """
        return self._date_started

    @date_started.setter
    def date_started(self, date_started: datetime):
        """Sets the date_started of this DeploymentNotification.

        Timestamp of when the deployment procedure launched.

        :param date_started: The date_started of this DeploymentNotification.
        """
        if date_started is None:
            raise ValueError("`date_started` may not be null")
        self._date_started = date_started

    @property
    def date_finished(self) -> datetime:
        """Gets the date_finished of this DeploymentNotification.

        Timestamp of when the deployment procedure completed.

        :return: The date_finished of this DeploymentNotification.
        """
        return self._date_finished

    @date_finished.setter
    def date_finished(self, date_finished: datetime):
        """Sets the date_finished of this DeploymentNotification.

        Timestamp of when the deployment procedure completed.

        :param date_finished: The date_finished of this DeploymentNotification.
        """
        if date_finished is None:
            raise ValueError("`date_finished` may not be null")
        self._date_finished = date_finished

    @property
    def conclusion(self) -> str:
        """Gets the conclusion of this DeploymentNotification.

        :return: The conclusion of this DeploymentNotification.
        """
        return self._conclusion

    @conclusion.setter
    def conclusion(self, conclusion: str):
        """Sets the conclusion of this DeploymentNotification.

        :param conclusion: The conclusion of this DeploymentNotification.
        """
        if conclusion not in DeploymentConclusion:
            raise ValueError(
                f"Invalid value for `conclusion`, must be one of {list(DeploymentConclusion)}")
        self._conclusion = conclusion

    @property
    def labels(self) -> Optional[dict]:
        """Gets the labels of this DeploymentNotification.

        Arbitrary key-value metadata that associates with the deployment.

        :return: The labels of this DeploymentNotification.
        """
        return self._labels

    @labels.setter
    def labels(self, labels: Optional[dict]):
        """Sets the labels of this DeploymentNotification.

        Arbitrary key-value metadata that associates with the deployment.

        :param labels: The labels of this DeploymentNotification.
        """
        self._labels = labels


class DeploymentNotification(DeploymentNotificationUnsealed):
    """Push message about a deployment. We remove unresolved components after 24h."""

    __enable_slots__ = True
