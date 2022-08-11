from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_metric_id import JIRAMetricID
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID
from athenian.api.models.web.release_metric_id import ReleaseMetricID


class GoalTemplateCreateRequest(Model):
    """Goal Template creation request."""

    attribute_types = {
        "account": int,
        "metric": str,
        "name": str,
    }

    def __init__(
        self,
        account: Optional[int] = None,
        metric: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """GoalTemplateCreateRequest - a model defined in OpenAPI

        :param account: The account of this GoalTemplateCreateRequest.
        :param metric: The metric of this GoalTemplateCreateRequest.
        :param name: The name of this GoalTemplateCreateRequest.
        """
        self._account = account
        self._metric = metric
        self._name = name

    @property
    def account(self) -> int:
        """Gets the account of this GoalTemplateCreateRequest.

        Account identifier. That account will own the created goal template.
        """
        return self._account

    @account.setter
    def account(self, account: int) -> None:
        """Sets the account of this GoalTemplateCreateRequest."""
        if account is None:
            raise ValueError("Invalid value for `account`, must not be `None`")

        self._account = account

    @property
    def metric(self) -> str:
        """Gets the metric of this GoalTemplateCreateRequest."""
        return self._metric

    @metric.setter
    def metric(self, metric: str) -> None:
        """Sets the metric of this GoalTemplateCreateRequest."""
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")
        if metric not in PullRequestMetricID | ReleaseMetricID | JIRAMetricID:
            raise ValueError(f"Unsupported value for `metric`: {metric}")

        self._metric = metric

    @property
    def name(self) -> str:
        """Gets the name of this GoalTemplateCreateRequest.

        Name of the template.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Sets the name of this GoalTemplateCreateRequest."""
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")
        if name is not None and len(name) < 1:
            raise ValueError(
                "Invalid value for `name`, length must be greater than or equal to `1`",
            )

        self._name = name
