from typing import Optional

from athenian.api.models.web.goal_template import BaseGoalTemplateModel, GoalTemplateMetricID


class GoalTemplateCreateRequest(BaseGoalTemplateModel):
    """Goal Template creation request."""

    attribute_types = {
        **BaseGoalTemplateModel.attribute_types,
        "account": int,
        "metric": GoalTemplateMetricID,
    }

    def __init__(
        self,
        account: int = None,
        metric: GoalTemplateMetricID = None,
        name: str = None,
        repositories: Optional[list[str]] = None,
    ):
        """GoalTemplateCreateRequest - a model defined in OpenAPI

        :param account: The account of this GoalTemplateCreateRequest.
        :param metric: The metric of this GoalTemplateCreateRequest.
        :param name: The name of this GoalTemplateCreateRequest.
        :param repositories: The repositories of this GoalTemplateCreateRequest.
        """
        super().__init__(name=name, repositories=repositories)
        self._account = account
        self._metric = metric

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
    def metric(self) -> GoalTemplateMetricID:
        """Gets the metric of this GoalTemplateCreateRequest."""
        return self._metric

    @metric.setter
    def metric(self, metric: GoalTemplateMetricID) -> None:
        """Sets the metric of this GoalTemplateCreateRequest."""
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")

        self._metric = metric
