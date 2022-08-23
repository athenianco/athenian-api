from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_metric_id import JIRAMetricID
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID
from athenian.api.models.web.release_metric_id import ReleaseMetricID


class GoalTemplate(Model):
    """A template to generate a goal."""

    attribute_types = {
        "id": int,
        "name": str,
        "metric": str,
    }

    def __init__(
        self,
        id: Optional[int] = None,
        name: Optional[str] = None,
        metric: Optional[str] = None,
    ):
        """GoalTemplate - a model defined in OpenAPI

        :param id: The id of this GoalTemplate.
        :param name: The name of this GoalTemplate.
        :param metric: The metric of this GoalTemplate.
        """
        self._id = id
        self._name = name
        self._metric = metric

    @property
    def id(self) -> int:
        """Gets the id of this GoalTemplate."""
        return self._id

    @id.setter
    def id(self, id: int) -> None:
        """Sets the id of this GoalTemplate."""
        if id is None:
            raise ValueError("Invalid value for `id`, must not be `None`")

        self._id = id

    @property
    def name(self) -> str:
        """Gets the name of this GoalTemplate."""
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Sets the name of this GoalTemplate."""
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")
        if name is not None and len(name) < 1:
            raise ValueError(
                "Invalid value for `name`, length must be greater than or equal to `1`",
            )

        self._name = name

    @property
    def metric(self) -> str:
        """Gets the metric of this GoalTemplate."""
        return self._metric

    @metric.setter
    def metric(self, metric: str) -> None:
        """Sets the metric of this GoalTemplate."""
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")
        self._validate_metric(metric)
        self._metric = metric

    @classmethod
    def _validate_metric(cls, metric: str) -> None:
        if metric not in JIRAMetricID | PullRequestMetricID | ReleaseMetricID:
            raise ValueError(f"Invalid value for `metric` {metric}")
