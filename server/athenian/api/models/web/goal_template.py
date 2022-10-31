from typing import Any, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_metric_id import JIRAMetricID
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID
from athenian.api.models.web.release_metric_id import ReleaseMetricID


class GoalTemplateCommon(Model, sealed=False):
    """A template to generate a goal - common properties used in several models."""

    name: str
    metric: str
    metric_params: Optional[dict[str, Any]]
    repositories: Optional[list[str]]

    def validate_name(self, name: str) -> str:
        """Sets the name of this GoalTemplate."""
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")
        if len(name) < 1:
            raise ValueError(
                "Invalid value for `name`, length must be greater than or equal to `1`",
            )

        return name

    def validate_metric(self, metric: str) -> str:
        """Sets the metric of this GoalTemplate."""
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")
        if metric not in JIRAMetricID | PullRequestMetricID | ReleaseMetricID:
            raise ValueError(f"Invalid value for `metric` {metric}")

        return metric


class GoalTemplate(GoalTemplateCommon):
    """A template to generate a goal."""

    id: int
