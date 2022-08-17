from __future__ import annotations

from typing import Optional

from athenian.api.db import Row
from athenian.api.models.state.models import GoalTemplate as DBGoalTemplate
from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_metric_id import JIRAMetricID
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID
from athenian.api.models.web.release_metric_id import ReleaseMetricID

GoalTemplateMetricID = JIRAMetricID | PullRequestMetricID | ReleaseMetricID


class BaseGoalTemplateModel(Model):
    """Base informations common to models related to goal template."""

    attribute_types = {
        "name": str,
        "repositories": Optional[list[str]],
    }

    def __init__(self, name: str = None, repositories: Optional[list[str]] = None):
        """BaseGoalTemplateModel - init the object."""
        self._name = name
        self._repositories = repositories

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
    def repositories(self) -> Optional[list[str]]:
        """Gets the repositories for this GoalTemplate."""
        return self._repositories

    @repositories.setter
    def repositories(self, repositories: Optional[list[str]]):
        """Sets the repositories of this GoalTemplate."""
        self._repositories = repositories


class GoalTemplate(BaseGoalTemplateModel):
    """A template to generate a goal."""

    attribute_types = {
        **BaseGoalTemplateModel.attribute_types,
        "id": int,
        "metric": GoalTemplateMetricID,
    }

    def __init__(
        self,
        id: int = None,
        name: str = None,
        metric: GoalTemplateMetricID = None,
        repositories: Optional[list[str]] = None,
    ):
        """GoalTemplate - a model defined in OpenAPI

        :param id: The id of this GoalTemplate.
        :param name: The name of this GoalTemplate.
        :param metric: The metric of this GoalTemplate.
        :param repositories: The repositories of this GoalTemplate.
        """
        super().__init__(name=name, repositories=repositories)
        self._id = id
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
    def metric(self) -> GoalTemplateMetricID:
        """Gets the metric of this GoalTemplate."""
        return self._metric

    @metric.setter
    def metric(self, metric: GoalTemplateMetricID) -> None:
        """Sets the metric of this GoalTemplate."""
        if metric is None:
            raise ValueError("Invalid value for `metric`, must not be `None`")
        self._metric = metric

    @classmethod
    def from_db_row(cls, row: Row, *, repositories: Optional[list[str]]) -> GoalTemplate:
        """Build a GoalTemplate starting from the db row."""
        fields = (
            ("id", DBGoalTemplate.id),
            ("name", DBGoalTemplate.name),
            ("metric", DBGoalTemplate.metric),
        )
        kwargs = {field_name: row[col.name] for field_name, col in fields}
        kwargs["repositories"] = repositories
        return cls(**kwargs)
