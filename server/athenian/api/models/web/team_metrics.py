from datetime import date
from typing import Any, Optional, Sequence

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.goal import MetricValue
from athenian.api.typing_utils import VerbatimOptional


class TeamDigest(Model):
    """Basic information about a team."""

    id: int
    name: str


class _TeamMetricParams(Model):
    team: int
    metric_params: dict[str, Any]


class TeamMetricWithParams(Model):
    """A metric with optional metric parameters and metric parameters overrides per team."""

    name: str
    metric_params: Optional[dict[str, Any]]
    teams_metric_params: Optional[list[_TeamMetricParams]]


class TeamMetricsRequest(Model):
    """Request the metrics values for a team."""

    team: int
    valid_from: date
    expires_at: date
    repositories: Optional[list[str]]
    jira_projects: Optional[list[str]]
    jira_priorities: Optional[list[str]]
    jira_issue_types: Optional[list[str]]
    metrics_with_params: list[TeamMetricWithParams]


class TeamMetricValueNode(Model):
    """A node of the tree returned in TeamMetricResponse, with team its and metric value."""

    team: TeamDigest
    children: Sequence["TeamMetricValueNode"]
    # see comment for GoalValue.current field about string type reference
    value: VerbatimOptional["MetricValue"]


class TeamMetricResponseElement(Model):
    """Computed metric values for the team and its child teams."""

    metric: str
    value: TeamMetricValueNode
