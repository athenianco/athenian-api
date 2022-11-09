from datetime import date
from typing import Any, Optional

from athenian.api.models.web.base_model_ import Model


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
