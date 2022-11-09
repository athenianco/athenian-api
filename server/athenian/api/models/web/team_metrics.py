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

    def params_for_team(self, team_id: int) -> dict:
        """Return the parameters to use for the given team."""
        params = self.metric_params.copy() if self.metric_params else {}
        if self.teams_metric_params:
            try:
                team_params = self.teams_metric_params[team_id]
            except KeyError:
                pass
            else:
                params |= team_params
        return params


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
