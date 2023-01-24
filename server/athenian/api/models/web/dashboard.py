from datetime import date
from typing import Optional, Sequence

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_filter import JIRAFilter


class DashboardChartFilters(Model):
    """Collection of filters applied to the chart."""

    repositories: Optional[list[str]]
    environments: Optional[list[str]]
    jira: Optional[JIRAFilter]


class DashboardChartGroupBy(Model):
    """The group by configured for the dashboard chart."""

    repositories: Optional[list[str]]
    teams: Optional[list[int]]
    jira_issue_types: Optional[list[str]]
    jira_labels: Optional[list[str]]
    jira_priorities: Optional[list[str]]


class _UpdatableChartInfo(Model):
    date_from: Optional[date]
    date_to: Optional[date]
    name: str
    time_interval: Optional[str]
    filters: Optional[DashboardChartFilters]
    group_by: Optional[DashboardChartGroupBy]


class _CommonChartInfo(_UpdatableChartInfo):
    description: str
    metric: str


class DashboardChart(_CommonChartInfo):
    """A chart showing metric values in a dashboard."""

    id: int


class TeamDashboard(Model):
    """A dashboard displaying metrics about a team."""

    charts: Sequence[DashboardChart]
    id: int
    team: int


class DashboardChartCreateRequest(_CommonChartInfo):
    """Dashboard chart creation request."""

    position: Optional[int]


class DashboardChartUpdateRequest(_UpdatableChartInfo):
    """The information to update an existing dashboard chart."""


class _DashboardUpdateChart(Model):
    id: int


class DashboardUpdateRequest(Model):
    """Dashboard update request."""

    charts: list[_DashboardUpdateChart]
