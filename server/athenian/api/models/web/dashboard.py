from datetime import date
from typing import Optional, Sequence

from athenian.api.models.web.base_model_ import Model


class _CommonChartInfo(Model):
    date_from: Optional[date]
    date_to: Optional[date]
    description: str
    metric: str
    name: str
    time_interval: Optional[str]


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


class _DashboardUpdateChart(Model):
    id: int


class DashboardUpdateRequest(Model):
    """Dashboard update request."""

    charts: list[_DashboardUpdateChart]
