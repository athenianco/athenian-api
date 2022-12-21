from datetime import date
from typing import Optional, Sequence

from athenian.api.models.web.base_model_ import Model


class DashboardChart(Model):
    """A chart showing metric values in a dashboard."""

    date_from: Optional[date]
    date_to: Optional[date]
    description: str
    id: int
    metric: str
    name: str
    time_interval: Optional[str]


class TeamDashboard(Model):
    """A dashboard displaying metrics about a team."""

    charts: Sequence[DashboardChart]
    id: int
    team: int
