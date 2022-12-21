from http import HTTPStatus
from typing import Sequence

import sqlalchemy as sa

from athenian.api.db import Connection, DatabaseLike, Row, is_postgresql
from athenian.api.internal.datetime_utils import datetimes_to_closed_dates_interval
from athenian.api.models.state.models import DashboardChart, TeamDashboard
from athenian.api.models.web import (
    DashboardChart as WebDashboardChart,
    GenericError,
    TeamDashboard as WebTeamDashboard,
)
from athenian.api.response import ResponseError


async def get_dashboard(dashboard_id: int, sdb_conn: DatabaseLike) -> Row:
    """Fetch the row for a dashboard from DB."""
    stmt = sa.select(TeamDashboard).where(TeamDashboard.id == dashboard_id)
    row = await sdb_conn.fetch_one(stmt)
    if row is None:
        raise TeamDashboardNotFoundError(dashboard_id)
    return row


async def get_team_default_dashboard(team_id, sdb_conn: Connection) -> Row:
    """Fetch the default dashboard for a team."""
    stmt = sa.select(TeamDashboard).where(TeamDashboard.team_id == team_id)
    rows = await sdb_conn.fetch_all(stmt)
    if len(rows) == 1:
        return rows[0]
    if len(rows) > 1:
        raise MultipleTeamDashboardsError(team_id)

    async with sdb_conn.transaction():
        # acquire the lock to avoid multiple dashboards concurrent creation
        # this just happens once per team
        if await is_postgresql(sdb_conn):
            await sdb_conn.execute("LOCK TABLE team_dashboards IN SHARE ROW EXCLUSIVE MODE")
        rows = await sdb_conn.fetch_all(stmt)
        if len(rows) == 1:
            return rows[0]
        if len(rows) > 1:
            raise MultipleTeamDashboardsError(team_id)

        values = TeamDashboard(team_id=team_id).create_defaults().explode()
        insert_stmt = sa.insert(TeamDashboard).values(values)
        await sdb_conn.execute(insert_stmt)
    return await get_team_default_dashboard(team_id, sdb_conn)


async def get_dashboard_charts(dashboard_id: int, sdb_conn: DatabaseLike) -> Sequence[Row]:
    """Fetch the rows for the charts of a dashboard from DB.

    Charts are returned with the proper order they have in the dashboard.

    """
    stmt = (
        sa.select(DashboardChart)
        .where(DashboardChart.dashboard_id == dashboard_id)
        .order_by(DashboardChart.position)
    )
    return await sdb_conn.fetch_all(stmt)


def build_dashboard_web_model(dashboard: Row, charts: Sequence[Row]) -> TeamDashboard:
    """Build the web model for a dashboard given the dashboard and charts DB rows."""
    return WebTeamDashboard(
        id=dashboard[TeamDashboard.id.name],
        team=dashboard[TeamDashboard.team_id.name],
        charts=[_build_chart_web_model(chart) for chart in charts],
    )


def _build_chart_web_model(chart: Row) -> WebDashboardChart:
    time_from = chart[DashboardChart.time_from.name]
    time_to = chart[DashboardChart.time_to.name]

    if time_from is None or time_to is None:
        date_from = date_to = None
    else:
        date_from, date_to = datetimes_to_closed_dates_interval(time_from, time_to)

    return WebDashboardChart(
        description=chart[DashboardChart.description.name],
        id=chart[DashboardChart.id.name],
        metric=chart[DashboardChart.metric.name],
        name=chart[DashboardChart.name.name],
        date_from=date_from,
        date_to=date_to,
        time_interval=chart[DashboardChart.time_interval.name],
    )


class TeamDashboardNotFoundError(ResponseError):
    """A team dashboard was not found."""

    def __init__(self, dashboard_id: int):
        """Init the TeamDashboardNotFoundError."""
        wrapped_error = GenericError(
            type="/errors/dashboards/TeamDashboardNotFoundError",
            status=HTTPStatus.NOT_FOUND,
            detail=f"Team Dashboard {dashboard_id} not found or access denied",
            title="Team Dashboard not found",
        )
        super().__init__(wrapped_error)


class MultipleTeamDashboardsError(ResponseError):
    """A team has multiple dashboards."""

    def __init__(self, team_id: int):
        """Init the MultipleTeamDashboardsError."""
        wrapped_error = GenericError(
            type="/errors/dashboards/MultipleTeamDashboardsError",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Team {team_id} has multiple dashboards",
            title="Multiple team dashboards",
        )
        super().__init__(wrapped_error)
