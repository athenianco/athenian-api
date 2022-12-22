from datetime import datetime, timezone
from http import HTTPStatus
from typing import Sequence

import sqlalchemy as sa

from athenian.api.db import Connection, DatabaseLike, Row, conn_in_transaction, is_postgresql
from athenian.api.internal.datetime_utils import (
    closed_dates_interval_to_datetimes,
    datetimes_to_closed_dates_interval,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.repos import parse_db_repositories
from athenian.api.models.state.models import DashboardChart, TeamDashboard
from athenian.api.models.web import (
    DashboardChart as WebDashboardChart,
    DashboardChartCreateRequest,
    DashboardChartFilters,
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


async def create_dashboard_chart(
    dashboard_id: int,
    req: DashboardChartCreateRequest,
    extra_values: dict,
    sdb_conn: Connection,
) -> int:
    """Create a new dashboard chart and return its ID."""
    assert await conn_in_transaction(sdb_conn)
    # lock existing chart rows, to ensure the position of the new chart is correct and unique
    existing_stmt = (
        sa.select(DashboardChart.id, DashboardChart.position)
        .where(DashboardChart.dashboard_id == dashboard_id)
        .order_by(DashboardChart.position)
        .with_for_update()
    )
    existing_rows = await sdb_conn.fetch_all(existing_stmt)

    now = datetime.now(timezone.utc)

    if not existing_rows:
        # first chart of the dashboard gets position 0 regardless of user request
        position = 0
    else:
        if req.position is None or req.position > len(existing_rows) - 1:
            # by default new chart is appended, position it's max + 1
            position = existing_rows[-1][DashboardChart.position.name] + 1
        else:
            # if there's a requested position new chart will get the position of the
            # chart at that index; following charts will shift by 1
            position = existing_rows[req.position][DashboardChart.position.name]
            ids_to_update = [r[DashboardChart.id.name] for r in existing_rows[req.position :]]
            await _reassign_charts_positions(ids_to_update, position + 1, now, sdb_conn)

    values = _build_chart_row_values(req, now)
    values.update(extra_values)
    values.update({DashboardChart.dashboard_id: dashboard_id, DashboardChart.position: position})
    insert_stmt = sa.insert(DashboardChart).values(values)
    return await sdb_conn.execute(insert_stmt)


async def delete_dashboard_chart(dashboard_id: int, chart_id: int, sdb_conn: DatabaseLike) -> None:
    """Delete a dashboard chart, raise errors if not existing."""
    where = (DashboardChart.dashboard_id == dashboard_id, DashboardChart.id == chart_id)
    # no rowcount is returned with aysncpg
    select_stmt = sa.select([1]).where(*where)
    if (await sdb_conn.fetch_val(select_stmt)) is None:
        raise DashboardChartNotFoundError(chart_id)

    delete_stmt = sa.delete(DashboardChart).where(*where)
    await sdb_conn.execute(delete_stmt)


async def reorder_dashboard_charts(
    dashboard_id: int,
    chart_ids: Sequence[int],
    sdb_conn: Connection,
) -> None:
    """Apply the given order to the existing chart dashboards.

    `chart_ids` are the identifiers of all the existing charts in the desired.
    """
    existing_stmt = (
        sa.select(DashboardChart.id)
        .where(DashboardChart.dashboard_id == dashboard_id)
        .with_for_update()
    )
    existing_rows = await sdb_conn.fetch_all(existing_stmt)
    existing_ids = [r[0] for r in existing_rows]

    if sorted(existing_ids) != sorted(chart_ids):
        existing_repr = ",".join(sorted(map(str, existing_ids)))
        requested_repr = ",".join(sorted(map(str, chart_ids)))
        raise InvalidDashboardChartOrder(
            dashboard_id,
            "Charts in requested ordering does not matc existing charts. "
            f"existing: {existing_repr}; requested: {requested_repr}",
        )

    if chart_ids:
        await _reassign_charts_positions(chart_ids, 0, datetime.now(timezone.utc), sdb_conn)


def build_dashboard_web_model(
    dashboard: Row,
    charts: Sequence[Row],
    prefixer: Prefixer,
) -> TeamDashboard:
    """Build the web model for a dashboard given the dashboard and charts DB rows."""
    return WebTeamDashboard(
        id=dashboard[TeamDashboard.id.name],
        team=dashboard[TeamDashboard.team_id.name],
        charts=[_build_chart_web_model(chart, prefixer) for chart in charts],
    )


def _build_chart_web_model(chart: Row, prefixer: Prefixer) -> WebDashboardChart:
    time_from = chart[DashboardChart.time_from.name]
    time_to = chart[DashboardChart.time_to.name]

    if time_from is None or time_to is None:
        date_from = date_to = None
    else:
        date_from, date_to = datetimes_to_closed_dates_interval(time_from, time_to)

    filters_kw = {}
    if (db_repos := parse_db_repositories(chart[DashboardChart.repositories.name])) is not None:
        filters_kw["repositories"] = [str(r) for r in prefixer.dereference_repositories(db_repos)]
    if chart[DashboardChart.environments.name] is not None:
        filters_kw["environments"] = chart[DashboardChart.environments.name]

    return WebDashboardChart(
        description=chart[DashboardChart.description.name],
        id=chart[DashboardChart.id.name],
        metric=chart[DashboardChart.metric.name],
        name=chart[DashboardChart.name.name],
        date_from=date_from,
        date_to=date_to,
        time_interval=chart[DashboardChart.time_interval.name],
        filters=DashboardChartFilters(**filters_kw) if filters_kw else None,
    )


def _build_chart_row_values(chart: DashboardChartCreateRequest, now: datetime) -> dict:
    if chart.date_from is None or chart.date_to is None:
        time_from = time_to = None
    else:
        time_from, time_to = closed_dates_interval_to_datetimes(chart.date_from, chart.date_to)

    return {
        DashboardChart.metric: chart.metric,
        DashboardChart.name: chart.name,
        DashboardChart.description: chart.description,
        DashboardChart.time_to: time_to,
        DashboardChart.time_from: time_from,
        DashboardChart.time_interval: chart.time_interval,
        DashboardChart.created_at: now,
        DashboardChart.updated_at: now,
    }


async def _reassign_charts_positions(
    chart_ids: Sequence[int],
    first_position: int,
    now: datetime,
    sdb_conn: DatabaseLike,
) -> None:
    # reassign the positions for the chart with given IDs starting from first_position
    new_positions = range(first_position, first_position + len(chart_ids))
    if await is_postgresql(sdb_conn):
        # a single UPDATE FROM VALUES will set all new positions
        # positions will be unique when all updates have been done so constraint is deferred
        await sdb_conn.execute("SET CONSTRAINTS uc_chart_dashboard_id_position DEFERRED")
        update_from = sa.values(
            sa.column("id", sa.Integer),
            sa.column("position", sa.Integer),
            name="newpositions",
            literal_binds=True,
        ).data(list(zip(chart_ids, new_positions)))
        stmt = (
            sa.update(DashboardChart)
            .where(DashboardChart.id == update_from.c.id)
            .values(
                {DashboardChart.position: update_from.c.position, DashboardChart.updated_at: now},
            )
        )
        await sdb_conn.execute(stmt)
    else:
        # for sqlite first set all positions to negative to avoid duplicate, then invert position
        # execute_many doesn't seem to work with morcilla/sqlite
        for id_, pos in zip(chart_ids, new_positions):
            values = {DashboardChart.position: -(pos + 1), DashboardChart.updated_at: now}
            stmt = sa.update(DashboardChart).where(DashboardChart.id == id_).values(values)
            await sdb_conn.execute(stmt)

        global_update_values = {
            DashboardChart.position: -DashboardChart.position - 1,
            DashboardChart.updated_at: now,
        }
        global_update = (
            sa.update(DashboardChart)
            .where(DashboardChart.id.in_(chart_ids))
            .values(global_update_values)
        )
        await sdb_conn.execute(global_update)


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


class DashboardChartNotFoundError(ResponseError):
    """A dashboard chart was not found."""

    def __init__(self, chart_id: int):
        """Init the DashboardChartNotFoundError."""
        wrapped_error = GenericError(
            type="/errors/dashboards/DashboardChartNotFoundError",
            status=HTTPStatus.NOT_FOUND,
            detail=f"Dashboard chart {chart_id} not found or access denied",
            title=" Chart not found",
        )
        super().__init__(wrapped_error)


class InvalidDashboardChartOrder(ResponseError):
    """An invalid order of the dashboard charts was requested."""

    def __init__(self, dashboard_id: int, msg: str):
        """Init the InvalidDashboardChartOrder."""
        wrapped_error = GenericError(
            type="/errors/dashboards/InvalidDashboardChartOrder",
            status=HTTPStatus.BAD_REQUEST,
            detail=f"Invalid charts ordering for dashboard {dashboard_id}: {msg}",
            title=" Invalid dashboard charts order",
        )
        super().__init__(wrapped_error)
