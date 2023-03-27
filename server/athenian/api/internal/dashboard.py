from collections.abc import Container
from datetime import datetime, timezone
from enum import Enum
from http import HTTPStatus
import logging
from typing import Any, Sequence

import sqlalchemy as sa
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.db import (
    Connection,
    DatabaseLike,
    Row,
    conn_in_transaction,
    dialect_specific_insert,
    is_postgresql,
)
from athenian.api.internal.datetime_utils import (
    closed_dates_interval_to_datetimes,
    datetimes_to_closed_dates_interval,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.repos import parse_db_repositories
from athenian.api.models.state.models import (
    DashboardChart,
    DashboardChartGroupBy,
    Team,
    TeamDashboard,
)
from athenian.api.models.web import (
    DashboardChart as WebDashboardChart,
    DashboardChartCreateRequest,
    DashboardChartFilters,
    DashboardChartGroupBy as WebDashboardChartGroupBy,
    DashboardChartUpdateRequest,
    GenericError,
    JIRAFilter,
    TeamDashboard as WebTeamDashboard,
)
from athenian.api.response import ResponseError
from athenian.api.tracing import sentry_span


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


class _ChartGroupByColAlias(Enum):
    TEAMS = f"group_by_{DashboardChartGroupBy.teams.name}"
    REPOSITORIES = f"group_by_{DashboardChartGroupBy.repositories.name}"
    JIRA_ISSUE_TYPES = f"group_by_{DashboardChartGroupBy.jira_issue_types.name}"
    JIRA_LABELS = f"group_by_{DashboardChartGroupBy.jira_labels.name}"
    JIRA_PRIORITIES = f"group_by_{DashboardChartGroupBy.jira_priorities.name}"


async def get_dashboard_charts(dashboard_id: int, sdb_conn: DatabaseLike) -> Sequence[Row]:
    """Fetch the rows for the charts of a dashboard from DB.

    Charts are returned with the proper order they have in the dashboard.
    Each row will include the columns from the joined _ChartGroupByColAlias table.
    """
    chart_group_by_cols = (
        DashboardChartGroupBy.teams.label(_ChartGroupByColAlias.TEAMS.value),
        DashboardChartGroupBy.repositories.label(_ChartGroupByColAlias.REPOSITORIES.value),
        DashboardChartGroupBy.jira_issue_types.label(_ChartGroupByColAlias.JIRA_ISSUE_TYPES.value),
        DashboardChartGroupBy.jira_labels.label(_ChartGroupByColAlias.JIRA_LABELS.value),
        DashboardChartGroupBy.jira_priorities.label(_ChartGroupByColAlias.JIRA_PRIORITIES.value),
    )
    stmt = (
        sa.select(DashboardChart, *chart_group_by_cols)
        .select_from(sa.outerjoin(DashboardChart, DashboardChartGroupBy))
        .where(DashboardChart.dashboard_id == dashboard_id)
        .order_by(DashboardChart.position)
    )
    return await sdb_conn.fetch_all(stmt)


async def create_dashboard_chart(
    dashboard_id: int,
    req: DashboardChartCreateRequest,
    extra_values: dict[InstrumentedAttribute, Any],
    group_by_values: dict[InstrumentedAttribute, Any],
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

    values = {
        **_build_new_chart_row_values(req, now),
        **extra_values,
        DashboardChart.dashboard_id: dashboard_id,
        DashboardChart.position: position,
    }

    insert_stmt = sa.insert(DashboardChart).values(values)
    new_chart_id = await sdb_conn.execute(insert_stmt)
    if group_by_values:
        await _upsert_chart_group_by(new_chart_id, group_by_values, now, sdb_conn)
    return new_chart_id


async def delete_dashboard_chart(dashboard_id: int, chart_id: int, sdb_conn: DatabaseLike) -> None:
    """Delete a dashboard chart, raise errors if not existing."""
    where = (DashboardChart.dashboard_id == dashboard_id, DashboardChart.id == chart_id)
    # no rowcount is returned with aysncpg
    select_stmt = sa.select(1).where(*where)
    if (await sdb_conn.fetch_val(select_stmt)) is None:
        raise DashboardChartNotFoundError(dashboard_id, chart_id)

    delete_stmt = sa.delete(DashboardChart).where(*where)
    await sdb_conn.execute(delete_stmt)


async def update_dashboard_chart(
    dashboard_id: int,
    chart_id: int,
    req: DashboardChartUpdateRequest,
    extra_values: dict[InstrumentedAttribute, Any],
    group_by_values: dict[InstrumentedAttribute, Any],
    sdb_conn: Connection,
) -> None:
    """Update an existing dashboard chart."""
    assert await conn_in_transaction(sdb_conn)

    now = datetime.now(timezone.utc)

    values = {**_build_update_chart_row_values(req, now), **extra_values}
    update_stmt = (
        sa.update(DashboardChart)
        .where(DashboardChart.id == chart_id, DashboardChart.dashboard_id == dashboard_id)
        .values(values)
    )
    if await is_postgresql(sdb_conn):
        updated = await sdb_conn.fetch_val(update_stmt.returning(DashboardChart.id))
    else:
        updated = await sdb_conn.execute(update_stmt)
    if not updated:
        raise DashboardChartNotFoundError(dashboard_id, chart_id)

    if group_by_values:
        await _upsert_chart_group_by(chart_id, group_by_values, now, sdb_conn)


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
            "Charts in requested ordering does not match existing charts. "
            f"existing: {existing_repr}; requested: {requested_repr}",
        )

    if chart_ids:
        await _reassign_charts_positions(chart_ids, 0, datetime.now(timezone.utc), sdb_conn)


@sentry_span
async def remove_team_refs_from_charts(team: int, account: int, sdb_conn: Connection) -> None:
    """Remove references to the team from account charts.

    Team will be removed from the "teams" group by of every account's chart.
    Charts with the specified team as the only "teams" group by, and with no other group by
    criteria, will be deleted.

    Charts owned by the team are removed by ON DELETE CASCADE so are not handled here.
    """
    assert await conn_in_transaction(sdb_conn)
    log = logging.getLogger(f"{metadata.__package__}.remove_team_refs_from_charts")
    where = [Team.owner_id == account]
    select_from = sa.select(DashboardChartGroupBy).select_from(DashboardChartGroupBy)
    for table in (DashboardChart, TeamDashboard, Team):
        select_from = select_from.join(table)

    if await is_postgresql(sdb_conn):
        # will use jsonb "@>" operator
        where.append(DashboardChartGroupBy.teams.contains(team))
    else:
        # expand teams to a virtual table with json_each, then filter
        teams_table = sa.func.json_each(DashboardChartGroupBy.teams).table_valued("value")
        select_from = select_from.select_from(teams_table)
        where.append(teams_table.c.value == team)

    for_update_of = [DashboardChart, DashboardChartGroupBy]
    select_stmt = select_from.where(*where).with_for_update(of=for_update_of)
    rows = await sdb_conn.fetch_all(select_stmt)

    todelete = set()
    toupdate = set()

    def _chart_to_be_deleted(r: Row) -> bool:
        if len(r[DashboardChartGroupBy.teams.name]) > 1:
            return False
        fields = set(DashboardChartGroupBy.GROUP_BY_FIELDS) - {DashboardChartGroupBy.teams}
        return all(not r[f.name] for f in fields)

    for r in rows:
        if _chart_to_be_deleted(r):
            todelete.add(r[DashboardChartGroupBy.chart_id.name])
        else:
            toupdate.add(r[DashboardChartGroupBy.chart_id.name])

    if todelete:
        # DashboardChartGroupBy will be removed due to CASCADE constraint
        delete_stmt = sa.delete(DashboardChart).where(DashboardChart.id.in_(todelete))
        await sdb_conn.execute(delete_stmt)

    if toupdate:
        now = datetime.now(timezone.utc)

        def _update_stmt(chart_ids: Container[int], teams: Any) -> sa.sql.Update:
            values = {DashboardChartGroupBy.teams: teams, DashboardChartGroupBy.updated_at: now}
            where = DashboardChartGroupBy.chart_id.in_(chart_ids)
            return sa.update(DashboardChartGroupBy).where(where).values(values)

        if await is_postgresql(sdb_conn):
            # remove the team id from the teams jsonb array and make [] => null
            cleared_teams_array = sa.func.jsonb_path_query_array(
                DashboardChartGroupBy.teams, f"$[*] ? (@ != {team})",
            )
            cleared_teams = sa.case((cleared_teams_array == [], None), else_=cleared_teams_array)
            await sdb_conn.execute(_update_stmt(toupdate, cleared_teams))
        else:
            # probably possible with a single query, not efficient but unused in production
            rows_toupdate = [r for r in rows if r[DashboardChartGroupBy.chart_id.name] in toupdate]
            for row_toupdate in rows_toupdate:
                teams = row_toupdate[DashboardChartGroupBy.teams.name]
                chart_id = row_toupdate[DashboardChartGroupBy.chart_id.name]
                cleared_teams_lst = [t for t in teams if t != team] or None
                await sdb_conn.execute(_update_stmt([chart_id], cleared_teams_lst))

    log.info("%d charts deleted", len(todelete))
    log.info("References to team %d removed from %d charts group by", team, len(todelete))


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

    jira_filter_kw = {
        arg: chart[col.name]
        for col, arg in (
            (DashboardChart.jira_issue_types, "issue_types"),
            (DashboardChart.jira_labels, "labels_include"),
            (DashboardChart.jira_priorities, "priorities"),
            (DashboardChart.jira_projects, "projects"),
        )
        if chart[col.name] is not None
    }
    if jira_filter_kw:
        filters_kw["jira"] = JIRAFilter(**jira_filter_kw)

    group_by_kw = {
        arg: chart[col_name]
        for col_name, arg in (
            (_ChartGroupByColAlias.TEAMS.value, "teams"),
            (_ChartGroupByColAlias.JIRA_ISSUE_TYPES.value, "jira_issue_types"),
            (_ChartGroupByColAlias.JIRA_LABELS.value, "jira_labels"),
            (_ChartGroupByColAlias.JIRA_PRIORITIES.value, "jira_priorities"),
        )
        if chart[col_name] is not None
    }
    group_by_repos = parse_db_repositories(chart[_ChartGroupByColAlias.REPOSITORIES.value])
    if group_by_repos is not None:
        group_by_kw["repositories"] = [
            str(r) for r in prefixer.dereference_repositories(group_by_repos)
        ]

    return WebDashboardChart(
        description=chart[DashboardChart.description.name],
        id=chart[DashboardChart.id.name],
        metric=chart[DashboardChart.metric.name],
        name=chart[DashboardChart.name.name],
        date_from=date_from,
        date_to=date_to,
        time_interval=chart[DashboardChart.time_interval.name],
        filters=DashboardChartFilters(**filters_kw) if filters_kw else None,
        group_by=WebDashboardChartGroupBy(**group_by_kw) if group_by_kw else None,
    )


def _build_new_chart_row_values(chart: DashboardChartCreateRequest, now: datetime) -> dict:
    values = _build_update_chart_row_values(chart, now)
    return {
        **values,
        DashboardChart.metric: chart.metric,
        DashboardChart.description: chart.description,
        DashboardChart.created_at: now,
    }


def _build_update_chart_row_values(
    chart: DashboardChartUpdateRequest | DashboardChartCreateRequest,
    now: datetime,
) -> dict:
    if chart.date_from is None or chart.date_to is None:
        time_from = time_to = None
    else:
        time_from, time_to = closed_dates_interval_to_datetimes(chart.date_from, chart.date_to)
    return {
        DashboardChart.name: chart.name,
        DashboardChart.time_to: time_to,
        DashboardChart.time_from: time_from,
        DashboardChart.time_interval: chart.time_interval,
        DashboardChart.updated_at: now,
    }


async def _upsert_chart_group_by(
    chart_id: int,
    values: dict[InstrumentedAttribute, Any],
    now: datetime,
    sdb_conn: Connection,
) -> None:
    all_values = {
        **values,
        DashboardChartGroupBy.chart_id: chart_id,
        DashboardChartGroupBy.created_at: now,
        DashboardChartGroupBy.updated_at: now,
    }
    insert = await dialect_specific_insert(sdb_conn)
    insert_stmt = insert(DashboardChartGroupBy)

    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=DashboardChartGroupBy.__table__.primary_key.columns,
        set_={
            **{col.name: getattr(insert_stmt.excluded, col.name) for col in values},
            DashboardChartGroupBy.updated_at: now,
        },
    ).values(all_values)
    await sdb_conn.execute(upsert_stmt)


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

    def __init__(self, dashboard_id: int, chart_id: int):
        """Init the DashboardChartNotFoundError."""
        wrapped_error = GenericError(
            type="/errors/dashboards/DashboardChartNotFoundError",
            status=HTTPStatus.NOT_FOUND,
            detail=f"Chart {chart_id} not found in dashboard {dashboard_id} or access denied",
            title="Chart not found",
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
