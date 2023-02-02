import dataclasses

from aiohttp import web

from athenian.api.async_utils import gather
from athenian.api.auth import disable_default_user
from athenian.api.db import Row
from athenian.api.internal.dashboard import (
    build_dashboard_web_model,
    create_dashboard_chart as create_dashboard_chart_in_db,
    delete_dashboard_chart as delete_dashboard_chart_from_db,
    get_dashboard as get_dashboard_from_db,
    get_dashboard_charts,
    get_team_default_dashboard,
    reorder_dashboard_charts,
    update_dashboard_chart as update_dashboard_chart_in_db,
)
from athenian.api.internal.jira import parse_request_issue_types, parse_request_priorities
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.repos import parse_request_repositories
from athenian.api.internal.team import get_team_from_db
from athenian.api.models.state.models import (
    DashboardChart,
    DashboardChartGroupBy,
    Team,
    TeamDashboard,
)
from athenian.api.models.web import (
    CreatedIdentifier,
    DashboardChartCreateRequest,
    DashboardChartFilters,
    DashboardChartGroupBy as WebDashboardChartGroupBy,
    DashboardChartUpdateRequest,
    DashboardUpdateRequest,
)
from athenian.api.request import AthenianWebRequest, model_from_body
from athenian.api.response import model_response


async def get_dashboard(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
) -> web.Response:
    """Retrieve a team dashboard."""
    dashboard = await _get_request_dashboard(request, team_id, dashboard_id)
    return _dashboard_response(request, dashboard)


@disable_default_user
async def update_dashboard(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
    body: dict,
) -> web.Response:
    """Update an existing team dashboard."""
    dashboard = await _get_request_dashboard(request, team_id, dashboard_id)
    update_request = model_from_body(DashboardUpdateRequest, body)
    chart_ids = [c.id for c in update_request.charts]
    async with request.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            await reorder_dashboard_charts(
                dashboard.row[TeamDashboard.id.name], chart_ids, sdb_conn,
            )
    return _dashboard_response(request, dashboard)


@disable_default_user
async def create_dashboard_chart(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
    body: dict,
) -> web.Response:
    """Create a dashboard chart."""
    dashboard = await _get_request_dashboard(request, team_id, dashboard_id)
    create_request = model_from_body(DashboardChartCreateRequest, body)

    extra_values = await _parse_request_chart_filters(
        create_request.filters, request, dashboard.account,
    )
    group_by_values = await _parse_request_chart_group_by(
        create_request.group_by, request, dashboard.account,
    )

    async with request.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            chart_id = await create_dashboard_chart_in_db(
                dashboard.row[TeamDashboard.id.name],
                create_request,
                extra_values,
                group_by_values,
                sdb_conn,
            )
    return model_response(CreatedIdentifier(id=chart_id))


@disable_default_user
async def delete_dashboard_chart(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
    chart_id: int,
) -> web.Response:
    """Delete an existing dashboard chart."""
    sdb = request.sdb
    dashboard = await _get_request_dashboard(request, team_id, dashboard_id)
    await delete_dashboard_chart_from_db(dashboard.row[TeamDashboard.id.name], chart_id, sdb)
    return web.Response(status=204)


@disable_default_user
async def update_dashboard_chart(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
    chart_id: int,
    body: dict,
) -> web.Response:
    """Update an existing dashboard chart."""
    dashboard = await _get_request_dashboard(request, team_id, dashboard_id)
    update_request = model_from_body(DashboardChartUpdateRequest, body)

    extra_values = await _parse_request_chart_filters(
        update_request.filters, request, dashboard.account,
    )
    for f in DashboardChart.FILTER_FIELDS:
        extra_values.setdefault(f, None)
    group_by_values = await _parse_request_chart_group_by(
        update_request.group_by, request, dashboard.account,
    )
    for f in DashboardChartGroupBy.GROUP_BY_FIELDS:
        group_by_values.setdefault(f, None)

    async with request.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            dashboard_id = dashboard.row[TeamDashboard.id.name]
            await update_dashboard_chart_in_db(
                dashboard_id, chart_id, update_request, extra_values, group_by_values, sdb_conn,
            )

    return model_response(CreatedIdentifier(id=chart_id))


@dataclasses.dataclass
class _RequestDashboard:
    row: Row
    account: int


async def _get_request_dashboard(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
) -> _RequestDashboard:
    sdb = request.sdb
    # this checks team existence and permissions
    team = await get_team_from_db(team_id, account_id=None, user_id=request.uid, sdb_conn=sdb)
    if dashboard_id == 0:
        async with sdb.connection() as sdb_conn:
            dashboard = await get_team_default_dashboard(team_id, sdb_conn)
    else:
        dashboard = await get_dashboard_from_db(dashboard_id, request.sdb)
    return _RequestDashboard(dashboard, team[Team.owner_id.name])


async def _parse_request_chart_filters(
    filters: DashboardChartFilters | None,
    request: AthenianWebRequest,
    account: int,
) -> dict:
    values: dict = {}
    if not filters:
        return values

    if filters.repositories is not None:
        values[DashboardChart.repositories] = await parse_request_repositories(
            filters.repositories, request, account, ".filters.repositories",
        )

    if filters.environments is not None:
        values[DashboardChart.environments] = filters.environments

    if (jira := filters.jira) is not None:
        if jira.issue_types is not None:
            values[DashboardChart.jira_issue_types] = parse_request_issue_types(jira.issue_types)
        if jira.labels_include is not None:
            values[DashboardChart.jira_labels] = jira.labels_include
        if jira.priorities is not None:
            values[DashboardChart.jira_priorities] = parse_request_priorities(jira.priorities)
        if jira.projects is not None:
            values[DashboardChart.jira_projects] = jira.projects

    return values


async def _parse_request_chart_group_by(
    group_by: WebDashboardChartGroupBy | None,
    request: AthenianWebRequest,
    account: int,
) -> dict:
    values: dict = {}
    if not group_by:
        return values

    if group_by.repositories is not None:
        values[DashboardChart.repositories] = await parse_request_repositories(
            group_by.repositories, request, account, ".group_by.repositories",
        )

    for req_val, col in (
        (group_by.teams, DashboardChartGroupBy.teams),
        (group_by.jira_priorities, DashboardChartGroupBy.jira_priorities),
        (group_by.jira_issue_types, DashboardChartGroupBy.jira_issue_types),
        (group_by.jira_labels, DashboardChartGroupBy.jira_labels),
    ):
        if req_val is not None:
            values[col] = req_val
    return values


async def _dashboard_response(
    request: AthenianWebRequest,
    dashboard: _RequestDashboard,
) -> web.Response:
    charts, prefixer = await gather(
        get_dashboard_charts(dashboard.row[TeamDashboard.id.name], request.sdb),
        Prefixer.from_request(request, dashboard.account),
    )
    dashboard_model = build_dashboard_web_model(dashboard.row, charts, prefixer)
    return model_response(dashboard_model)
