from aiohttp import web

from athenian.api.db import Row
from athenian.api.internal.dashboard import (
    build_dashboard_web_model,
    create_dashboard_chart as create_dashboard_chart_in_db,
    delete_dashboard_chart as delete_dashboard_chart_from_db,
    get_dashboard as get_dashboard_from_db,
    get_dashboard_charts,
    get_team_default_dashboard,
)
from athenian.api.internal.team import get_team_from_db
from athenian.api.models.state.models import TeamDashboard
from athenian.api.models.web import CreatedIdentifier, DashboardChartCreateRequest
from athenian.api.request import AthenianWebRequest, model_from_body
from athenian.api.response import model_response


async def get_dashboard(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
) -> web.Response:
    """Retrieve a team dashboard."""
    dashboard = await _get_request_dashboard(request, team_id, dashboard_id)
    charts = await get_dashboard_charts(dashboard[TeamDashboard.id.name], request.sdb)
    dashboard_model = build_dashboard_web_model(dashboard, charts)
    return model_response(dashboard_model)


async def update_dashboard(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
) -> web.Response:
    """Update an existing team dashboard."""


async def create_dashboard_chart(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
    body: dict,
) -> web.Response:
    """Create a dashboard chart."""
    dashboard = await _get_request_dashboard(request, team_id, dashboard_id)
    create_request = model_from_body(DashboardChartCreateRequest, body)

    async with request.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            chart_id = await create_dashboard_chart_in_db(
                dashboard[TeamDashboard.id.name], create_request, sdb_conn,
            )
    return model_response(CreatedIdentifier(id=chart_id))


async def delete_dashboard_chart(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
    chart_id: int,
) -> web.Response:
    """Delete an existing dashboard chart."""
    dashboard = await _get_request_dashboard(request, team_id, dashboard_id)
    await delete_dashboard_chart_from_db(dashboard[TeamDashboard.id.name], chart_id, request.sdb)
    return web.Response(status=204)


async def _get_request_dashboard(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
) -> Row:
    # this is to check team existence and permissions
    await get_team_from_db(team_id, account_id=None, user_id=request.uid, sdb_conn=request.sdb)
    if dashboard_id == 0:
        async with request.sdb.connection() as sdb_conn:
            return await get_team_default_dashboard(team_id, sdb_conn)
    else:
        return await get_dashboard_from_db(dashboard_id, request.sdb)
