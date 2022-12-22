import dataclasses

from aiohttp import web

from athenian.api.async_utils import gather
from athenian.api.db import Row
from athenian.api.internal.dashboard import (
    build_dashboard_web_model,
    create_dashboard_chart as create_dashboard_chart_in_db,
    delete_dashboard_chart as delete_dashboard_chart_from_db,
    get_dashboard as get_dashboard_from_db,
    get_dashboard_charts,
    get_team_default_dashboard,
    reorder_dashboard_charts,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.repos import dump_db_repositories
from athenian.api.internal.team import get_team_from_db
from athenian.api.models.state.models import DashboardChart, Team, TeamDashboard
from athenian.api.models.web import (
    CreatedIdentifier,
    DashboardChartCreateRequest,
    DashboardUpdateRequest,
    InvalidRequestError,
)
from athenian.api.request import AthenianWebRequest, model_from_body
from athenian.api.response import ResponseError, model_response


async def get_dashboard(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
) -> web.Response:
    """Retrieve a team dashboard."""
    dashboard = await _get_request_dashboard(request, team_id, dashboard_id)
    return _dashboard_response(request, dashboard)


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
    await reorder_dashboard_charts(dashboard.row[TeamDashboard.id.name], chart_ids, request.sdb)
    return _dashboard_response(request, dashboard)


async def create_dashboard_chart(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
    body: dict,
) -> web.Response:
    """Create a dashboard chart."""
    dashboard = await _get_request_dashboard(request, team_id, dashboard_id)
    create_request = model_from_body(DashboardChartCreateRequest, body)

    extra_values = await _parse_request_chart_filters(create_request, dashboard.account, request)

    async with request.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            chart_id = await create_dashboard_chart_in_db(
                dashboard.row[TeamDashboard.id.name], create_request, extra_values, sdb_conn,
            )
    return model_response(CreatedIdentifier(id=chart_id))


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
    create_req: DashboardChartCreateRequest,
    account: int,
    request: AthenianWebRequest,
) -> dict:
    values = {}
    if create_req.filters and (req_repositories := create_req.filters.repositories):
        prefixer = await Prefixer.from_request(request, account)
        try:
            values[DashboardChart.repositories] = dump_db_repositories(
                prefixer.reference_repositories(req_repositories),
            )
        except ValueError as e:
            raise ResponseError(InvalidRequestError.from_validation_error(e))
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
