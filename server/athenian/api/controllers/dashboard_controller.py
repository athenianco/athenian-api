from aiohttp import web

from athenian.api.internal.dashboard import (
    build_dashboard_web_model,
    get_dashboard as get_dashboard_from_db,
    get_dashboard_charts,
)
from athenian.api.internal.team import get_team_from_db
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response


async def get_dashboard(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
) -> web.Response:
    """Retrieve a team dashboard."""
    await get_team_from_db(team_id, account_id=None, user_id=request.uid, sdb_conn=request.sdb)
    if dashboard_id == 0:
        pass
    else:
        dashboard = await get_dashboard_from_db(dashboard_id, request.sdb)

    charts = await get_dashboard_charts(dashboard_id, request.sdb)
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


async def delete_dashboard_chart(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
    chart_id: int,
) -> web.Response:
    """Delete an existing dashboard chart."""
