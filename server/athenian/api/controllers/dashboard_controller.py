from aiohttp import web

from athenian.api.request import AthenianWebRequest


async def get_dashboard(
    request: AthenianWebRequest,
    team_id: int,
    dashboard_id: int,
) -> web.Response:
    """Retrieve a team dashboard."""


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
