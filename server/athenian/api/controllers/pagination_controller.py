from aiohttp import web

from athenian.api.request import AthenianWebRequest


async def paginate_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Compute the balanced pagination plan for `/filter/pull_requests`."""
    raise NotImplementedError
