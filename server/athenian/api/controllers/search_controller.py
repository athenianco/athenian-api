from aiohttp import web

from athenian.api.request import AthenianWebRequest


async def search_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Search pull requests that satisfy the filters."""
    raise NotImplementedError
