from aiohttp import web

from athenian.api import FriendlyJson


async def get_user(request: web.Request) -> web.Response:
    """Return details about the current user."""
    return web.json_response(vars(request.user), status=200, dumps=FriendlyJson.dumps)
