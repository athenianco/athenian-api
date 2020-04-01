from aiohttp import web

from athenian.api.request import AthenianWebRequest


async def set_release_match(request: AthenianWebRequest, body: dict) -> web.Response:
    """Set the release matching rule for a list of repositories."""
    pass
