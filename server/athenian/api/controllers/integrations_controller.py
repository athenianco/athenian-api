from aiohttp import web

from athenian.api.balancing import weight
from athenian.api.request import AthenianWebRequest


@weight(1.0)
async def match_identities(request: AthenianWebRequest, body: dict) -> web.Response:
    """Match provided people names/logins/emails to the account's GitHub organization members."""
    raise NotImplementedError
