from aiohttp import web

from athenian.api import ResponseError
from athenian.api.controllers.settings import Match, Settings
from athenian.api.models.web.release_match_request import ReleaseMatchRequest
from athenian.api.request import AthenianWebRequest


async def list_release_match_settings(request: AthenianWebRequest, id: int) -> web.Response:
    """List the current release matching settings."""
    settings = Settings.from_request(request, id)
    try:
        model = await settings.list_release_matches()
    except ResponseError as e:
        return e.response
    return web.json_response({k: m.to_dict() for k, m in model.items()})


async def set_release_match(request: AthenianWebRequest, body: dict) -> web.Response:
    """Set the release matching rule for a list of repositories."""
    rule = ReleaseMatchRequest.from_dict(body)  # type: ReleaseMatchRequest
    settings = Settings.from_request(request, rule.account)
    match = Match[rule.match]
    try:
        repos = await settings.set_release_matches(
            rule.repositories, rule.branches, rule.tags, match)
    except ResponseError as e:
        return e.response
    return web.json_response(sorted(repos))
