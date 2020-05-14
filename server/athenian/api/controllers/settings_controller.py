from aiohttp import web

from athenian.api import ResponseError
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.settings import Match, Settings
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.web import ForbiddenError, ReleaseMatchSetting
from athenian.api.models.web.release_match_request import ReleaseMatchRequest
from athenian.api.request import AthenianWebRequest


async def list_release_match_settings(request: AthenianWebRequest, id: int) -> web.Response:
    """List the current release matching settings."""
    settings = Settings.from_request(request, id)
    try:
        settings = await settings.list_release_matches()
    except ResponseError as e:
        return e.response
    model = {k: ReleaseMatchSetting.from_dataclass(m).to_dict() for k, m in settings.items()}
    repos = [r.split("/", 1)[1] for r in settings]
    _, default_branches = await extract_branches(repos, request.mdb, request.cache)
    prefix = PREFIXES["github"]
    for repo, name in default_branches.items():
        model[prefix + repo]["default_branch"] = name
    return web.json_response(model)


async def set_release_match(request: AthenianWebRequest, body: dict) -> web.Response:
    """Set the release matching rule for a list of repositories."""
    if request.is_default_user:
        return ResponseError(ForbiddenError("%s is the default user" % request.uid)).response
    rule = ReleaseMatchRequest.from_dict(body)  # type: ReleaseMatchRequest
    settings = Settings.from_request(request, rule.account)
    match = Match[rule.match]
    try:
        repos = await settings.set_release_matches(
            rule.repositories, rule.branches, rule.tags, match)
    except ResponseError as e:
        return e.response
    return web.json_response(sorted(repos))
