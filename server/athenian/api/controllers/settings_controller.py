from aiohttp import web
from sqlalchemy import and_, delete, insert

from athenian.api import ResponseError
from athenian.api.controllers.account import get_user_account_status
from athenian.api.controllers.reposet import resolve_repos
from athenian.api.models.state.models import ReleaseSetting
from athenian.api.models.web import ForbiddenError, InvalidRequestError
from athenian.api.models.web.release_match_request import ReleaseMatchRequest
from athenian.api.request import AthenianWebRequest


async def set_release_match(request: AthenianWebRequest, body: dict) -> web.Response:
    """Set the release matching rule for a list of repositories."""
    rule = ReleaseMatchRequest.from_dict(body)  # type: ReleaseMatchRequest
    if rule.match == "branch" and not rule.branches:
        return ResponseError(InvalidRequestError(
            ".branches", detail='"branches" may not be empty given "match" = "branch"')).response
    if rule.match == "tag" and not rule.tags:
        return ResponseError(InvalidRequestError(
            ".tags", detail='"tags" may not be empty given "match" = "tag"')).response
    match_code = ["branch", "tag"].index(rule.match)
    async with request.sdb.connection() as conn:
        try:
            if not await get_user_account_status(request.uid, rule.account, conn, request.cache):
                return ResponseError(ForbiddenError(
                    detail="User %s is not an admin of %d" % (request.uid, rule.account))).response
        except ResponseError as e:
            return e.response
        try:
            repos = await resolve_repos(
                rule.repositories, rule.account, request.uid, request.native_uid,
                conn, request.mdb, request.cache, strip_prefix=False)
        except ResponseError as e:
            return e.response
        values = [ReleaseSetting(repository=r,
                                 account_id=rule.account,
                                 branches=rule.branches,
                                 tags=rule.tags,
                                 match=match_code,
                                 ).create_defaults().explode(with_primary_keys=True)
                  for r in repos]
        query = insert(ReleaseSetting).prefix_with("OR REPLACE", dialect="sqlite")
        if request.sdb.url.dialect != "sqlite":
            await conn.execute(delete([ReleaseSetting]).where(and_(
                ReleaseSetting.account_id == rule.account,
                ReleaseSetting.repository.in_(repos),
            )))
        await conn.execute_many(query, values)
    return web.json_response(sorted(repos))
