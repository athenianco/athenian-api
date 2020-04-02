from enum import IntEnum
import re
from typing import Dict

from aiohttp import web
from sqlalchemy import and_, delete, insert, select

from athenian.api import ResponseError
from athenian.api.controllers.account import get_user_account_status
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.controllers.reposet import resolve_repos
from athenian.api.models.state.models import ReleaseSetting
from athenian.api.models.web import ForbiddenError, InvalidRequestError, \
    ReleaseMatchSetting as ReleaseMatchSettingWeb, ReleaseMatchStrategy
from athenian.api.models.web.release_match_request import ReleaseMatchRequest
from athenian.api.request import AthenianWebRequest


Match = IntEnum("Match", {ReleaseMatchStrategy.BRANCH: 0,
                          ReleaseMatchStrategy.TAG: 1,
                          ReleaseMatchStrategy.TAG_OR_BRANCH: 2})
Match.__doc__ = """Supported release matching strategies."""

default_branch_alias = "{{default}}"


async def list_release_match_settings(request: AthenianWebRequest, id: int) -> web.Response:
    """List the current release matching settings."""
    async with request.sdb.connection() as conn:
        try:
            await get_user_account_status(request.uid, id, conn, request.cache)
            repos = set()
            for cls in access_classes.values():
                repos.update(
                    (await cls(id, conn, request.mdb, request.cache).load()).installed_repos())
        except ResponseError as e:
            return e.response
        model: Dict[str, ReleaseMatchSettingWeb] = {}
        setting_rows = await conn.fetch_all(
            select([ReleaseSetting]).where(and_(ReleaseSetting.account_id == id,
                                                ReleaseSetting.repository.in_(repos))))
        for row in setting_rows:
            model[row[ReleaseSetting.repository.key]] = ReleaseMatchSettingWeb(
                branches=row[ReleaseSetting.branches.key],
                tags=row[ReleaseSetting.tags.key],
                match=Match(row[ReleaseSetting.match.key]).name,
            )
        for repo in repos:
            if repo not in model:
                model[repo] = ReleaseMatchSettingWeb(
                    branches=default_branch_alias,
                    tags=".*",
                    match=Match.tag_or_branch.name,
                )
    return web.json_response({k: m.to_dict() for k, m in model.items()})


async def set_release_match(request: AthenianWebRequest, body: dict) -> web.Response:
    """Set the release matching rule for a list of repositories."""
    rule = ReleaseMatchRequest.from_dict(body)  # type: ReleaseMatchRequest
    for propname, s in (("branches", Match.branch.name), ("tags", Match.tag.name)):
        propval = getattr(rule, propname, None)
        if rule.match in (s, Match.tag_or_branch.name) and not propval:
            return ResponseError(InvalidRequestError(
                "." + propname,
                detail='Value may not be empty given "match" = "%s"' % rule.match),
            ).response
        try:
            re.compile(propval)
        except re.error as e:
            return ResponseError(InvalidRequestError(
                "." + propname,
                detail="Invalid regular expression: %s" % e),
            ).response
    if not rule.branches:
        rule.branches = default_branch_alias
    if not rule.tags:
        rule.tags = ".*"
    match_code = Match[rule.match]
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
