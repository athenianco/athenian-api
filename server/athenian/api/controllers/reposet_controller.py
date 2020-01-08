from typing import List

from aiohttp import web
from sqlalchemy import delete, insert, select, update

from athenian.api.controllers.response import response, ResponseError
from athenian.api.models.state.models import RepositorySet
from athenian.api.models.web import CreatedIdentifier, ForbiddenError, NotFoundError


async def create_reposet(request: web.Request, body: List[str]) -> web.Response:
    """Create a repository set.

    :param id: Numeric identifier of the repository set to list.
    :param body: List of repositories to group.
    """
    # TODO(vmarkovtsev): get user's repos and check the access
    rs = RepositorySet(owner=request.user.username, items=body)
    rs.create_defaults()
    rid = await request.sdb.execute(insert(RepositorySet).values(rs.explode()))
    return response(CreatedIdentifier(rid))


async def delete_reposet(request: web.Request, id: int) -> web.Response:
    """Delete a repository set.

    :param id: Numeric identifier of the repository set to list.
    :type id: int
    """
    rs = await request.sdb.fetch_one(select([RepositorySet.owner])
                                     .where(RepositorySet.id == id))
    if len(rs) == 0:
        return ResponseError(NotFoundError(
            detail="Repository set %d does not exist" % id)).response
    if rs.owner != request.user.username:
        return ResponseError(ForbiddenError(
            detail="User %s is not allowed to access %d" % (request.user.username, id))).response
    await request.sdb.execute(delete(RepositorySet).where(RepositorySet.id == id))
    return web.Response(status=200)


async def get_reposet(request: web.Request, id: int) -> web.Response:
    """List a repository set.

    :param id: Numeric identifier of the repository set to list.
    :type id: int
    """
    rs = await request.sdb.fetch_one(select([RepositorySet.items, RepositorySet.owner])
                                     .where(RepositorySet.id == id))
    if len(rs) == 0:
        return ResponseError(NotFoundError(
            detail="Repository set %d does not exist" % id)).response
    if rs.owner != request.user.username:
        return ResponseError(ForbiddenError(
            detail="User %s is not allowed to access %d" % (request.user.username, id))).response
    # "items" collides with dict.items() so we have to access the list via []
    return web.json_response(rs["items"], status=200)


async def update_reposet(request: web.Request, id: int, body: List[str]) -> web.Response:
    """Update a repository set.

    :param id: Numeric identifier of the repository set to list.
    :type id: int
    :param body: New list of repositories in the group.
    """
    rs = await request.sdb.fetch_one(select([RepositorySet]).where(RepositorySet.id == id))
    if len(rs) == 0:
        return ResponseError(NotFoundError(
            detail="Repository set %d does not exist" % id)).response
    if rs.owner != request.user.username:
        return ResponseError(ForbiddenError(
            detail="User %s is not allowed to access %d" % (request.user.username, id))).response
    rs = RepositorySet(**rs)
    rs.items = body
    rs.refresh()
    # TODO(vmarkovtsev): get user's repos and check the access
    await request.sdb.execute(update(RepositorySet)
                              .where(RepositorySet.id == id)
                              .values(rs.explode()))
    return web.json_response(body, status=200)
