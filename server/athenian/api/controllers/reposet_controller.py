from typing import List

from aiohttp import web
from sqlalchemy import delete, select

from athenian.api.controllers.response import response, ResponseError
from athenian.api.models.state.models import RepositorySet as DBRepositorySet
from athenian.api.models.web import ForbiddenError, NotFoundError, RepositorySet as WebRepositorySet


async def create_reposet(request: web.Request, body=None) -> web.Response:
    """Create a repository set.

    :param id: Numeric identifier of the repository set to list.
    :param body: List of repositories to group.
    :type body: List[str]
    """
    body = WebRepositorySet.from_dict(body)
    # TODO(vmarkovtsev): get user's repos and check the access
    await request.sdb.execute(DBRepositorySet(owner=request.user.username, items=body))
    return web.Response(status=200)


async def delete_reposet(request: web.Request, id: int) -> web.Response:
    """Delete a repository set.

    :param id: Numeric identifier of the repository set to list.
    :type id: int
    """
    rs = await request.sdb.fetch_one(select([DBRepositorySet.owner])
                                     .where(DBRepositorySet.id == id))
    if len(rs) == 0:
        return ResponseError(NotFoundError(
            detail="Repository set %d does not exist" % id)).response
    if rs.owner != request.user.username:
        return ResponseError(ForbiddenError(
            detail="User %s is not allowed to access %d" % (request.user.username, id))).response
    await request.sdb.execute(delete(DBRepositorySet).where(DBRepositorySet.id == id))
    return web.Response(status=200)


async def get_reposet(request: web.Request, id: int) -> web.Response:
    """List a repository set.

    :param id: Numeric identifier of the repository set to list.
    :type id: int
    """
    rs = await request.sdb.fetch_one(select([DBRepositorySet.items, DBRepositorySet.owner])
                                     .where(DBRepositorySet.id == id))
    if len(rs) == 0:
        return ResponseError(NotFoundError(
            detail="Repository set %d does not exist" % id)).response
    if rs.owner != request.user.username:
        return ResponseError(ForbiddenError(
            detail="User %s is not allowed to access %d" % (request.user.username, id))).response
    # "items" collides with dict.items() so we have to access the list via []
    return web.json_response(rs["items"], status=200)


async def update_reposet(request: web.Request, id: int, body=None) -> web.Response:
    """Update a repository set.

    :param id: Numeric identifier of the repository set to list.
    :type id: int
    :param body:
    :type body: List[str]
    """
    rs = await request.sdb.fetch_one(select([DBRepositorySet.items, DBRepositorySet.owner])
                                     .where(DBRepositorySet.id == id))
    if len(rs) == 0:
        return ResponseError(NotFoundError(
            detail="Repository set %d does not exist" % id)).response
    if rs.owner != request.user.username:
        return ResponseError(ForbiddenError(
            detail="User %s is not allowed to access %d" % (request.user.username, id))).response

    return web.Response(status=200)
