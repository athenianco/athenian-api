from typing import List

from aiohttp import web
from sqlalchemy import delete, insert, update

from athenian.api.controllers.reposet import fetch_reposet
from athenian.api.controllers.response import response, ResponseError
from athenian.api.models.state.models import RepositorySet
from athenian.api.models.web import CreatedIdentifier


async def create_reposet(request: web.Request, body: List[str]) -> web.Response:
    """Create a repository set.

    :param body: List of repositories to group.
    """
    # TODO(vmarkovtsev): get user's repos and check the access
    rs = RepositorySet(owner=request.user.username, items=body)
    rs.create_defaults()
    rid = await request.sdb.execute(insert(RepositorySet).values(rs.explode()))
    return response(CreatedIdentifier(rid))


async def delete_reposet(request: web.Request, id: int) -> web.Response:
    """Delete a repository set.

    :param id: Numeric identifier of the repository set to delete.
    :type id: int
    """
    try:
        await fetch_reposet(id, [], request.sdb, request.user)
    except ResponseError as e:
        return e.response
    await request.sdb.execute(delete(RepositorySet).where(RepositorySet.id == id))
    return web.Response(status=200)


async def get_reposet(request: web.Request, id: int) -> web.Response:
    """List a repository set.

    :param id: Numeric identifier of the repository set to list.
    :type id: int
    """
    try:
        rs = await fetch_reposet(id, [RepositorySet.items], request.sdb, request.user)
    except ResponseError as e:
        return e.response
    # "items" collides with dict.items() so we have to access the list via []
    return web.json_response(rs["items"], status=200)


async def update_reposet(request: web.Request, id: int, body: List[str]) -> web.Response:
    """Update a repository set.

    :param id: Numeric identifier of the repository set to update.
    :type id: int
    :param body: New list of repositories in the group.
    """
    try:
        rs = await fetch_reposet(id, [RepositorySet], request.sdb, request.user)
    except ResponseError as e:
        return e.response
    rs = RepositorySet(**rs)
    rs.items = body
    rs.refresh()
    # TODO(vmarkovtsev): get user's repos and check the access
    await request.sdb.execute(update(RepositorySet)
                              .where(RepositorySet.id == id)
                              .values(rs.explode()))
    return web.json_response(body, status=200)
