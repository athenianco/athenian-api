from typing import List

from aiohttp import web
from sqlalchemy import delete, insert, select, update

from athenian.api import FriendlyJson
from athenian.api.controllers.reposet import fetch_reposet
from athenian.api.controllers.user import is_admin
from athenian.api.models.metadata.github import InstallationOwner, InstallationRepo
from athenian.api.models.state.models import RepositorySet
from athenian.api.models.web import CreatedIdentifier, ForbiddenError, NoSourceDataError
from athenian.api.models.web.repository_set_create_request import RepositorySetCreateRequest
from athenian.api.models.web.repository_set_list_item import RepositorySetListItem
from athenian.api.request import AthenianWebRequest
from athenian.api.response import response, ResponseError


async def create_reposet(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create a repository set.

    :param body: List of repositories to group.
    """
    body = RepositorySetCreateRequest.from_dict(body)
    user = request.uid
    account = body.account
    try:
        adm = await is_admin(request.sdb, user, account)
    except ResponseError as e:
        return e.response
    if not adm:
        return ResponseError(ForbiddenError(
            detail="User %s is not an admin of the account %d" % (user, account))).response
    # TODO(vmarkovtsev): get user's repos and check the access
    rs = RepositorySet(owner=account, items=body.items).create_defaults()
    rid = await request.sdb.execute(insert(RepositorySet).values(rs.explode()))
    return response(CreatedIdentifier(rid))


async def delete_reposet(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete a repository set.

    :param id: Numeric identifier of the repository set to delete.
    :type id: int
    """
    try:
        _, is_admin = await fetch_reposet(id, [], request.sdb, request.uid)
    except ResponseError as e:
        return e.response
    if not is_admin:
        return ResponseError(ForbiddenError(
            detail="User %s may not modify reposet %d" % (request.uid, id))).response
    await request.sdb.execute(delete(RepositorySet).where(RepositorySet.id == id))
    return web.Response(status=200)


async def get_reposet(request: AthenianWebRequest, id: int) -> web.Response:
    """List a repository set.

    :param id: Numeric identifier of the repository set to list.
    :type id: int
    """
    try:
        rs, _ = await fetch_reposet(id, [RepositorySet.items], request.sdb, request.uid)
    except ResponseError as e:
        return e.response
    # "items" collides with dict.items() so we have to access the list via []
    return web.json_response(rs.items, status=200)


async def update_reposet(request: AthenianWebRequest, id: int, body: List[str]) -> web.Response:
    """Update a repository set.

    :param id: Numeric identifier of the repository set to update.
    :type id: int
    :param body: New list of repositories in the group.
    """
    try:
        rs, is_admin = await fetch_reposet(id, [RepositorySet], request.sdb, request.uid)
    except ResponseError as e:
        return e.response
    if not is_admin:
        return ResponseError(ForbiddenError(
            detail="User %s may not modify reposet %d" % (request.uid, id))).response
    rs.items = body
    rs.refresh()
    # TODO(vmarkovtsev): get user's repos and check the access
    await request.sdb.execute(update(RepositorySet)
                              .where(RepositorySet.id == id)
                              .values(rs.explode()))
    return web.json_response(body, status=200)


async def list_reposets(request: AthenianWebRequest, id: int) -> web.Response:
    """List the current user's repository sets."""
    try:
        await is_admin(request.sdb, request.uid, id)
    except ResponseError as e:
        return e.response
    rss = await request.sdb.fetch_all(
        select([RepositorySet]).where(RepositorySet.owner == id))
    if len(rss) == 0:
        # new account, discover the repos the metadata DB
        async with request.mdb.connection() as conn:
            iid = await conn.fetch_val(
                select([InstallationOwner.install_id])
                .where(InstallationOwner.user_id == int(request.native_uid)))
            if iid is not None:
                repos = await conn.fetch_all(
                    select([InstallationRepo.repo_full_name])
                    .where(InstallationRepo.install_id == iid))
                rs = RepositorySet(owner=id, items=repos).create_defaults()
                rs.id = await conn.execute(insert(RepositorySet).values(rs.explode()))
                rss = [vars(rs)]
            else:
                return ResponseError(NoSourceDataError(
                    detail="Metadata installation has not been registered yet.")).response
    items = [RepositorySetListItem(
        id=rs[RepositorySet.id.key],
        created=rs[RepositorySet.created_at.key],
        updated=rs[RepositorySet.updated_at.key],
        items_count=rs[RepositorySet.items_count.key],
    ).to_dict() for rs in rss]
    return web.json_response(items, status=200, dumps=FriendlyJson.dumps)
