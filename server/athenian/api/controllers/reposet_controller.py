import logging
from typing import List, Mapping, Optional, Union

from aiohttp import web
import aiomcache
import databases.core
from sqlalchemy import delete, insert, select, update

from athenian.api import FriendlyJson
from athenian.api.controllers.account import get_installation_id
from athenian.api.controllers.reposet import fetch_reposet
from athenian.api.controllers.user import is_admin
from athenian.api.metadata import __package__
from athenian.api.models.metadata.github import InstallationOwner, InstallationRepo
from athenian.api.models.state.models import Account, RepositorySet
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


async def load_account_reposets(account: int,
                                native_uid: str,
                                fields: list,
                                sdb_conn: Union[databases.Database, databases.core.Connection],
                                mdb_conn: Union[databases.Database, databases.core.Connection],
                                cache: Optional[aiomcache.Client],
                                ) -> List[Mapping]:
    """
    Load the account's repository sets and create one if no exists.

    :param sdb_conn: Connection to the state DB.
    :param mdb_conn: Connection to the metadata DB, needed only if no reposet exists.
    :param cache: memcached Client.
    :param account: Owner of the loaded reposets.
    :param native_uid: Native user ID, needed only if no reposet exists.
    :param fields: Which columns to fetch for each RepositorySet.
    :return: List of DB rows or __dict__-s representing the loaded RepositorySets.
    """
    rss = await sdb_conn.fetch_all(select(fields)
                                   .where(RepositorySet.owner == account)
                                   .order_by(RepositorySet.created_at))
    if rss:
        return rss

    async def create_new_reposet(_mdb_conn: databases.core.Connection):
        # a new account, discover their repos from the installation and create the first reposet
        iid = await get_installation_id(account, sdb_conn, cache)
        if iid is None:
            iid = await _mdb_conn.fetch_val(select([InstallationOwner.install_id])
                                            .where(InstallationOwner.user_id == int(native_uid))
                                            .order_by(InstallationOwner.created_at.desc()))
            if iid is None:
                raise ResponseError(NoSourceDataError(
                    detail="The metadata installation has not registered yet."))
            await sdb_conn.execute(update(Account)
                                   .where(Account.id == account)
                                   .values({Account.installation_id.key: iid}))
        repos = await _mdb_conn.fetch_all(select([InstallationRepo.repo_full_name])
                                          .where(InstallationRepo.install_id == iid))
        repos = [("github.com/" + r[InstallationRepo.repo_full_name.key]) for r in repos]
        rs = RepositorySet(owner=account, items=repos).create_defaults()
        rs.id = await sdb_conn.execute(insert(RepositorySet).values(rs.explode()))
        logging.getLogger(__package__).info(
            "Created the first reposet %d for account %d with %d repos on behalf of %s",
            rs.id, account, len(repos), native_uid,
        )
        return [vars(rs)]

    if isinstance(mdb_conn, databases.Database):
        async with mdb_conn.connection() as _mdb_conn:
            return await create_new_reposet(_mdb_conn)
    return await create_new_reposet(mdb_conn)


async def list_reposets(request: AthenianWebRequest, id: int) -> web.Response:
    """List the current user's repository sets."""
    try:
        await is_admin(request.sdb, request.uid, id)
    except ResponseError as e:
        return e.response
    async with request.sdb.connection() as sdb_conn:
        try:
            rss = await load_account_reposets(
                id, request.native_uid, [RepositorySet], sdb_conn, request.mdb, request.cache)
        except ResponseError as e:
            return e.response
    items = [RepositorySetListItem(
        id=rs[RepositorySet.id.key],
        created=rs[RepositorySet.created_at.key],
        updated=rs[RepositorySet.updated_at.key],
        items_count=rs[RepositorySet.items_count.key],
    ).to_dict() for rs in rss]
    return web.json_response(items, status=200, dumps=FriendlyJson.dumps)
