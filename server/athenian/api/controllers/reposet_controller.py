from datetime import timezone
import logging
from typing import List, Mapping, Optional, Union

from aiohttp import web
import aiomcache
import databases.core
from sqlalchemy import delete, insert, select, update

from athenian.api import FriendlyJson
from athenian.api.controllers.account import get_installation_id, get_user_account_status
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.controllers.reposet import fetch_reposet
from athenian.api.metadata import __package__
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import InstallationOwner, InstallationRepo
from athenian.api.models.state.models import Account, RepositorySet
from athenian.api.models.web import BadRequestError, CreatedIdentifier, ForbiddenError, \
    NoSourceDataError
from athenian.api.models.web.repository_set_create_request import RepositorySetCreateRequest
from athenian.api.models.web.repository_set_list_item import RepositorySetListItem
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


async def create_reposet(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create a repository set.

    :param body: List of repositories to group.
    """
    body = RepositorySetCreateRequest.from_dict(body)
    user = request.uid
    account = body.account
    async with request.sdb.connection() as sdb_conn:
        try:
            adm = await get_user_account_status(user, account, sdb_conn, request.cache)
        except ResponseError as e:
            return e.response
        if not adm:
            return ResponseError(ForbiddenError(
                detail="User %s is not an admin of the account %d" % (user, account))).response
        try:
            items = await _check_reposet(request, sdb_conn, body.account, body.items)
        except ResponseError as e:
            return e.response
        rs = RepositorySet(owner=account, items=items).create_defaults()
        rid = await sdb_conn.execute(insert(RepositorySet).values(rs.explode()))
        return model_response(CreatedIdentifier(rid))


async def delete_reposet(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete a repository set.

    :param id: Numeric identifier of the repository set to delete.
    :type id: int
    """
    try:
        _, is_admin = await fetch_reposet(id, [], request.uid, request.sdb, request.cache)
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
        rs, _ = await fetch_reposet(
            id, [RepositorySet.items], request.uid, request.sdb, request.cache)
    except ResponseError as e:
        return e.response
    # "items" collides with dict.items() so we have to access the list via []
    return web.json_response(rs.items, status=200)


async def _check_reposet(request: AthenianWebRequest,
                         sdb_conn: Optional[databases.core.Connection],
                         account: int,
                         body: List[str],
                         ) -> List[str]:
    service = None
    repos = set()
    for repo in body:
        for key, prefix in PREFIXES.items():
            if repo.startswith(prefix):
                if service is None:
                    service = key
                elif service != key:
                    raise ResponseError(BadRequestError(
                        detail="mixed services: %s, %s" % (service, key),
                    ))
                repos.add(repo[len(prefix):])
    if service is None:
        raise ResponseError(BadRequestError(
            detail="repository prefixes do not match to any supported service",
        ))
    checker = await access_classes[service](account, sdb_conn, request.mdb, request.cache).load()
    denied = await checker.check(repos)
    if denied:
        raise ResponseError(ForbiddenError(
            detail="the following repositories are access denied for %s: %s" % (service, denied),
        ))
    return sorted(set(body))


async def update_reposet(request: AthenianWebRequest, id: int, body: List[str]) -> web.Response:
    """Update a repository set.

    :param id: Numeric identifier of the repository set to update.
    :type id: int
    :param body: New list of repositories in the group.
    """
    async with request.sdb.connection() as sdb_conn:
        try:
            rs, is_admin = await fetch_reposet(
                id, [RepositorySet], request.uid, sdb_conn, request.cache)
        except ResponseError as e:
            return e.response
        if not is_admin:
            return ResponseError(ForbiddenError(
                detail="User %s may not modify reposet %d" % (request.uid, id))).response
        try:
            body = await _check_reposet(request, sdb_conn, id, body)
        except ResponseError as e:
            return e.response
        rs.items = body
        rs.refresh()
        await sdb_conn.execute(update(RepositorySet)
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
        try:
            iid = await get_installation_id(account, sdb_conn, cache)
        except ResponseError:
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
        await get_user_account_status(request.uid, id, request.sdb, request.cache)
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
        created=rs[RepositorySet.created_at.key].replace(tzinfo=timezone.utc),
        updated=rs[RepositorySet.updated_at.key].replace(tzinfo=timezone.utc),
        items_count=rs[RepositorySet.items_count.key],
    ).to_dict() for rs in rss]
    return web.json_response(items, status=200, dumps=FriendlyJson.dumps)
