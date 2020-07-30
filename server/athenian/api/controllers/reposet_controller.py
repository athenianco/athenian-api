from datetime import timezone
from sqlite3 import IntegrityError, OperationalError
from typing import List, Optional

from aiohttp import web
from asyncpg import UniqueViolationError
import databases.core
from sqlalchemy import and_, delete, insert, select, update

from athenian.api.controllers.account import get_user_account_status
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.controllers.reposet import fetch_reposet, load_account_reposets
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.state.models import RepositorySet
from athenian.api.models.web import BadRequestError, CreatedIdentifier, DatabaseConflict, \
    ForbiddenError, RepositorySetWithName
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
        adm = await get_user_account_status(user, account, sdb_conn, request.cache)
        if not adm:
            raise ResponseError(ForbiddenError(
                detail="User %s is not an admin of the account %d" % (user, account)))
        dupe_id = await sdb_conn.fetch_val(select([RepositorySet.id])
                                           .where(and_(RepositorySet.owner_id == account,
                                                       RepositorySet.name == body.name)))
        if dupe_id is not None:
            raise ResponseError(DatabaseConflict(
                detail="there is an existing reposet %s with the same name" % dupe_id))
        items = await _check_reposet(request, sdb_conn, body.account, body.items)
        rs = RepositorySet(name=body.name, owner_id=account, items=items).create_defaults()
        try:
            rid = await sdb_conn.execute(insert(RepositorySet).values(rs.explode()))
        except (UniqueViolationError, IntegrityError, OperationalError):
            raise ResponseError(DatabaseConflict(
                detail="there is an existing reposet with the same items"))
        return model_response(CreatedIdentifier(rid))


async def delete_reposet(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete a repository set.

    :param id: Numeric identifier of the repository set to delete.
    :type id: int
    """
    _, is_admin = await fetch_reposet(id, [], request.uid, request.sdb, request.cache)
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
    rs, _ = await fetch_reposet(id, [RepositorySet.name, RepositorySet.items,
                                     RepositorySet.precomputed],
                                request.uid, request.sdb, request.cache)
    # "items" collides with dict.items() so we have to access the list via []
    return model_response(RepositorySetWithName(
        name=rs.name, items=rs.items, precomputed=rs.precomputed))


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


async def update_reposet(request: AthenianWebRequest, id: int, body: dict) -> web.Response:
    """Update a repository set.

    :param id: Numeric identifier of the repository set to update.
    :type id: int
    :param body: New reposet definition.
    """
    body = RepositorySetWithName.from_dict(body)  # type: RepositorySetWithName
    async with request.sdb.connection() as sdb_conn:
        rs, is_admin = await fetch_reposet(
            id, [RepositorySet], request.uid, sdb_conn, request.cache)
        if not is_admin:
            raise ResponseError(ForbiddenError(
                detail="User %s may not modify reposet %d" % (request.uid, id)))
        new_items = await _check_reposet(request, sdb_conn, id, body.items)
        changed = False
        if body.name != rs.name:
            dupe_id = await sdb_conn.fetch_val(select([RepositorySet.id])
                                               .where(and_(RepositorySet.owner_id == rs.owner_id,
                                                           RepositorySet.name == body.name)))
            if dupe_id is not None:
                raise ResponseError(DatabaseConflict(
                    detail="there is an existing reposet %s with the same name" % dupe_id))
            rs.name = body.name
            changed = True
        if new_items != rs.items:
            rs.items = new_items
            rs.precomputed = False
            rs.refresh()
            changed = True
        if changed:
            try:
                await sdb_conn.execute(update(RepositorySet)
                                       .where(RepositorySet.id == id)
                                       .values(rs.explode()))
            except (UniqueViolationError, IntegrityError, OperationalError):
                raise ResponseError(DatabaseConflict(
                    detail="there is an existing reposet with the same items"))
        return model_response(body)


async def list_reposets(request: AthenianWebRequest, id: int) -> web.Response:
    """List the current user's repository sets."""
    await get_user_account_status(request.uid, id, request.sdb, request.cache)
    async with request.sdb.connection() as sdb_conn:
        rss = await load_account_reposets(
            id, request.native_uid, [RepositorySet], sdb_conn, request.mdb, request.cache,
            request.app["slack"])
    items = [RepositorySetListItem(
        id=rs[RepositorySet.id.key],
        name=rs[RepositorySet.name.key],
        created=rs[RepositorySet.created_at.key].replace(tzinfo=timezone.utc),
        updated=rs[RepositorySet.updated_at.key].replace(tzinfo=timezone.utc),
        items_count=rs[RepositorySet.items_count.key],
    ) for rs in rss]
    return model_response(items)
