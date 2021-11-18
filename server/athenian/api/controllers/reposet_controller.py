from datetime import timezone
from sqlite3 import IntegrityError, OperationalError
from typing import List, Optional

from aiohttp import web
from asyncpg import UniqueViolationError
import databases.core
from sqlalchemy import and_, delete, insert, select, update

from athenian.api.auth import disable_default_user
from athenian.api.controllers.account import get_metadata_account_ids, \
    get_metadata_account_ids_or_empty, get_user_account_status
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.controllers.reposet import fetch_reposet, load_account_reposets
from athenian.api.controllers.settings import Settings
from athenian.api.models.state.models import RepositorySet
from athenian.api.models.web import CreatedIdentifier, DatabaseConflict, ForbiddenError, \
    InvalidRequestError, RepositorySetWithName
from athenian.api.models.web.repository_set_create_request import RepositorySetCreateRequest
from athenian.api.models.web.repository_set_list_item import RepositorySetListItem
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


@disable_default_user
async def create_reposet(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create a repository set.

    :param body: List of repositories to group.
    """
    body = RepositorySetCreateRequest.from_dict(body)
    user = request.uid
    account = body.account
    async with request.sdb.connection() as sdb_conn:
        if not await get_user_account_status(user, account, sdb_conn, request.cache):
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


@disable_default_user
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
    :type id: repository set ID.
    """
    rs_cols = [
        RepositorySet.name,
        RepositorySet.items,
        RepositorySet.precomputed,
        RepositorySet.tracking_re,
    ]
    rs, _ = await fetch_reposet(id, rs_cols, request.uid, request.sdb, request.cache)
    repos = set(rs.items)
    meta_ids = await get_metadata_account_ids_or_empty(rs.owner_id, request.sdb, request.cache)
    if meta_ids:
        # append logical repositories, if they exist
        prefixer = await Prefixer.schedule_load(meta_ids, request.mdb, request.cache)
        settings = Settings.from_request(request, rs.owner_id)
        logical_settings = await settings.list_logical_repositories(prefixer, rs.items)
        repos = logical_settings.append_logical_repos_to_reposet(rs)
    return model_response(RepositorySetWithName(
        name=rs.name, items=sorted(repos), precomputed=rs.precomputed))


async def _check_reposet(request: AthenianWebRequest,
                         sdb_conn: Optional[databases.core.Connection],
                         account: int,
                         body: List[str],
                         ) -> List[str]:
    repos = {repo.split("/", 1)[1] for repo in body}
    meta_ids = await get_metadata_account_ids(account, sdb_conn, request.cache)
    checker = await access_classes["github"](
        account, meta_ids, sdb_conn, request.mdb, request.cache,
    ).load()
    denied = await checker.check(repos)
    if denied:
        raise ResponseError(ForbiddenError(
            detail="the following repositories are access denied for account %d: %s" %
                   (account, denied),
        ))
    return sorted(set(body))


@disable_default_user
async def update_reposet(request: AthenianWebRequest, id: int, body: dict) -> web.Response:
    """Update a repository set.

    :param id: Numeric identifier of the repository set to update.
    :type id: int
    :param body: New reposet definition.
    """
    body = RepositorySetWithName.from_dict(body)
    if body.items is not None and len(body.items) == 0:
        raise ResponseError(InvalidRequestError(
            detail="Reposet may not be empty (did you mean DELETE?).",
            pointer=".items",
        ))
    if body.name is not None and body.name == "":
        raise ResponseError(InvalidRequestError(
            detail="Reposet may not be named as an empty string.",
            pointer=".name",
        ))
    async with request.sdb.connection() as sdb_conn:
        rs, is_admin = await fetch_reposet(
            id, [RepositorySet], request.uid, sdb_conn, request.cache)
        if not is_admin:
            raise ResponseError(ForbiddenError(
                detail="User %s may not modify reposet %d" % (request.uid, id)))
        if body.items is not None:
            new_items = await _check_reposet(request, sdb_conn, rs.owner_id, body.items)
        else:
            new_items = None
        changed = False
        if body.name is not None and body.name != rs.name:
            dupe_id = await sdb_conn.fetch_val(select([RepositorySet.id])
                                               .where(and_(RepositorySet.owner_id == rs.owner_id,
                                                           RepositorySet.name == body.name)))
            if dupe_id is not None:
                raise ResponseError(DatabaseConflict(
                    detail="there is an existing reposet %s with the same name" % dupe_id))
            rs.name = body.name
            changed = True
        if new_items is not None and new_items != rs.items:
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
        return model_response(RepositorySetWithName(
            name=rs.name,
            items=rs.items,
            precomputed=rs.precomputed,
        ))


async def list_reposets(request: AthenianWebRequest, id: int) -> web.Response:
    """List the current user's repository sets."""
    await get_user_account_status(request.uid, id, request.sdb, request.cache)

    async def login_loader() -> str:
        return (await request.user()).login

    rss = await load_account_reposets(
        id, login_loader, [RepositorySet], request.sdb, request.mdb, request.cache,
        request.app["slack"])
    items = [RepositorySetListItem(
        id=rs[RepositorySet.id.name],
        name=rs[RepositorySet.name.name],
        created=rs[RepositorySet.created_at.name].replace(tzinfo=timezone.utc),
        updated=rs[RepositorySet.updated_at.name].replace(tzinfo=timezone.utc),
        items_count=rs[RepositorySet.items_count.name],
    ) for rs in rss]
    return model_response(items)
