from datetime import timezone
from sqlite3 import IntegrityError, OperationalError
from typing import List, Optional, Sequence, Tuple, Type, Union

from aiohttp import web
import aiomcache
from asyncpg import UniqueViolationError
from sqlalchemy import and_, delete, insert, select, update
from sqlalchemy.orm import InstrumentedAttribute

from athenian.api.auth import disable_default_user
from athenian.api.db import DatabaseLike
from athenian.api.internal.account import (
    get_metadata_account_ids,
    get_user_account_status,
    get_user_account_status_from_request,
    only_admin,
)
from athenian.api.internal.miners.access_classes import access_classes
from athenian.api.internal.reposet import fetch_reposet, load_account_reposets
from athenian.api.models.state.models import RepositorySet
from athenian.api.models.web import (
    CreatedIdentifier,
    DatabaseConflict,
    ForbiddenError,
    InvalidRequestError,
    RepositorySetWithName,
)
from athenian.api.models.web.repository_set_create_request import RepositorySetCreateRequest
from athenian.api.models.web.repository_set_list_item import RepositorySetListItem
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response


@disable_default_user
@only_admin
async def create_reposet(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create a repository set.

    :param body: List of repositories to group.
    """
    body = RepositorySetCreateRequest.from_dict(body)
    account = body.account
    async with request.sdb.connection() as sdb_conn:
        dupe_id = await sdb_conn.fetch_val(
            select([RepositorySet.id]).where(
                and_(RepositorySet.owner_id == account, RepositorySet.name == body.name),
            ),
        )
        if dupe_id is not None:
            raise ResponseError(
                DatabaseConflict(
                    detail="there is an existing reposet %s with the same name" % dupe_id,
                ),
            )
        items = await _ensure_reposet(
            body.account, body.items, sdb_conn, request.mdb, request.cache,
        )
        rs = RepositorySet(name=body.name, owner_id=account, items=items).create_defaults()
        try:
            rid = await sdb_conn.execute(insert(RepositorySet).values(rs.explode()))
        except (UniqueViolationError, IntegrityError, OperationalError):
            raise ResponseError(
                DatabaseConflict(detail="there is an existing reposet with the same items"),
            )
        return model_response(CreatedIdentifier(id=rid))


async def _fetch_reposet_with_owner(
    id: int,
    columns: Union[Sequence[Type[RepositorySet]], Sequence[InstrumentedAttribute]],
    uid: str,
    sdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
):
    rs = await fetch_reposet(id, columns, sdb)
    is_admin = await get_user_account_status(uid, rs.owner_id, sdb, None, None, None, cache)
    return rs, is_admin


@disable_default_user
async def delete_reposet(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete a repository set.

    :param id: Numeric identifier of the repository set to delete.
    """
    _, is_admin = await _fetch_reposet_with_owner(id, [], request.uid, request.sdb, request.cache)
    if not is_admin:
        raise ResponseError(
            ForbiddenError(detail="User %s may not modify reposet %d" % (request.uid, id)),
        )
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
    rs, _ = await _fetch_reposet_with_owner(id, rs_cols, request.uid, request.sdb, request.cache)
    return model_response(
        RepositorySetWithName(
            name=rs.name, items=[r[0] for r in rs.items], precomputed=rs.precomputed,
        ),
    )


async def _ensure_reposet(
    account: int,
    body: List[str],
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> List[Tuple[str, int]]:
    repos = {repo.split("/", 1)[1] for repo in body}
    meta_ids = await get_metadata_account_ids(account, sdb, cache)
    checker = await access_classes["github"](account, meta_ids, sdb, mdb, cache).load()
    denied = await checker.check(repos)
    if denied:
        raise ResponseError(
            ForbiddenError(
                detail="the following repositories are access denied for account %d: %s"
                % (account, denied),
            ),
        )
    return sorted((name, checker.installed_repos[name.split("/", 1)[1]]) for name in set(body))


@disable_default_user
async def update_reposet(request: AthenianWebRequest, id: int, body: dict) -> web.Response:
    """Update a repository set.

    :param id: Numeric identifier of the repository set to update.
    :type id: int
    :param body: New reposet definition.
    """
    body = RepositorySetWithName.from_dict(body)
    if body.items is not None and len(body.items) == 0:
        raise ResponseError(
            InvalidRequestError(
                detail="Reposet may not be empty (did you mean DELETE?).",
                pointer=".items",
            ),
        )
    if body.name is not None and body.name == "":
        raise ResponseError(
            InvalidRequestError(
                detail="Reposet may not be named as an empty string.",
                pointer=".name",
            ),
        )
    async with request.sdb.connection() as sdb_conn:
        rs, is_admin = await _fetch_reposet_with_owner(
            id, [RepositorySet], request.uid, sdb_conn, request.cache,
        )
        if not is_admin:
            raise ResponseError(
                ForbiddenError(detail="User %s may not modify reposet %d" % (request.uid, id)),
            )
        if body.items is not None:
            new_items = await _ensure_reposet(
                rs.owner_id, body.items, sdb_conn, request.mdb, request.cache,
            )
        else:
            new_items = None
        changed = False
        if body.name is not None and body.name != rs.name:
            dupe_id = await sdb_conn.fetch_val(
                select([RepositorySet.id]).where(
                    and_(RepositorySet.owner_id == rs.owner_id, RepositorySet.name == body.name),
                ),
            )
            if dupe_id is not None:
                raise ResponseError(
                    DatabaseConflict(
                        detail="there is an existing reposet %s with the same name" % dupe_id,
                    ),
                )
            rs.name = body.name
            changed = True
        if new_items is not None and new_items != rs.items:
            rs.items = new_items
            rs.precomputed = False
            rs.refresh()
            changed = True
        if changed:
            try:
                await sdb_conn.execute(
                    update(RepositorySet).where(RepositorySet.id == id).values(rs.explode()),
                )
            except (UniqueViolationError, IntegrityError, OperationalError):
                raise ResponseError(
                    DatabaseConflict(detail="there is an existing reposet with the same items"),
                )
        return model_response(
            RepositorySetWithName(
                name=rs.name,
                items=[r[0] for r in rs.items],
                precomputed=rs.precomputed,
            ),
        )


async def list_reposets(request: AthenianWebRequest, id: int) -> web.Response:
    """List the current user's repository sets."""
    await get_user_account_status_from_request(request, id)

    async def login_loader() -> str:
        return (await request.user()).login

    rss = await load_account_reposets(
        id,
        login_loader,
        [RepositorySet],
        request.sdb,
        request.mdb,
        request.cache,
        request.app["slack"],
    )
    items = [
        RepositorySetListItem(
            id=rs[RepositorySet.id.name],
            name=rs[RepositorySet.name.name],
            created=rs[RepositorySet.created_at.name].replace(tzinfo=timezone.utc),
            updated=rs[RepositorySet.updated_at.name].replace(tzinfo=timezone.utc),
            items_count=len(rs[RepositorySet.items.name]),
        )
        for rs in rss
    ]
    return model_response(items)
