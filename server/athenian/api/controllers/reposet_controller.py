import asyncio
from datetime import timezone
import logging
from sqlite3 import IntegrityError, OperationalError
from typing import Optional, Sequence, Type

from aiohttp import web
import aiomcache
from asyncpg import UniqueViolationError
from sqlalchemy import and_, delete, insert, select, update
from sqlalchemy.orm import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.auth import disable_default_user
from athenian.api.db import DatabaseLike
from athenian.api.internal.account import (
    RepositoryReference,
    get_metadata_account_ids,
    get_user_account_status,
    get_user_account_status_from_request,
    only_admin,
)
from athenian.api.internal.miners.access_classes import access_classes
from athenian.api.internal.prefixer import Prefixer, RepositoryName
from athenian.api.internal.reposet import (
    fetch_reposet,
    load_account_reposets,
    reposet_items_to_refs,
)
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
from athenian.api.tracing import sentry_span


@disable_default_user
@only_admin
async def create_reposet(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create a repository set.

    :param body: list of repositories to group.
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
    columns: Sequence[Type[RepositorySet]] | Sequence[InstrumentedAttribute],
    uid: str,
    sdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> tuple[RepositorySet, bool]:
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
    return web.Response()


async def get_reposet(request: AthenianWebRequest, id: int) -> web.Response:
    """List a repository set.

    :param id: Numeric identifier of the repository set to list.
    """
    rs_cols = [
        RepositorySet.name,
        RepositorySet.items,
        RepositorySet.owner_id,
        RepositorySet.precomputed,
        RepositorySet.tracking_re,
    ]
    rs, _ = await _fetch_reposet_with_owner(id, rs_cols, request.uid, request.sdb, request.cache)
    meta_ids = await get_metadata_account_ids(rs.owner_id, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    items, missing = prefixer.dereference_repositories(
        reposet_items_to_refs(rs.items), return_missing=True,
    )
    if missing:
        log = logging.getLogger(f"{metadata.__package__}.get_reposet")
        log.error(
            "reposet-sync did not delete %d repositories in account %d: %s",
            len(missing),
            rs.owner_id,
            ", ".join("(%s, %d, %s)" % tuple(rs.items[i]) for i in missing),
        )
    return model_response(
        RepositorySetWithName(
            name=rs.name,
            items=sorted(str(r) for r in items),
            precomputed=rs.precomputed,
        ),
    )


async def _ensure_reposet(
    account: int,
    body: list[str],
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> list[RepositoryReference]:
    try:
        repos = {RepositoryName.from_prefixed(repo) for repo in body}
    except ValueError as e:
        raise ResponseError(InvalidRequestError("items", str(e))) from None
    grouped = {}
    for name in repos:
        grouped.setdefault(name.prefix, set()).add(name.unprefixed)
    meta_ids = await get_metadata_account_ids(account, sdb, cache)
    checkers = {}

    async def check_access(prefix: str, unprefixed: set[str]):
        checkers[prefix] = checker = await access_classes[prefix](
            account, meta_ids, sdb, mdb, cache,
        ).load()
        denied = await checker.check(unprefixed)
        if denied:
            raise ResponseError(
                ForbiddenError(
                    detail="the following repositories are access denied for account %d: %s"
                    % (account, denied),
                ),
            )

    await gather(*(check_access(*pair) for pair in grouped.items()))
    return sorted(
        RepositoryReference(
            name.prefix, checkers[name.prefix].installed_repos[name.unprefixed], name.logical,
        )
        for name in repos
    )


@disable_default_user
async def update_reposet(request: AthenianWebRequest, id: int, body: dict) -> web.Response:
    """Update a repository set.

    :param id: Numeric identifier of the repository set to update.
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

    @sentry_span
    async def load_prefixer(account: int) -> Prefixer:
        meta_ids = await get_metadata_account_ids(account, request.sdb, request.cache)
        return await Prefixer.load(meta_ids, request.mdb, request.cache)

    async with request.sdb.connection() as sdb_conn:
        rs, is_admin = await _fetch_reposet_with_owner(
            id, [RepositorySet], request.uid, sdb_conn, request.cache,
        )
        if not is_admin:
            raise ResponseError(
                ForbiddenError(detail="User %s may not modify reposet %d" % (request.uid, id)),
            )
        prefix_loader = asyncio.create_task(load_prefixer(rs.owner_id), name="load_prefixer")
        try:
            if body.items is not None:
                new_items = await _ensure_reposet(
                    rs.owner_id, body.items, sdb_conn, request.mdb, request.cache,
                )
            else:
                new_items = None
            changed = False
            if body.name is not None and body.name != rs.name:
                dupe_id = await sdb_conn.fetch_val(
                    select(RepositorySet.id).where(
                        RepositorySet.owner_id == rs.owner_id, RepositorySet.name == body.name,
                    ),
                )
                if dupe_id is not None:
                    raise ResponseError(
                        DatabaseConflict(
                            detail=f"there is an existing reposet {dupe_id} with the same name",
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
                        DatabaseConflict(
                            detail="there is an existing reposet with the same items",
                        ),
                    )
        finally:
            await prefix_loader
        prefixer = prefix_loader.result()
        items = prefixer.dereference_repositories(reposet_items_to_refs(rs.items))
        return model_response(
            RepositorySetWithName(
                name=rs.name,
                items=sorted(str(r) for r in items),
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
