from datetime import datetime, timezone
from sqlite3 import IntegrityError, OperationalError
import struct
from typing import Any, Dict, Optional

from aiohttp import web
from asyncpg import UniqueViolationError
import morcilla.core
from sqlalchemy import and_, delete, insert, select, update

from athenian.api.internal.account import get_user_account_status_from_request
from athenian.api.models.state.models import UserToken
from athenian.api.models.web import (
    BadRequestError,
    CreatedToken,
    CreateTokenRequest,
    DatabaseConflict,
    ForbiddenError,
    ListedToken,
    NotFoundError,
    PatchTokenRequest,
    ServerNotImplementedError,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response


def info_from_bearerAuth(token: str, required_scopes) -> Optional[Dict[str, Any]]:
    """
    Check and retrieve authentication information from a Bearer token.

    Returned value will be passed in 'token_info' parameter of your operation function, if there
    is one. 'sub' or 'uid' will be set in 'user' parameter of your operation function, if there
    is one. Should return None if auth is invalid or does not allow access to called API.

    The real work happens in Auth0._set_user().
    """
    return {"token": token, "method": "bearer"}


def info_from_apiKeyAuth(token: str, required_scopes) -> Optional[Dict[str, Any]]:
    """
    Check and retrieve authentication information from an API key.

    Returned value will be passed in 'token_info' parameter of your operation function, if there
    is one. 'sub' or 'uid' will be set in 'user' parameter of your operation function, if there
    is one. Should return None if auth is invalid or does not allow access to called API.

    The real work happens in Auth0._set_user().
    """
    return {"token": token, "method": "apikey"}


async def create_token(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create a new Personal Access Token for the current user and the specified account."""
    kms = request.app["kms"]
    if kms is None:
        raise ResponseError(ServerNotImplementedError(detail="Google KMS was not initialized."))
    model = CreateTokenRequest.from_dict(body)
    _check_token_name(model.name)
    async with request.sdb.connection() as conn:
        try:
            token = await conn.execute(
                insert(UserToken).values(
                    UserToken(name=model.name, account_id=model.account, user_id=request.uid)
                    .create_defaults()
                    .explode()
                )
            )
        except (UniqueViolationError, IntegrityError, OperationalError) as e:
            raise ResponseError(
                DatabaseConflict(
                    detail="Token '%s' already exists: %s: %s" % (model.name, type(e).__name__, e)
                ),
            ) from None
    plaintext = struct.pack("<q", token)
    cyphertext = await kms.encrypt(plaintext)
    return model_response(CreatedToken(id=token, token=cyphertext))


async def delete_token(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete a Personal Access Token belonging to the user."""
    async with request.sdb.connection() as conn:
        await _check_token_access(request, id, conn)
        await conn.execute(delete(UserToken).where(UserToken.id == id))
    return web.json_response({})


async def patch_token(request: AthenianWebRequest, id: int, body: dict) -> web.Response:
    """Change Personal Access Token's details."""
    model = PatchTokenRequest.from_dict(body)
    _check_token_name(model.name)
    async with request.sdb.connection() as conn:
        await _check_token_access(request, id, conn)
        try:
            await conn.execute(
                update(UserToken)
                .where(UserToken.id == id)
                .values(
                    {
                        UserToken.name: model.name,
                        UserToken.updated_at: datetime.now(timezone.utc),
                    }
                )
            )
        except (UniqueViolationError, IntegrityError, OperationalError) as e:
            raise ResponseError(
                DatabaseConflict(
                    detail="Token '%s' already exists: %s: %s" % (model.name, type(e).__name__, e)
                ),
            ) from None
    return web.json_response({})


async def list_tokens(request: AthenianWebRequest, id: int) -> web.Response:
    """List Personal Access Tokens of the user in the account."""
    sqlite = request.sdb.url.dialect == "sqlite"
    async with request.sdb.connection() as conn:
        await get_user_account_status_from_request(request, id)
        rows = await conn.fetch_all(
            select([UserToken.id, UserToken.name, UserToken.last_used_at]).where(
                and_(UserToken.user_id == request.uid, UserToken.account_id == id)
            )
        )
        model = []
        for row in rows:
            last_used = row[UserToken.last_used_at.name]
            if sqlite:
                last_used = last_used.replace(tzinfo=timezone.utc)
            model.append(
                ListedToken(
                    id=row[UserToken.id.name], name=row[UserToken.name.name], last_used=last_used
                )
            )
    return model_response(model)


async def _check_token_access(
    request: AthenianWebRequest, id: int, conn: morcilla.core.Connection
) -> None:
    token = await conn.fetch_one(select([UserToken]).where(UserToken.id == id))
    if token is None:
        raise ResponseError(NotFoundError(detail="Token %d was not found" % id))
    try:
        await get_user_account_status_from_request(request, token[UserToken.account_id.name])
    except ResponseError:
        # do not leak the account number
        raise ResponseError(NotFoundError(detail="Token %d was not found" % id)) from None
    if token[UserToken.user_id.name] != request.uid:
        raise ResponseError(ForbiddenError(detail="Token %d belongs to a different user" % id))


def _check_token_name(name: str) -> None:
    if len(name) == 0 or len(name) > 256:
        raise ResponseError(BadRequestError(detail='Token name is invalid: "%s"' % name))
