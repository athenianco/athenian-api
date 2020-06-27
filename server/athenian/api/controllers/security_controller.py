from datetime import datetime, timezone
import random
from typing import Any, Dict, Optional

from aiohttp import web
import databases.core
from sqlalchemy import and_, delete, insert, select, update

from athenian.api.controllers.account import get_user_account_status
from athenian.api.models.state.models import AccountToken
from athenian.api.models.web import BadRequestError, ForbiddenError, NotFoundError
from athenian.api.models.web.create_token_request import CreateTokenRequest
from athenian.api.models.web.created_token import CreatedToken
from athenian.api.models.web.generic_error import ServerNotImplementedError
from athenian.api.models.web.listed_token import ListedToken
from athenian.api.models.web.patch_token_request import PatchTokenRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


def info_from_bearerAuth(token: str) -> Optional[Dict[str, Any]]:
    """
    Check and retrieve authentication information from a Bearer token.

    Returned value will be passed in 'token_info' parameter of your operation function, if there
    is one. 'sub' or 'uid' will be set in 'user' parameter of your operation function, if there
    is one. Should return None if auth is invalid or does not allow access to called API.

    The real work happens in Auth0._set_user().
    """
    return {"token": token}


def info_from_apiKeyAuth(token: str) -> Optional[Dict[str, Any]]:
    """
    Check and retrieve authentication information from an API key.

    Returned value will be passed in 'token_info' parameter of your operation function, if there
    is one. 'sub' or 'uid' will be set in 'user' parameter of your operation function, if there
    is one. Should return None if auth is invalid or does not allow access to called API.

    The real work happens in Auth0._set_user().
    """
    return {"token": token}


async def create_token(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create a new Personal Access Token for the current user and the specified account."""
    kms = request.app["kms"]
    if kms is None:
        raise ResponseError(ServerNotImplementedError(detail="Google KMS was not initialized."))
    model = CreateTokenRequest.from_dict(body)  # type: CreateTokenRequest
    _check_token_name(model.name)
    async with request.sdb.connection() as conn:
        await get_user_account_status(request.uid, model.account, conn, request.cache)
        token = await conn.execute(insert(AccountToken).values(AccountToken(
            name=model.name, account_id=model.account, user_id=request.uid,
        ).create_defaults().explode()))
    plaintext = "%016x|%d|athenian" % (random.randint(0, 1 << 64 - 1), token)
    cyphertext = await kms.encrypt(plaintext)
    return model_response(CreatedToken(id=token, token=cyphertext))


async def delete_token(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete a Personal Access Token belonging to the user."""
    async with request.sdb.connection() as conn:
        await _check_token_access(request, id, conn)
        await conn.execute(delete(AccountToken).where(AccountToken.id == id))
    return web.json_response({})


async def patch_token(request: AthenianWebRequest, id: int, body: dict) -> web.Response:
    """Change Personal Access Token's details."""
    model = PatchTokenRequest.from_dict(body)  # type: PatchTokenRequest
    _check_token_name(model.name)
    async with request.sdb.connection() as conn:
        await _check_token_access(request, id, conn)
        await conn.execute(update(AccountToken).where(AccountToken.id == id).values({
            AccountToken.name: model.name,
            AccountToken.updated_at: datetime.now(timezone.utc),
        }))
    return web.json_response({})


async def list_tokens(request: AthenianWebRequest, id: int) -> web.Response:
    """List Personal Access Tokens of the user in the account."""
    async with request.sdb.connection() as conn:
        await get_user_account_status(request.uid, id, conn, request.cache)
        rows = await conn.fetch_all(
            select([AccountToken.id, AccountToken.name, AccountToken.last_used_at])
            .where(and_(AccountToken.user_id == request.uid,
                        AccountToken.account_id == id)))
        model = [ListedToken(id=row[AccountToken.id.key],
                             name=row[AccountToken.name.key],
                             last_used=row[AccountToken.last_used_at.key])
                 for row in rows]
    return model_response(model)


async def _check_token_access(request: AthenianWebRequest,
                              id: int,
                              conn: databases.core.Connection) -> None:
    token = await conn.fetch_one(select([AccountToken]).where(AccountToken.id == id))
    if token is None:
        raise ResponseError(NotFoundError(detail="Token %d was not found" % id))
    try:
        await get_user_account_status(
            request.uid, token[AccountToken.account_id.key], conn, request.cache)
    except ResponseError:
        # do not leak the account number
        raise ResponseError(NotFoundError(detail="Token %d was not found" % id)) from None
    if token[AccountToken.user_id.key] != request.uid:
        raise ResponseError(ForbiddenError(detail="Token %d belongs to a different user" % id))


def _check_token_name(name: str) -> None:
    if len(name) == 0 or len(name) > 256:
        raise ResponseError(BadRequestError(detail='Token name is invalid: "%s"' % name))
