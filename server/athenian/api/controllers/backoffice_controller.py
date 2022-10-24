from aiohttp import web
from sqlalchemy import select, update

from athenian.api.internal.account import only_god
from athenian.api.internal.user import load_user_accounts
from athenian.api.models.state.models import God, UserAccount
from athenian.api.models.web import InvalidRequestError, NotFoundError, ResetRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response


@only_god
async def become_user(request: AthenianWebRequest, id: str = "") -> web.Response:
    """God mode ability to turn into any user. The current user must be marked internally as \
    a super admin."""
    user_id = request.god_id
    async with request.sdb.connection() as conn:
        if (
            id
            and (await conn.fetch_one(select([UserAccount]).where(UserAccount.user_id == id)))
            is None
        ):
            raise ResponseError(NotFoundError(detail="User %s does not exist" % id))
        god = await conn.fetch_one(select([God]).where(God.user_id == user_id))
        god = God(**god).refresh()
        god.mapped_id = id or None
        await conn.execute(update(God).where(God.user_id == user_id).values(god.explode()))
    user = await request.app["auth"].get_user(id or user_id)
    user.accounts = await load_user_accounts(
        user.id,
        getattr(request, "god_id", user.id),
        request.sdb,
        request.mdb,
        request.rdb,
        request.app["slack"],
        request.user,
        request.cache,
    )
    return model_response(user)


@only_god
async def reset_account(request: AthenianWebRequest, body: dict) -> web.Response:
    """Clear the selected tables in precomputed DB, drop the related caches."""
    try:
        request_model = ResetRequest.from_dict(body)
    except ValueError as e:
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    _ = request_model
    raise NotImplementedError
