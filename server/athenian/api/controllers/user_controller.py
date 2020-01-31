from aiohttp import web
from sqlalchemy import select, update

from athenian.api.controllers.response import response, ResponseError
from athenian.api.models.state.models import God, UserAccount
from athenian.api.models.web import ForbiddenError, NotFoundError
from athenian.api.models.web.account import Account
from athenian.api.request import AthenianWebRequest


async def get_user(request: AthenianWebRequest) -> web.Response:
    """Return details about the current user."""
    user = await (await request.user()).load_accounts(request.sdb)
    return response(user)


async def get_account(request: AthenianWebRequest, id: int) -> web.Response:
    """Return details about the current account."""
    user_id = request.uid
    users = await request.sdb.fetch_all(select([UserAccount]).where(UserAccount.account_id == id))
    for user in users:
        if user[UserAccount.user_id.key] == user_id:
            break
    else:
        return ResponseError(ForbiddenError(
            detail="User %s is not allowed to access account %d" % (user_id, id))).response
    admins = []
    regulars = []
    for user in users:
        role = admins if user[UserAccount.is_admin.key] else regulars
        role.append(user[UserAccount.user_id.key])
    users = await request.auth.get_users(regulars + admins)
    account = Account(regulars=[users[k] for k in regulars if k in users],
                      admins=[users[k] for k in admins if k in users])
    return response(account)


async def become_user(request: AthenianWebRequest, id: str = "") -> web.Response:
    """God mode ability to turn into any user. The current user must be marked internally as \
    a super admin."""
    user_id = getattr(request, "god_id", None)
    if user_id is None:
        return ResponseError(ForbiddenError(
            detail="User %s is not allowed to mutate" % user_id)).response
    async with request.sdb.connection() as conn:
        if id and (await conn.fetch_one(
                select([UserAccount]).where(UserAccount.user_id == id))) is None:
            return ResponseError(NotFoundError(detail="User %s does not exist" % id)).response
        god = await conn.fetch_one(select([God]).where(God.user_id == user_id))
        god = God(**god).refresh()
        god.mapped_id = id or None
        await conn.execute(update(God).where(God.user_id == user_id).values(god.explode()))
    user = await (await request.auth.get_user(id or user_id)).load_accounts(request.sdb)
    return response(user)
