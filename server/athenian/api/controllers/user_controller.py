from aiohttp import web
from sqlalchemy import select

from athenian.api.controllers.response import response, ResponseError
from athenian.api.models.state.models import UserAccount
from athenian.api.models.web import ForbiddenError
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
