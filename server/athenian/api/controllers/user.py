from typing import Union

import databases.core
from sqlalchemy import and_, select

from athenian.api.models.state.models import UserAccount
from athenian.api.models.web import ForbiddenError
from athenian.api.response import ResponseError


async def is_admin(sdb: Union[databases.Database, databases.core.Connection],
                   user: str, account: int) -> bool:
    """Check if the user is an admin of the account."""
    status = await sdb.fetch_one(select([UserAccount.is_admin]).where(
        and_(UserAccount.user_id == user, UserAccount.account_id == account)))
    if status is None:
        raise ResponseError(ForbiddenError(
            detail="User %s does not belong to the account %d" % (user, account)))
    return status[UserAccount.is_admin.key]
