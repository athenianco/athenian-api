from aiohttp import web
from sqlalchemy import and_, delete, select, update

from athenian.api.controllers.account import get_user_account_status
from athenian.api.models.state.models import AccountFeature, Feature, FeatureComponent, God, \
    UserAccount
from athenian.api.models.web import ForbiddenError, NotFoundError
from athenian.api.models.web.account import Account
from athenian.api.models.web.account_user_change_request import AccountUserChangeRequest, \
    UserChangeStatus
from athenian.api.models.web.product_feature import ProductFeature
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


async def get_user(request: AthenianWebRequest) -> web.Response:
    """Return details about the current user."""
    user = await (await request.user()).load_accounts(request.sdb)
    return model_response(user)


async def get_account_members(request: AthenianWebRequest, id: int) -> web.Response:
    """Return the members of the account."""
    user_id = request.uid
    users = await request.sdb.fetch_all(select([UserAccount]).where(UserAccount.account_id == id))
    if len(users) == 0:
        raise ResponseError(NotFoundError(detail="Account %d does not exist." % id))
    for user in users:
        if user[UserAccount.user_id.key] == user_id:
            break
    else:
        raise ResponseError(ForbiddenError(
            detail="User %s is not allowed to access account %d" % (user_id, id)))
    admins = []
    regulars = []
    for user in users:
        role = admins if user[UserAccount.is_admin.key] else regulars
        role.append(user[UserAccount.user_id.key])
    users = await request.app["auth"].get_users(regulars + admins)
    account = Account(regulars=[users[k] for k in regulars if k in users],
                      admins=[users[k] for k in admins if k in users])
    return model_response(account)


async def get_account_features(request: AthenianWebRequest, id: int) -> web.Response:
    """Return enabled product features for the account."""
    async with request.sdb.connection() as conn:
        await get_user_account_status(request.uid, id, conn, request.cache)
        account_features = await conn.fetch_all(
            select([AccountFeature.feature_id, AccountFeature.parameters])
            .where(and_(AccountFeature.account_id == id, AccountFeature.enabled)))
        account_features = {row[0]: row[1] for row in account_features}
        features = await conn.fetch_all(
            select([Feature.id, Feature.name, Feature.default_parameters])
            .where(and_(Feature.id.in_(account_features),
                        Feature.component == FeatureComponent.webapp,
                        Feature.enabled)))
        features = {row[0]: (row[1], row[2]) for row in features}
        for k, v in account_features.items():
            if v is not None:
                for pk, pv in v.items():
                    features[k][1][pk] = pv
        models = [ProductFeature(*v) for k, v in sorted(features.items())]
        return model_response(models)


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
    user = await (await request.app["auth"].get_user(id or user_id)).load_accounts(request.sdb)
    return model_response(user)


async def change_user(request: AthenianWebRequest, body: dict) -> web.Response:
    """Change the status of an account member: regular, admin, or banished (deleted)."""
    aucr = AccountUserChangeRequest.from_dict(body)  # type: AccountUserChangeRequest
    async with request.sdb.connection() as conn:
        is_admin = await get_user_account_status(request.uid, aucr.account, conn, request.cache)
        if not is_admin:
            return ResponseError(ForbiddenError(
                detail="User %s is not an admin of account %d" % (request.uid, aucr.account)),
            ).response
        users = await request.sdb.fetch_all(
            select([UserAccount]).where(UserAccount.account_id == aucr.account))
        for user in users:
            if user[UserAccount.user_id.key] == aucr.user:
                break
        else:
            return ResponseError(NotFoundError(
                detail="User %s was not found in account %d" % (aucr.user, aucr.account)),
            ).response
        if len(users) == 1:
            return ResponseError(ForbiddenError(
                detail="Forbidden to edit the last user of account %d" % aucr.account),
            ).response
        admins = set()
        for user in users:
            if user[UserAccount.is_admin.key]:
                admins.add(user[UserAccount.user_id.key])
        if aucr.status == UserChangeStatus.REGULAR:
            if len(admins) == 1 and aucr.user in admins:
                return ResponseError(ForbiddenError(
                    detail="Forbidden to demote the last admin of account %d" % aucr.account),
                ).response
            await conn.execute(update(UserAccount)
                               .where(and_(UserAccount.user_id == aucr.user,
                                           UserAccount.account_id == aucr.account))
                               .values({UserAccount.is_admin: False}))
        elif aucr.status == UserChangeStatus.ADMIN:
            await conn.execute(update(UserAccount)
                               .where(and_(UserAccount.user_id == aucr.user,
                                           UserAccount.account_id == aucr.account))
                               .values({UserAccount.is_admin: True}))
        elif aucr.status == UserChangeStatus.BANISHED:
            if len(admins) == 1 and aucr.user in admins:
                return ResponseError(ForbiddenError(
                    detail="Forbidden to banish the last admin of account %d" % aucr.account),
                ).response
            await conn.execute(delete(UserAccount)
                               .where(and_(UserAccount.user_id == aucr.user,
                                           UserAccount.account_id == aucr.account)))
    return await get_account_members(request, aucr.account)
