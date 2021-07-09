import asyncio
import pickle
from typing import Optional

from aiohttp import web
import aiomcache
import sentry_sdk
from sqlalchemy import and_, delete, select, update

from athenian.api.async_utils import gather
from athenian.api.cache import cached, max_exptime
from athenian.api.controllers.account import get_account_organizations, get_user_account_status
from athenian.api.controllers.jira import get_jira_id
from athenian.api.controllers.user import load_user_accounts
from athenian.api.db import DatabaseLike
from athenian.api.models.metadata.jira import Installation as JIRAInstallation, \
    Project as JIRAProject
from athenian.api.models.state.models import AccountFeature, Feature, FeatureComponent, God, \
    UserAccount
from athenian.api.models.web import Account, AccountUserChangeRequest, ForbiddenError, \
    JIRAInstallation as WebJIRAInstallation, NotFoundError, Organization, ProductFeature, \
    UserChangeStatus
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


async def get_user(request: AthenianWebRequest) -> web.Response:
    """Return details about the current user."""
    user = await request.user()
    user.accounts = await load_user_accounts(
        user.id, request.sdb, request.mdb, request.rdb, request.cache)
    if (god_id := getattr(request, "god_id", request.uid)) != request.uid:
        user.impersonated_by = god_id
    return model_response(user)


async def get_account_details(request: AthenianWebRequest, id: int) -> web.Response:
    """Return the members and installed GitHub and JIRA organizations of the account."""
    user_id = request.uid
    users = await request.sdb.fetch_all(select([UserAccount]).where(UserAccount.account_id == id))
    if len(users) == 0:
        raise ResponseError(NotFoundError(detail="Account %d does not exist." % id))
    for user in users:
        if user[UserAccount.user_id.name] == user_id:
            break
    else:
        raise ResponseError(ForbiddenError(
            detail="User %s is not allowed to access account %d" % (user_id, id)))
    admins = []
    regulars = []
    for user in users:
        role = admins if user[UserAccount.is_admin.name] else regulars
        role.append(user[UserAccount.user_id.name])
    tasks = [
        request.app["auth"].get_users(regulars + admins),
        get_account_organizations(id, request.sdb, request.mdb, request.cache),
        _get_account_jira(id, request.sdb, request.mdb, request.cache),
    ]
    with sentry_sdk.start_span(op="fetch"):
        users, orgs, jira = await asyncio.gather(*tasks, return_exceptions=True)
    # not orgs! The account is probably being installed.
    # not jira! It raises ResponseError if no JIRA installation exists.
    if isinstance(users, Exception):
        raise users from None
    if isinstance(orgs, ResponseError):
        orgs = []
    elif isinstance(orgs, Exception):
        raise orgs from None
    if isinstance(jira, ResponseError):
        jira = None
    elif isinstance(jira, Exception):
        raise jira from None
    account = Account(regulars=[users[k] for k in regulars if k in users],
                      admins=[users[k] for k in admins if k in users],
                      organizations=[Organization(name=org.name,
                                                  avatar_url=org.avatar_url,
                                                  login=org.login)
                                     for org in orgs],
                      jira=jira)
    return model_response(account)


@cached(
    exptime=max_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda account, **_: (account,),
    refresh_on_access=True,
)
async def _get_account_jira(account: int,
                            sdb: DatabaseLike,
                            mdb: DatabaseLike,
                            cache: Optional[aiomcache.Client]) -> WebJIRAInstallation:
    jira_id = await get_jira_id(account, sdb, cache)
    tasks = [
        mdb.fetch_all(select([JIRAProject.key])
                      .where(and_(JIRAProject.acc_id == jira_id,
                                  JIRAProject.is_deleted.is_(False)))
                      .order_by(JIRAProject.key)),
        mdb.fetch_val(select([JIRAInstallation.base_url])
                      .where(JIRAInstallation.acc_id == jira_id)),
    ]
    projects, base_url = await gather(*tasks)
    return WebJIRAInstallation(url=base_url, projects=[r[0] for r in projects])


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
        features = {row[0]: [row[1], row[2]] for row in features}
        for k, v in account_features.items():
            try:
                fk = features[k]
            except KeyError:
                continue
            if v is not None:
                if isinstance(v, dict):
                    for pk, pv in v.items():
                        fk[1][pk] = pv
                else:
                    fk[1] = v
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
    user = await request.app["auth"].get_user(id or user_id)
    user.accounts = await load_user_accounts(
        user.id, request.sdb, request.mdb, request.rdb, request.cache)
    return model_response(user)


async def change_user(request: AthenianWebRequest, body: dict) -> web.Response:
    """Change the status of an account member: regular, admin, or banished (deleted)."""
    aucr = AccountUserChangeRequest.from_dict(body)
    async with request.sdb.connection() as conn:
        if not await get_user_account_status(request.uid, aucr.account, conn, request.cache):
            return ResponseError(ForbiddenError(
                detail="User %s is not an admin of account %d" % (request.uid, aucr.account)),
            ).response
        users = await request.sdb.fetch_all(
            select([UserAccount]).where(UserAccount.account_id == aucr.account))
        for user in users:
            if user[UserAccount.user_id.name] == aucr.user:
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
            if user[UserAccount.is_admin.name]:
                admins.add(user[UserAccount.user_id.name])
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
    return await get_account_details(request, aucr.account)
