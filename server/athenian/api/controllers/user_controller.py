import asyncio
import pickle
from typing import Optional

from aiohttp import web
import aiomcache
import morcilla
import sentry_sdk
from sqlalchemy import and_, delete, insert, select, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert

from athenian.api.async_utils import gather
from athenian.api.cache import cached, max_exptime
from athenian.api.controllers.account import get_account_organizations, get_user_account_status, \
    only_admin
from athenian.api.controllers.jira import get_jira_id
from athenian.api.controllers.user import load_user_accounts
from athenian.api.db import DatabaseLike
from athenian.api.models.metadata.jira import Installation as JIRAInstallation, \
    Project as JIRAProject
from athenian.api.models.state.models import Account as DBAccount, AccountFeature, Feature, \
    FeatureComponent, God, UserAccount
from athenian.api.models.web import Account, AccountUserChangeRequest, ForbiddenError, \
    InvalidRequestError, JIRAInstallation as WebJIRAInstallation, NotFoundError, Organization, \
    ProductFeature, UserChangeStatus
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError
from athenian.api.serialization import deserialize_datetime


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
        return await _get_account_features(conn, id)


async def _get_account_features(conn: morcilla.Connection, id: int) -> web.Response:
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


async def set_account_features(request: AthenianWebRequest, id: int, body: dict) -> web.Response:
    """Set account features if you are a god."""
    if getattr(request, "god_id", None) is None:  # no hasattr() please
        raise ResponseError(ForbiddenError(
            detail="User %s is not allowed to set features of accounts" % request.uid))
    features = [ProductFeature.from_dict(f) for f in body]
    async with request.sdb.connection() as conn:
        await get_user_account_status(request.uid, id, conn, request.cache)
        for i, feature in enumerate(features):
            if feature.name == "expires_at":
                try:
                    expires_at = deserialize_datetime(feature.parameters)
                except (TypeError, ValueError):
                    raise ResponseError(InvalidRequestError(
                        pointer=f".[{i}].parameters",
                        detail=f"Invalid datetime string: {feature.parameters}"))
                await conn.execute(update(DBAccount).where(DBAccount.id == id).values({
                    DBAccount.expires_at: expires_at,
                }))
            else:
                if not isinstance(feature.parameters, dict) or \
                        not isinstance(feature.parameters.get("enabled"), bool):
                    raise ResponseError(InvalidRequestError(
                        pointer=f".[{i}].parameters",
                        detail='Parameters must be {"enabled": true|false, ...}',
                    ))
                fid = await conn.fetch_val(select([Feature.id])
                                           .where(Feature.name == feature.name))
                if fid is None:
                    raise ResponseError(InvalidRequestError(
                        pointer=f".[{i}].name",
                        detail=f"Feature is not supported: {feature.name}"))
                if request.sdb.url.dialect == "postgresql":
                    query = postgres_insert(AccountFeature)
                    query = query.on_conflict_do_update(
                        constraint=AccountFeature.__table__.primary_key,
                        set_={
                            AccountFeature.enabled.name: query.excluded.enabled,
                            AccountFeature.parameters.name: query.excluded.parameters,
                        },
                    )
                else:
                    query = insert(AccountFeature).prefix_with("OR REPLACE")
                await conn.execute(query.values(AccountFeature(
                    account_id=id,
                    feature_id=fid,
                    enabled=feature.parameters["enabled"],
                    parameters=feature.parameters.get("parameters"),
                ).create_defaults().explode(with_primary_keys=True)))
        return await _get_account_features(conn, id)


async def become_user(request: AthenianWebRequest, id: str = "") -> web.Response:
    """God mode ability to turn into any user. The current user must be marked internally as \
    a super admin."""
    if (user_id := getattr(request, "god_id", None)) is None:
        raise ResponseError(ForbiddenError(
            detail="User %s is not allowed to mutate" % request.uid))
    async with request.sdb.connection() as conn:
        if id and (await conn.fetch_one(
                select([UserAccount]).where(UserAccount.user_id == id))) is None:
            raise ResponseError(NotFoundError(detail="User %s does not exist" % id))
        god = await conn.fetch_one(select([God]).where(God.user_id == user_id))
        god = God(**god).refresh()
        god.mapped_id = id or None
        await conn.execute(update(God).where(God.user_id == user_id).values(god.explode()))
    user = await request.app["auth"].get_user(id or user_id)
    user.accounts = await load_user_accounts(
        user.id, request.sdb, request.mdb, request.rdb, request.cache)
    return model_response(user)


@only_admin
async def change_user(request: AthenianWebRequest, body: dict) -> web.Response:
    """Change the status of an account member: regular, admin, or banished (deleted)."""
    aucr = AccountUserChangeRequest.from_dict(body)
    async with request.sdb.connection() as conn:
        users = await request.sdb.fetch_all(
            select([UserAccount]).where(UserAccount.account_id == aucr.account))
        for user in users:
            if user[UserAccount.user_id.name] == aucr.user:
                break
        else:
            raise ResponseError(NotFoundError(
                detail="User %s was not found in account %d" % (aucr.user, aucr.account)),
            )
        if len(users) == 1:
            raise ResponseError(ForbiddenError(
                detail="Forbidden to edit the last user of account %d" % aucr.account),
            )
        admins = set()
        for user in users:
            if user[UserAccount.is_admin.name]:
                admins.add(user[UserAccount.user_id.name])
        if aucr.status == UserChangeStatus.REGULAR:
            if len(admins) == 1 and aucr.user in admins:
                raise ResponseError(ForbiddenError(
                    detail="Forbidden to demote the last admin of account %d" % aucr.account),
                )
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
                raise ResponseError(ForbiddenError(
                    detail="Forbidden to banish the last admin of account %d" % aucr.account),
                )
            await conn.execute(delete(UserAccount)
                               .where(and_(UserAccount.user_id == aucr.user,
                                           UserAccount.account_id == aucr.account)))
    return await get_account_details(request, aucr.account)
