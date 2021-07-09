import base64
import binascii
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
import logging
import marshal
import os
from random import randint
from sqlite3 import IntegrityError, OperationalError
import struct
from typing import Callable, Coroutine, Optional, Tuple

from aiohttp import web
import aiomcache
import aiosqlite.core
from asyncpg import IntegrityConstraintViolationError
import databases.core
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
from sqlalchemy import and_, delete, func, insert, select, update

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.auth import Auth0, disable_default_user
from athenian.api.cache import cached
from athenian.api.controllers.account import fetch_github_installation_progress, \
    generate_jira_invitation_link, \
    get_metadata_account_ids, get_user_account_status, jira_url_template
from athenian.api.controllers.ffx import decrypt, encrypt
from athenian.api.controllers.reposet import load_account_reposets
from athenian.api.controllers.user import load_user_accounts
from athenian.api.db import DatabaseLike, FastConnection, ParallelDatabase
from athenian.api.defer import defer
from athenian.api.models.metadata.github import NodeUser, OrganizationMember
from athenian.api.models.state.models import Account, AccountFeature, Feature, FeatureComponent, \
    Invitation, RepositorySet, UserAccount
from athenian.api.models.web import BadRequestError, ForbiddenError, GenericError, \
    NotFoundError, User
from athenian.api.models.web.generic_error import DatabaseConflict, TooManyRequestsError
from athenian.api.models.web.installation_progress import InstallationProgress
from athenian.api.models.web.invitation_check_result import InvitationCheckResult
from athenian.api.models.web.invitation_link import InvitationLink
from athenian.api.models.web.invited_user import InvitedUser
from athenian.api.models.web.table_fetching_progress import TableFetchingProgress
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError
from athenian.api.tracing import sentry_span

admin_backdoor = (1 << 24) - 1
url_prefix = os.getenv("ATHENIAN_INVITATION_URL_PREFIX")
# we add 4 hours to compensate the installation time
trial_period = timedelta(days=14, hours=4)


def validate_env():
    """Check that the required global parameters are set."""
    if Auth0.KEY is None:
        raise EnvironmentError("ATHENIAN_INVITATION_KEY environment variable must be set")
    if url_prefix is None:
        raise EnvironmentError("ATHENIAN_INVITATION_URL_PREFIX environment variable must be set")
    if jira_url_template is None:
        raise EnvironmentError(
            "ATHENIAN_JIRA_INSTALLATION_URL_TEMPLATE environment variable must be set")


async def gen_invitation(request: AthenianWebRequest, id: int) -> web.Response:
    """Generate a new regular member invitation URL."""
    async with request.sdb.connection() as sdb_conn:
        await get_user_account_status(request.uid, id, sdb_conn, request.cache)
        existing = await sdb_conn.fetch_one(
            select([Invitation.id, Invitation.salt])
            .where(and_(Invitation.is_active, Invitation.account_id == id)))
        if existing is not None:
            invitation_id = existing[Invitation.id.name]
            salt = existing[Invitation.salt.name]
        else:
            # create a new invitation
            salt = randint(0, (1 << 16) - 1)  # 0:65535 - 2 bytes
            inv = Invitation(salt=salt, account_id=id, created_by=request.uid).create_defaults()
            invitation_id = await sdb_conn.execute(insert(Invitation).values(inv.explode()))
        slug = encode_slug(invitation_id, salt, request.app["auth"].key)
        model = InvitationLink(url=url_prefix + slug)
        return model_response(model)


async def _check_admin_access(uid: str, account: int, sdb_conn: databases.core.Connection):
    status = await sdb_conn.fetch_one(
        select([UserAccount.is_admin])
        .where(and_(UserAccount.user_id == uid, UserAccount.account_id == account)))
    if status is None:
        raise ResponseError(NotFoundError(
            detail="User %s is not in the account %d" % (uid, account)))
    if not status[UserAccount.is_admin.name]:
        raise ResponseError(ForbiddenError(
            detail="User %s is not an admin of the account %d" % (uid, account)))


def encode_slug(iid: int, salt: int, key: str) -> str:
    """Encode an invitation ID and some extra data to 8 chars."""
    part1 = struct.pack("!H", salt)  # 2 bytes
    part2 = struct.pack("!I", iid)[1:]  # 3 bytes
    binseq = part1 + part2  # 5 bytes, 10 hex chars
    encseq = encrypt(binseq, key.encode())  # encrypted 5 bytes, 10 hex chars
    finseq = base64.b32encode(bytes.fromhex(encseq)).lower().decode()  # 8 base32 chars
    finseq = finseq.replace("o", "8").replace("l", "9")
    return finseq


def decode_slug(slug: str, key: str) -> (int, int):
    """Decode an invitation ID and some extra data from 8 chars."""
    assert len(slug) == 8
    assert isinstance(slug, str)
    b32 = slug.replace("8", "o").replace("9", "l").upper().encode()
    x = base64.b32decode(b32).hex()
    x = decrypt(x, key.encode())
    salt = struct.unpack("!H", x[:2])[0]
    iid = struct.unpack("!I", b"\x00" + x[2:])[0]
    return iid, salt


@disable_default_user
async def accept_invitation(request: AthenianWebRequest, body: dict) -> web.Response:
    """Accept the membership invitation."""
    if getattr(request, "god_id", request.uid) != request.uid:
        raise ResponseError(ForbiddenError(
            detail="You must not be an active god to accept an invitation."))

    def bad_req():
        raise ResponseError(BadRequestError(detail="Invalid invitation URL")) from None

    sdb = request.sdb
    url = InvitationLink.from_dict(body).url
    if not url.startswith(url_prefix):
        bad_req()
    x = url[len(url_prefix):].strip("/")
    if len(x) != 8:
        bad_req()
    try:
        iid, salt = decode_slug(x, request.app["auth"].key)
    except binascii.Error:
        bad_req()
    async with sdb.connection() as conn:
        try:
            async with conn.transaction():
                acc_id, user = await _accept_invitation(
                    iid, salt, request, conn, sdb, request.mdb, request.rdb, request.cache)
        except (IntegrityConstraintViolationError, IntegrityError, OperationalError) as e:
            raise ResponseError(DatabaseConflict(detail=str(e)))
    return model_response(InvitedUser(account=acc_id, user=user))


async def _accept_invitation(iid: int,
                             salt: int,
                             request: AthenianWebRequest,
                             sdb_transaction: FastConnection,
                             sdb: ParallelDatabase,
                             mdb: ParallelDatabase,
                             rdb: ParallelDatabase,
                             cache: Optional[aiomcache.Client],
                             ) -> Tuple[int, User]:
    """
    We need both `sdb_conn` and `sdb` because `sdb` is required in the deferred code outside of
    the transaction.
    You should work with `sdb_conn` in the code that blocks the request flow and with `sdb`
    in the code that defer()-s.
    """  # noqa: D
    log = logging.getLogger(metadata.__package__)
    inv = await sdb_transaction.fetch_one(
        select([Invitation.account_id, Invitation.accepted, Invitation.is_active])
        .where(and_(Invitation.id == iid, Invitation.salt == salt)))
    if inv is None:
        raise ResponseError(NotFoundError(detail="Invitation was not found."))
    if not inv[Invitation.is_active.name]:
        raise ResponseError(ForbiddenError(detail="This invitation is disabled."))
    acc_id = inv[Invitation.account_id.name]
    is_admin = acc_id == admin_backdoor
    slack = request.app["slack"]  # type: SlackWebClient
    if is_admin:
        other_accounts = await sdb_transaction.fetch_all(
            select([UserAccount.account_id])
            .where(and_(UserAccount.user_id == request.uid,
                        UserAccount.is_admin)))
        if other_accounts:
            other_accounts = {row[0] for row in other_accounts}
            installed_accounts = await sdb_transaction.fetch_all(
                select([RepositorySet.owner_id])
                .where(and_(RepositorySet.owner_id.in_(other_accounts),
                            RepositorySet.name == RepositorySet.ALL,
                            RepositorySet.precomputed)))
            installed = {row[0] for row in installed_accounts}
            if other_accounts - installed:
                raise ResponseError(TooManyRequestsError(
                    type="/errors/DuplicateAccountRegistrationError",
                    detail="You cannot accept new admin invitations until your account's "
                           "installation finishes."))
        # create a new account for the admin user
        acc_id = await create_new_account(sdb_transaction, request.app["auth"].key)
        if acc_id >= admin_backdoor:
            await sdb_transaction.execute(delete(Account).where(Account.id == acc_id))
            raise ResponseError(GenericError(
                type="/errors/LockedError",
                title=HTTPStatus.LOCKED.phrase,
                status=HTTPStatus.LOCKED,
                detail="Invitation was not found."))
        log.info("Created new account %d", acc_id)
        if slack is not None:
            async def report_new_account_to_slack():
                jira_link = await generate_jira_invitation_link(acc_id, sdb)
                await slack.post("new_account.jinja2",
                                 user=await request.user(),
                                 account=acc_id,
                                 jira_link=jira_link)

            await defer(report_new_account_to_slack(), "report_new_account_to_slack")
        status = None
    else:
        status = await sdb_transaction.fetch_one(
            select([UserAccount.is_admin])
            .where(and_(UserAccount.user_id == request.uid,
                        UserAccount.account_id == acc_id)))
    user = None
    if status is None:
        if is_admin or (user := await _check_user_org_membership(
                request, acc_id, sdb_transaction, log)) is None:
            user = await request.user()
        # create the user<>account record
        await sdb_transaction.execute(insert(UserAccount).values(UserAccount(
            user_id=request.uid,
            account_id=acc_id,
            is_admin=is_admin,
        ).create_defaults().explode(with_primary_keys=True)))
        log.info("Assigned user %s to account %d (admin: %s)", request.uid, acc_id, is_admin)
        if slack is not None:
            async def report_new_user_to_slack():
                repos = await request.sdb.fetch_val(select([RepositorySet.items]).where(and_(
                    RepositorySet.owner_id == acc_id, RepositorySet.name == RepositorySet.ALL)))
                if repos is not None:
                    prefixes = {r.split("/", 2)[1] for r in repos}
                else:
                    prefixes = {"N/A"}
                await slack.post("new_user.jinja2", user=user, account=acc_id, prefixes=prefixes)

            await defer(report_new_user_to_slack(), "report_new_user_to_slack")
        values = {Invitation.accepted.name: inv[Invitation.accepted.name] + 1}
        await sdb_transaction.execute(update(Invitation)
                                      .where(Invitation.id == iid).values(values))
    if user is None:
        user = await request.user()
    user.accounts = await load_user_accounts(user.id, sdb_transaction, mdb, rdb, cache)
    return acc_id, user


async def _check_user_org_membership(request: AthenianWebRequest,
                                     acc_id: int,
                                     sdb_conn: DatabaseLike,
                                     log: logging.Logger,
                                     ) -> Optional[User]:
    user_org_membership_check_row = await sdb_conn.fetch_one(
        select([Feature.id, Feature.enabled])
        .where(and_(Feature.name == Feature.USER_ORG_MEMBERSHIP_CHECK,
                    Feature.component == FeatureComponent.server)))
    if user_org_membership_check_row is not None:
        user_org_membership_check_feature_id = user_org_membership_check_row[Feature.id.name]
        global_enabled = user_org_membership_check_row[Feature.enabled.name]
        enabled = await sdb_conn.fetch_val(
            select([AccountFeature.enabled]).where(and_(
                AccountFeature.account_id == acc_id,
                AccountFeature.feature_id == user_org_membership_check_feature_id,
            )))
        if (enabled is None and not global_enabled) or (enabled is not None and not enabled):
            return None

    mdb = request.mdb
    cache = request.cache

    # check whether the user is a member of the GitHub org
    async def load_org_members():
        meta_ids = await get_metadata_account_ids(acc_id, sdb_conn, cache)
        user_node_ids = [
            r[0] for r in await mdb.fetch_all(
                select([OrganizationMember.child_id])
                .where(OrganizationMember.acc_id.in_(meta_ids)))
        ]
        log.debug("Discovered %d organization members", len(user_node_ids))
        return meta_ids, user_node_ids

    user, (meta_ids, user_node_ids) = await gather(request.user(), load_org_members())
    user_node_id = await mdb.fetch_val(select([NodeUser.node_id])
                                       .where(and_(NodeUser.acc_id.in_(meta_ids),
                                                   NodeUser.login == user.login)))
    if user_node_id not in user_node_ids:
        raise ResponseError(ForbiddenError(
            detail="User %s does not belong to the GitHub organization." % request.uid))
    return user


async def create_new_account(conn: DatabaseLike, secret: str) -> int:
    """Create a new account."""
    if isinstance(conn, databases.Database):
        slow = conn.url.dialect == "sqlite"
    else:
        slow = isinstance(conn.raw_connection, aiosqlite.core.Connection)
    if slow:
        return await _create_new_account_slow(conn, secret)
    return await _create_new_account_fast(conn, secret)


async def _create_new_account_fast(conn: DatabaseLike, secret: str) -> int:
    """Create a new account.

    Should be used for PostgreSQL.
    """
    account_id = await conn.execute(
        insert(Account).values(Account(secret_salt=0,
                                       secret=Account.missing_secret,
                                       expires_at=datetime.now(timezone.utc) + trial_period)
                               .create_defaults().explode()))
    salt, secret = _generate_account_secret(account_id, secret)
    await conn.execute(update(Account).where(Account.id == account_id).values({
        Account.secret_salt: salt,
        Account.secret: secret,
    }))
    return account_id


async def _create_new_account_slow(conn: DatabaseLike, secret: str) -> int:
    """Create a new account without relying on autoincrement.

    SQLite does not allow resetting the primary key sequence, so we have to increment the ID
    by hand.
    """
    acc = Account(secret_salt=0,
                  secret=Account.missing_secret,
                  expires_at=datetime.now() + trial_period).create_defaults()
    max_id = (await conn.fetch_one(select([func.max(Account.id)])
                                   .where(Account.id < admin_backdoor)))[0] or 0
    acc.id = max_id + 1
    acc_id = await conn.execute(insert(Account).values(acc.explode(with_primary_keys=True)))
    salt, secret = _generate_account_secret(acc_id, secret)
    await conn.execute(update(Account).where(Account.id == acc_id).values({
        Account.secret_salt: salt,
        Account.secret: secret,
    }))
    return acc_id


async def check_invitation(request: AthenianWebRequest, body: dict) -> web.Response:
    """Given an invitation URL, get its type (admin or regular account member) and find whether \
    it is enabled or disabled."""
    url = InvitationLink.from_dict(body).url
    result = InvitationCheckResult(valid=False)
    if not url.startswith(url_prefix):
        return model_response(result)
    x = url[len(url_prefix):].strip("/")
    if len(x) != 8:
        return model_response(result)
    try:
        iid, salt = decode_slug(x, request.app["auth"].key)
    except binascii.Error:
        return model_response(result)
    inv = await request.sdb.fetch_one(
        select([Invitation.account_id, Invitation.is_active])
        .where(and_(Invitation.id == iid, Invitation.salt == salt)))
    if inv is None:
        return model_response(result)
    result.valid = True
    result.active = inv[Invitation.is_active.name]
    types = [InvitationCheckResult.INVITATION_TYPE_REGULAR,
             InvitationCheckResult.INVITATION_TYPE_ADMIN]
    result.type = types[inv[Invitation.account_id.name] == admin_backdoor]
    return model_response(result)


@sentry_span
async def _append_precomputed_progress(model: InstallationProgress,
                                       account: int,
                                       uid: str,
                                       login: Callable[[], Coroutine[None, None, str]],
                                       sdb: DatabaseLike,
                                       mdb: databases.Database,
                                       cache: Optional[aiomcache.Client],
                                       slack: Optional[SlackWebClient]) -> None:
    assert model.finished_date is not None
    try:
        reposets = await load_account_reposets(
            account, login,
            [RepositorySet.name, RepositorySet.precomputed, RepositorySet.created_at],
            sdb, mdb, cache, slack, check_progress=False)
    except ResponseError:
        # not ready yet
        model.finished_date = None
        return
    precomputed = False
    created = None
    for reposet in reposets:
        if reposet[RepositorySet.name.name] == RepositorySet.ALL:
            precomputed = reposet[RepositorySet.precomputed.name]
            created = reposet[RepositorySet.created_at.name].replace(tzinfo=timezone.utc)
            break
    if slack is not None and not precomputed and model.finished_date is not None \
            and datetime.now(timezone.utc) - model.finished_date > timedelta(hours=2) \
            and datetime.now(timezone.utc) - created > timedelta(hours=2):
        await _notify_precomputed_failure(slack, uid, account, model, created, cache)
    model.tables.append(TableFetchingProgress(
        name="precomputed", fetched=int(precomputed), total=1))
    if not precomputed:
        model.finished_date = None


@cached(
    exptime=2 * 3600,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda account, **_: (account,),
    refresh_on_access=True,
)
async def _notify_precomputed_failure(slack: Optional[SlackWebClient],
                                      uid: str,
                                      account: int,
                                      model: InstallationProgress,
                                      created: datetime,
                                      cache: Optional[aiomcache.Client]) -> None:
    await slack.post(
        "precomputed_failure.jinja2", uid=uid, account=account, model=model, created=created)


async def eval_invitation_progress(request: AthenianWebRequest, id: int) -> web.Response:
    """Return the current Athenian GitHub app installation progress."""
    await get_user_account_status(request.uid, id, request.sdb, request.cache)
    async with request.mdb.connection() as mdb_conn:
        model = await fetch_github_installation_progress(id, request.sdb, mdb_conn, request.cache)

    async def login_loader() -> str:
        return (await request.user()).login

    if model.finished_date is not None:
        await _append_precomputed_progress(
            model, id, request.uid, login_loader, request.sdb, request.mdb,
            request.cache, request.app["slack"])
    return model_response(model)


def _generate_account_secret(account_id: int, key: str) -> Tuple[int, str]:
    salt = randint(0, (1 << 16) - 1)  # 0:65535 - 2 bytes
    secret = encode_slug(account_id, salt, key)
    return salt, secret


async def gen_jira_link(request: AthenianWebRequest, id: int) -> web.Response:
    """Generate JIRA integration installation link."""
    account_id = id
    async with request.sdb.connection() as sdb_conn:
        await _check_admin_access(request.uid, account_id, sdb_conn)
        url = await generate_jira_invitation_link(account_id, sdb_conn)
        model = InvitationLink(url=url)
        return model_response(model)
