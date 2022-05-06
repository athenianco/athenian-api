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
from typing import Any, Callable, Coroutine, List, Mapping, Optional, Tuple

from aiohttp import web
import aiomcache
import aiosqlite.core
from asyncpg import IntegrityConstraintViolationError
from dateutil.relativedelta import relativedelta
import morcilla.core
import sentry_sdk
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
from sqlalchemy import and_, delete, func, insert, select, text, update

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.auth import Auth0, disable_default_user
from athenian.api.cache import cached, expires_header, middle_term_exptime
from athenian.api.controllers.account import fetch_github_installation_progress, \
    generate_jira_invitation_link, get_account_name_or_stub, get_metadata_account_ids_or_empty, \
    get_user_account_status_from_request, is_membership_check_enabled, jira_url_template, only_god
from athenian.api.controllers.ffx import decrypt, encrypt
from athenian.api.controllers.jira import fetch_jira_installation_progress
from athenian.api.controllers.reposet import load_account_reposets
from athenian.api.controllers.user import load_user_accounts
from athenian.api.db import Connection, Database, DatabaseLike
from athenian.api.defer import defer
from athenian.api.models.metadata.github import NodeUser, OrganizationMember
from athenian.api.models.state.models import Account, BanishedUserAccount, Invitation, \
    RepositorySet, UserAccount
from athenian.api.models.web import AcceptedInvitation, BadRequestError, DatabaseConflict, \
    ForbiddenError, GenericError, InstallationProgress, InvalidRequestError, \
    InvitationCheckResult, InvitationLink, InvitedUser, NotFoundError, \
    ServiceUnavailableError, TableFetchingProgress, TooManyRequestsError, User
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError
from athenian.api.tracing import sentry_span

admin_backdoor = (1 << 24) - 1
url_prefix = os.getenv("ATHENIAN_INVITATION_URL_PREFIX")
# we add 4 hours to compensate the installation time
TRIAL_PERIOD = relativedelta(months=1, hours=4)


def validate_env():
    """Check that the required global parameters are set."""
    if Auth0.KEY is None:
        raise EnvironmentError("ATHENIAN_INVITATION_KEY environment variable must be set")
    if url_prefix is None:
        raise EnvironmentError("ATHENIAN_INVITATION_URL_PREFIX environment variable must be set")
    if jira_url_template is None:
        raise EnvironmentError(
            "ATHENIAN_JIRA_INSTALLATION_URL_TEMPLATE environment variable must be set")


async def gen_user_invitation(request: AthenianWebRequest, id: int) -> web.Response:
    """Generate a new regular member invitation URL."""
    async with request.sdb.connection() as sdb_conn:
        await get_user_account_status_from_request(request, id)
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


@only_god
async def gen_account_invitation(request: AthenianWebRequest) -> web.Response:
    """Generate a new account invitation URL."""
    async with request.sdb.connection() as conn:
        async with conn.transaction():
            salt = randint(0, (1 << 16) - 1)  # 0:65535 - 2 bytes
            acc = await create_new_account(conn, request.app["auth"].key)
            inv = Invitation(salt=salt, account_id=acc, created_by=request.uid).create_defaults()
            invitation_id = await conn.execute(insert(Invitation).values(inv.explode()))
    slug = encode_slug(invitation_id, salt, request.app["auth"].key)
    model = InvitationLink(url=url_prefix + slug)
    return model_response(model)


async def _check_admin_access(uid: str, account: int, sdb_conn: morcilla.core.Connection):
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
    assert 0 <= salt < (1 << 16)
    assert 0 < iid < (1 << 24)
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
    try:
        invitation = AcceptedInvitation.from_dict(body)
    except ValueError as e:
        raise ResponseError(InvalidRequestError(getattr(e, "path", "?"), detail=str(e)))
    if invitation.name or invitation.email:
        if not await request.app["auth"].update_user_profile(
                request.uid, name=invitation.name, email=invitation.email):
            event_id = sentry_sdk.capture_message(
                "Auth0 update_user_profile() failure", level="error")
            raise ResponseError(ServiceUnavailableError(
                type="/errors/Auth0Error",
                detail="Unable to update user profile in Auth0.",
                instance=event_id,
            ))
    url = invitation.url
    if not url.startswith(url_prefix):
        bad_req()
    x = url[len(url_prefix):].strip("/")
    if len(x) != 8:
        bad_req()
    try:
        iid, salt = decode_slug(x, request.app["auth"].key)
    except binascii.Error:
        bad_req()
    async with sdb.connection() as sdb_conn:
        try:
            async with sdb_conn.transaction():
                inv = await sdb_conn.fetch_one(
                    select([Invitation.id, Invitation.account_id, Invitation.accepted,
                            Invitation.is_active])
                    .where(and_(Invitation.id == iid, Invitation.salt == salt)))
                if inv is None:
                    raise ResponseError(NotFoundError(detail="Invitation was not found."))
                if not inv[Invitation.is_active.name]:
                    raise ResponseError(ForbiddenError(detail="This invitation is disabled."))
                acc_id, user = await _join_account(
                    inv[Invitation.account_id.name], request, sdb_conn, invitation=inv)
        except (IntegrityConstraintViolationError, IntegrityError, OperationalError) as e:
            raise ResponseError(DatabaseConflict(detail=str(e)))
    return model_response(InvitedUser(account=acc_id, user=user))


async def join_account(acc_id: int,
                       request: AthenianWebRequest,
                       user: Optional[User] = None,
                       check_org_membership: bool = True,
                       ) -> User:
    """
    Join the `request`-ing user to the account `acc_id`.

    :param user: Prefetched `request.user()` for better performance.
    """
    async with request.sdb.connection() as sdb_conn:
        try:
            async with sdb_conn.transaction():
                return (await _join_account(
                    acc_id, request, sdb_conn, user=user,
                    check_org_membership=check_org_membership,
                ))[1]
        except (IntegrityConstraintViolationError, IntegrityError, OperationalError) as e:
            raise ResponseError(DatabaseConflict(detail=str(e)))


async def _join_account(acc_id: int,
                        request: AthenianWebRequest,
                        sdb_transaction: Connection,
                        user: Optional[User] = None,
                        invitation: Optional[Mapping[str, Any]] = None,
                        check_org_membership: bool = True,
                        ) -> Tuple[int, User]:
    """
    Join the `request`-ing user to the account `acc_id`.

    We need both `sdb_conn` and `sdb` because `sdb` is required in the deferred code outside of
    the transaction.
    You should work with `sdb_conn` in the code that blocks the request flow and with `sdb`
    in the code that defer()-s.
    """
    sdb, mdb, rdb, cache = request.sdb, request.mdb, request.rdb, request.cache
    log = logging.getLogger(f"{metadata.__package__}.join_account")
    if not (is_admin := acc_id == admin_backdoor):
        is_admin = 0 == await sdb_transaction.fetch_val(select([func.count(text("*"))])
                                                        .where(UserAccount.account_id == acc_id))
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
        if acc_id == admin_backdoor:
            # create a new account for the admin user
            acc_id = await create_new_account(sdb_transaction, request.app["auth"].key)
            log.info("Created new account %d", acc_id)
        else:
            log.info("Activated new account %d", acc_id)
        if slack is not None:
            async def report_new_account_to_slack():
                jira_link = await generate_jira_invitation_link(acc_id, sdb)
                await slack.post_install("new_account.jinja2",
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
    if status is None:
        if check_org_membership:
            user = await _check_user_org_membership(
                request, user, acc_id, sdb_transaction, sdb, slack, log)
        elif user is None:
            user = await request.user()
        # create the user<>account record if not blocked
        if await sdb_transaction.fetch_val(
                select([func.count(text("*"))])
                .where(and_(BanishedUserAccount.user_id == request.uid,
                            BanishedUserAccount.account_id == acc_id))):
            if slack is not None:
                async def report_blocked_registration():
                    account_name = await get_account_name_or_stub(acc_id, sdb, mdb, request.cache)
                    await slack.post_account("blocked_registration_banished.jinja2",
                                             user=user,
                                             account_id=acc_id,
                                             account_name=account_name)
                await defer(report_blocked_registration(), "report_blocked_registration_to_slack")
            log.warning("Did not allow blocked user %s to register in account %d",
                        request.uid, acc_id)
            raise ResponseError(ForbiddenError(detail="You were deleted from this account."))
        await sdb_transaction.execute(insert(UserAccount).values(UserAccount(
            user_id=request.uid,
            account_id=acc_id,
            is_admin=is_admin,
        ).create_defaults().explode(with_primary_keys=True)))
        log.info("Assigned user %s to account %d (admin: %s)", request.uid, acc_id, is_admin)
        if slack is not None:
            async def report_new_user_to_slack():
                account_name = await get_account_name_or_stub(acc_id, sdb, mdb, request.cache)
                await slack.post_account("new_user.jinja2",
                                         user=user,
                                         account_id=acc_id,
                                         account_name=account_name)

            await defer(report_new_user_to_slack(), "report_new_user_to_slack")
        if invitation is not None:
            values = {Invitation.accepted.name: invitation[Invitation.accepted.name] + 1}
            await sdb_transaction.execute(
                update(Invitation)
                .where(Invitation.id == invitation[Invitation.id.name])
                .values(values))
    if user is None:
        user = await request.user()
    user.accounts = await load_user_accounts(
        user.id, getattr(request, "god_id", user.id),
        sdb_transaction, mdb, rdb, slack, request.user, cache)
    return acc_id, user


async def _check_user_org_membership(request: AthenianWebRequest,
                                     user: Optional[User],
                                     acc_id: int,
                                     sdb_conn: DatabaseLike,
                                     sdb: Database,
                                     slack: Optional[SlackWebClient],
                                     log: logging.Logger,
                                     ) -> User:
    async def _load_user():
        if user is not None:
            return user
        return await request.user()

    if not await is_membership_check_enabled(acc_id, sdb_conn):
        return await _load_user()

    mdb = request.mdb
    cache = request.cache

    # check whether the user is a member of the GitHub org
    async def load_org_members() -> Tuple[Tuple[int, ...], List[int]]:
        if not (meta_ids := await get_metadata_account_ids_or_empty(acc_id, sdb_conn, cache)):
            log.warning("Could not check the organization membership of %s because "
                        "no metadata installation exists in account %d",
                        request.uid, acc_id)
            return (), []
        user_node_ids = [
            r[0] for r in await mdb.fetch_all(
                select([OrganizationMember.child_id])
                .where(OrganizationMember.acc_id.in_(meta_ids)))
        ]
        log.debug("Discovered %d organization members", len(user_node_ids))
        return meta_ids, user_node_ids

    user, (meta_ids, user_node_ids) = await gather(_load_user(), load_org_members())
    if not meta_ids:
        return user
    user_node_id = await mdb.fetch_val(select([NodeUser.node_id])
                                       .where(and_(NodeUser.acc_id.in_(meta_ids),
                                                   NodeUser.login == user.login)))
    if user_node_id not in user_node_ids:
        if slack is not None:
            async def report_blocked_registration():
                account_name = await get_account_name_or_stub(acc_id, sdb, mdb, request.cache)
                await slack.post_account("blocked_registration_membership.jinja2",
                                         user=user,
                                         account_id=acc_id,
                                         account_name=account_name)

            await defer(report_blocked_registration(), "report_blocked_registration_to_slack")
        raise ResponseError(ForbiddenError(
            detail="User %s does not belong to the GitHub organization." % request.uid))
    return user


async def create_new_account(conn: DatabaseLike, secret: str) -> int:
    """Create a new account."""
    if isinstance(conn, morcilla.Database):
        slow = conn.url.dialect == "sqlite"
    else:
        async with conn.raw_connection() as raw_connection:
            slow = isinstance(raw_connection, aiosqlite.core.Connection)
    if slow:
        acc_id = await _create_new_account_slow(conn, secret)
    else:
        acc_id = await _create_new_account_fast(conn, secret)
    if acc_id >= admin_backdoor:
        # overflow, we are not ready for you
        await conn.execute(delete(Account).where(Account.id == acc_id))
        raise ResponseError(GenericError(
            type="/errors/LockedError",
            title=HTTPStatus.LOCKED.phrase,
            status=HTTPStatus.LOCKED,
            detail="Invitation was not found."))
    return acc_id


async def _create_new_account_fast(conn: DatabaseLike, secret: str) -> int:
    """Create a new account.

    Should be used for PostgreSQL.
    """
    account_id = await conn.execute(
        insert(Account).values(Account(secret_salt=0,
                                       secret=Account.missing_secret,
                                       expires_at=datetime.now(timezone.utc) + TRIAL_PERIOD)
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
                  expires_at=datetime.now() + TRIAL_PERIOD).create_defaults()
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
                                       sdb: Database,
                                       mdb: Database,
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
    if slack is not None and cache is not None and not precomputed \
            and model.finished_date is not None \
            and datetime.now(timezone.utc) - model.finished_date > timedelta(hours=2) \
            and datetime.now(timezone.utc) - created > timedelta(hours=2):
        expires = await sdb.fetch_val(select([Account.expires_at]).where(Account.id == account))
        await _notify_precomputed_failure(slack, uid, account, model, created, expires, cache)
    model.tables.append(TableFetchingProgress(
        name="precomputed", fetched=int(precomputed), total=1))
    if not precomputed:
        model.finished_date = None


@cached(
    exptime=middle_term_exptime,
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
                                      expires: datetime,
                                      cache: Optional[aiomcache.Client]) -> None:
    await slack.post_install(
        "precomputed_failure.jinja2",
        uid=uid,
        account=account,
        model=model,
        created=created,
        expires=expires,
    )


@expires_header(5)
async def eval_metadata_progress(request: AthenianWebRequest, id: int) -> web.Response:
    """Return the current GitHub installation progress in Athenian."""
    await get_user_account_status_from_request(request, id)
    async with request.mdb.connection() as mdb_conn:
        model = await fetch_github_installation_progress(id, request.sdb, mdb_conn, request.cache)

    async def login_loader() -> str:
        return (await request.user()).login

    if model.finished_date is not None:
        await _append_precomputed_progress(
            model, id, request.uid, login_loader, request.sdb, request.mdb,
            request.cache, request.app["slack"])
    return model_response(model)


@expires_header(5)
async def eval_jira_progress(request: AthenianWebRequest, id: int) -> web.Response:
    """Return the current JIRA installation progress in Athenian."""
    await get_user_account_status_from_request(request, id)
    model = await fetch_jira_installation_progress(id, request.sdb, request.mdb, request.cache)
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
