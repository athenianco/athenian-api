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
from typing import List, Optional, Tuple

from aiohttp import web
import aiomcache
import aiosqlite.core
from asyncpg import IntegrityConstraintViolationError
import databases.core
import pyffx
from sqlalchemy import and_, delete, func, insert, select, update

from athenian.api import metadata
from athenian.api.cache import cached, max_exptime
from athenian.api.controllers.account import get_installation_ids, get_user_account_status
from athenian.api.models.metadata.github import FetchProgress, Installation, InstallationOwner, \
    InstallationRepo
from athenian.api.models.state.models import Account, Invitation, UserAccount
from athenian.api.models.web import BadRequestError, ForbiddenError, GenericError, \
    NoSourceDataError, NotFoundError
from athenian.api.models.web.generic_error import DatabaseConflict
from athenian.api.models.web.installation_progress import InstallationProgress
from athenian.api.models.web.invitation_check_result import InvitationCheckResult
from athenian.api.models.web.invitation_link import InvitationLink
from athenian.api.models.web.invited_user import InvitedUser
from athenian.api.models.web.table_fetching_progress import TableFetchingProgress
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError
from athenian.api.typing_utils import DatabaseLike

ikey = os.getenv("ATHENIAN_INVITATION_KEY")
admin_backdoor = (1 << 24) - 1
url_prefix = os.getenv("ATHENIAN_INVITATION_URL_PREFIX")
accept_admin_cooldown = timedelta(minutes=1)


def validate_env():
    """Check that the required global parameters are set."""
    if ikey is None:
        raise EnvironmentError("ATHENIAN_INVITATION_KEY environment variable must be set")
    if url_prefix is None:
        raise EnvironmentError("ATHENIAN_INVITATION_URL_PREFIX environment variable must be set")


async def gen_invitation(request: AthenianWebRequest, id: int) -> web.Response:
    """Generate a new regular member invitation URL."""
    sdb = request.sdb
    status = await sdb.fetch_one(
        select([UserAccount.is_admin])
        .where(and_(UserAccount.user_id == request.uid, UserAccount.account_id == id)))
    if status is None:
        return ResponseError(NotFoundError(
            detail="User %s is not in the account %d" % (request.uid, id))).response
    if not status[UserAccount.is_admin.key]:
        return ResponseError(ForbiddenError(
            detail="User %s is not an admin of the account %d" % (request.uid, id))).response
    existing = await sdb.fetch_one(
        select([Invitation.id, Invitation.salt])
        .where(and_(Invitation.is_active, Invitation.account_id == id)))
    if existing is not None:
        invitation_id = existing[Invitation.id.key]
        salt = existing[Invitation.salt.key]
    else:
        # create a new invitation
        salt = randint(0, (1 << 16) - 1)  # 0:65535 - 2 bytes
        inv = Invitation(salt=salt, account_id=id, created_by=request.uid).create_defaults()
        invitation_id = await sdb.execute(insert(Invitation).values(inv.explode()))
    slug = encode_slug(invitation_id, salt)
    model = InvitationLink(url=url_prefix + slug)
    return model_response(model)


def encode_slug(iid: int, salt: int) -> str:
    """Encode an invitation ID and some extra data to 8 chars."""
    part1 = struct.pack("!H", salt)  # 2 bytes
    part2 = struct.pack("!I", iid)[1:]  # 3 bytes
    binseq = (part1 + part2).hex()  # 5 bytes, 10 hex chars
    e = pyffx.String(ikey.encode(), alphabet="0123456789abcdef", length=len(binseq))
    encseq = e.encrypt(binseq)  # encrypted 5 bytes, 10 hex chars
    finseq = base64.b32encode(bytes.fromhex(encseq)).lower().decode()  # 8 base32 chars
    finseq = finseq.replace("o", "8").replace("l", "9")
    return finseq


def decode_slug(slug: str) -> (int, int):
    """Decode an invitation ID and some extra data from 8 chars."""
    assert len(slug) == 8
    assert isinstance(slug, str)
    b32 = slug.replace("8", "o").replace("9", "l").upper().encode()
    x = base64.b32decode(b32).hex()
    e = pyffx.String(ikey.encode(), alphabet="0123456789abcdef", length=len(x))
    x = bytes.fromhex(e.decrypt(x))
    salt = struct.unpack("!H", x[:2])[0]
    iid = struct.unpack("!I", b"\x00" + x[2:])[0]
    return iid, salt


async def accept_invitation(request: AthenianWebRequest, body: dict) -> web.Response:
    """Accept the membership invitation."""
    if getattr(request, "god_id", request.uid) != request.uid:
        return ResponseError(ForbiddenError(
            detail="You must not be an active god to accept an invitation.")).response

    def bad_req():
        return ResponseError(BadRequestError(detail="Invalid invitation URL")).response

    sdb = request.sdb
    url = InvitationLink.from_dict(body).url
    if not url.startswith(url_prefix):
        return bad_req()
    x = url[len(url_prefix):].strip("/")
    if len(x) != 8:
        return bad_req()
    try:
        iid, salt = decode_slug(x)
    except binascii.Error:
        return bad_req()
    async with sdb.connection() as conn:
        try:
            async with conn.transaction():
                acc_id, user = await _accept_invitation(iid, salt, request, conn)
        except (IntegrityConstraintViolationError, IntegrityError, OperationalError) as e:
            return ResponseError(DatabaseConflict(detail=str(e))).response
    return model_response(InvitedUser(account=acc_id, user=user))


async def _accept_invitation(iid, salt, request, conn):
    log = logging.getLogger(metadata.__package__)
    inv = await conn.fetch_one(
        select([Invitation.account_id, Invitation.accepted, Invitation.is_active])
        .where(and_(Invitation.id == iid, Invitation.salt == salt)))
    if inv is None:
        raise ResponseError(NotFoundError(detail="Invitation was not found."))
    if not inv[Invitation.is_active.key]:
        raise ResponseError(ForbiddenError(detail="This invitation is disabled."))
    acc_id = inv[Invitation.account_id.key]
    is_admin = acc_id == admin_backdoor
    slack = request.app["slack"]  # type: slack.WebClient
    if is_admin:
        timestamp = await conn.fetch_val(
            select([UserAccount.created_at]).where(and_(UserAccount.user_id == request.uid,
                                                        UserAccount.is_admin)))
        if timestamp is not None:
            if timestamp.tzinfo is None:
                now = datetime.utcnow()
            else:
                now = datetime.now(tz=timestamp.tzinfo)
            if now - timestamp < accept_admin_cooldown:
                raise ResponseError(GenericError(
                    type="/errors/AdminCooldownError",
                    title=HTTPStatus.TOO_MANY_REQUESTS.phrase,
                    status=HTTPStatus.TOO_MANY_REQUESTS,
                    detail="You accepted an admin invitation less than %s ago." %
                           accept_admin_cooldown))
        # create a new account for the admin user
        acc_id = await create_new_account(conn)
        if acc_id >= admin_backdoor:
            await conn.execute(delete(Account).where(Account.id == acc_id))
            raise ResponseError(GenericError(
                type="/errors/LockedError",
                title=HTTPStatus.LOCKED.phrase,
                status=HTTPStatus.LOCKED,
                detail="Invitation was not found."))
        log.info("Created new account %d", acc_id)
        if slack is not None:
            await slack.post("new_account.jinja2", uid=request.uid, account=acc_id)
        status = None
    else:
        status = await conn.fetch_one(select([UserAccount.is_admin])
                                      .where(and_(UserAccount.user_id == request.uid,
                                                  UserAccount.account_id == acc_id)))
    if status is None:
        # create the user<>account record
        user = UserAccount(
            user_id=request.uid,
            account_id=acc_id,
            is_admin=is_admin,
        ).create_defaults()
        await conn.execute(insert(UserAccount).values(user.explode(with_primary_keys=True)))
        log.info("Assigned user %s to account %d (admin: %s)", request.uid, acc_id, is_admin)
        if slack is not None:
            await slack.post("new_user.jinja2", user=await request.user(), account=acc_id)
        values = {Invitation.accepted.key: inv[Invitation.accepted.key] + 1}
        await conn.execute(update(Invitation).where(Invitation.id == iid).values(values))
    user = await (await request.user()).load_accounts(conn)
    return acc_id, user


async def create_new_account(conn: DatabaseLike) -> int:
    """Create a new account."""
    if isinstance(conn, databases.Database):
        slow = conn.url.dialect == "sqlite"
    else:
        slow = isinstance(conn.raw_connection, aiosqlite.core.Connection)
    if slow:
        return await _create_new_account_slow(conn)
    return await _create_new_account_fast(conn)


async def _create_new_account_fast(
        conn: DatabaseLike) -> int:
    """Create a new account.

    Should be used for PostgreSQL.
    """
    return await conn.execute(insert(Account).values(Account().create_defaults().explode()))


async def _create_new_account_slow(
        conn: DatabaseLike) -> int:
    """Create a new account without relying on autoincrement.

    SQLite does not allow resetting the primary key sequence, so we have to increment the ID
    by hand.
    """
    acc = Account().create_defaults()
    max_id = (await conn.fetch_one(select([func.max(Account.id)])
                                   .where(Account.id < admin_backdoor)))[0] or 0
    acc.id = max_id + 1
    return await conn.execute(insert(Account).values(acc.explode(with_primary_keys=True)))


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
        iid, salt = decode_slug(x)
    except binascii.Error:
        return model_response(result)
    inv = await request.sdb.fetch_one(
        select([Invitation.account_id, Invitation.is_active])
        .where(and_(Invitation.id == iid, Invitation.salt == salt)))
    if inv is None:
        return model_response(result)
    result.valid = True
    result.active = inv[Invitation.is_active.key]
    types = [InvitationCheckResult.INVITATION_TYPE_REGULAR,
             InvitationCheckResult.INVITATION_TYPE_ADMIN]
    result.type = types[inv[Invitation.account_id.key] == admin_backdoor]
    return model_response(result)


@cached(
    exptime=24 * 3600,  # 1 day
    serialize=lambda t: marshal.dumps(t),
    deserialize=lambda buf: marshal.loads(buf),
    key=lambda account, **_: (account,),
)
async def get_installation_delivery_ids(account: int,
                                        sdb_conn: databases.core.Connection,
                                        mdb_conn: databases.core.Connection,
                                        cache: Optional[aiomcache.Client],
                                        ) -> List[Tuple[int, str]]:
    """Load the app installation and delivery IDs for the given account."""
    installation_ids = await get_installation_ids(account, sdb_conn, cache)
    dids = await mdb_conn.fetch_all(
        select([Installation.id, Installation.delivery_id])
        .where(Installation.id.in_(installation_ids)))
    diff = len(installation_ids) - len(dids)
    if diff > 0:
        raise ResponseError(NoSourceDataError(detail="There %s %d missing installation%s." %
                                                     ("are" if diff > 1 else "is", diff,
                                                      "s" if diff > 1 else "")))
    return [(did[0], did[1]) for did in dids]


@cached(
    exptime=max_exptime,
    serialize=lambda s: s.encode(),
    deserialize=lambda b: b.decode(),
    key=lambda installation_id, **_: (installation_id,),
    refresh_on_access=True,
)
async def get_installation_owner(installation_id: int,
                                 mdb_conn: databases.core.Connection,
                                 cache: Optional[aiomcache.Client],
                                 ) -> str:
    """Load the native user ID who installed the app."""
    user_login = await mdb_conn.fetch_val(
        select([InstallationOwner.user_login])
        .where(InstallationOwner.install_id == installation_id))
    if user_login is None:
        raise ResponseError(NoSourceDataError(detail="The installation has not started yet."))
    return user_login


async def eval_invitation_progress(request: AthenianWebRequest, id: int) -> web.Response:
    """Return the current Athenian GitHub app installation progress."""
    async with request.sdb.connection() as sdb_conn:
        await get_user_account_status(request.uid, id, sdb_conn, request.cache)
        async with request.mdb.connection() as mdb_conn:
            id_ids = await get_installation_delivery_ids(id, sdb_conn, mdb_conn, request.cache)
            owner = await get_installation_owner(id_ids[0][0], mdb_conn, request.cache)
            # we don't cache this because the number of repos can dynamically change
            models = []
            for installation_id, delivery_id in id_ids:
                repositories = await mdb_conn.fetch_val(
                    select([func.count(InstallationRepo.repo_id)])
                    .where(InstallationRepo.install_id == installation_id))
                rows = await mdb_conn.fetch_all(
                    select([FetchProgress]).where(FetchProgress.event_id == delivery_id))
                tables = [TableFetchingProgress(fetched=r[FetchProgress.nodes_processed.key],
                                                name=r[FetchProgress.node_type.key],
                                                total=r[FetchProgress.nodes_total.key])
                          for r in rows]
                started_date = min(r[FetchProgress.created_at.key] for r in rows)
                if started_date is not None:
                    started_date = started_date.replace(tzinfo=timezone.utc)
                if any(t.fetched < t.total for t in tables):
                    finished_date = None
                else:
                    finished_date = max(r[FetchProgress.updated_at.key] for r in rows)
                if finished_date is not None:
                    finished_date = finished_date.replace(tzinfo=timezone.utc)
                model = InstallationProgress(started_date=started_date,
                                             finished_date=finished_date,
                                             owner=owner,
                                             repositories=repositories,
                                             tables=tables)
                models.append(model)
            tables = {}
            for m in models:
                for t in m.tables:
                    table = tables.setdefault(
                        t.name, TableFetchingProgress(name=t.name, fetched=0, total=0))
                    table.fetched += t.fetched
                    table.total += t.total
            model = InstallationProgress(started_date=min(m.started_date for m in models),
                                         finished_date=max(m.finished_date for m in models),
                                         owner=owner,
                                         repositories=sum(m.repositories for m in models),
                                         tables=sorted(tables.values()))
            return model_response(model)
