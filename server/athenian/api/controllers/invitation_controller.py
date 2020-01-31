import base64
import binascii
from http import HTTPStatus
import os
from random import randint
import struct

from aiohttp import web
import pyffx
from sqlalchemy import and_, delete, insert, select, update

from athenian.api.controllers.response import response, ResponseError
from athenian.api.models.state.models import Account, Invitation, UserAccount
from athenian.api.models.web import BadRequestError, ForbiddenError, GenericError, NotFoundError
from athenian.api.models.web.invitation_check_result import InvitationCheckResult
from athenian.api.models.web.invitation_link import InvitationLink
from athenian.api.models.web.invited_user import InvitedUser
from athenian.api.request import AthenianWebRequest


ikey = os.getenv("ATHENIAN_INVITATION_KEY")
prefix = "https://app.athenian.co/i/"
admin_backdoor = (1 << 24) - 1


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
    model = InvitationLink(url=prefix + slug)
    return response(model)


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
    def bad_req():
        return ResponseError(BadRequestError(detail="Invalid invitation URL")).response

    sdb = request.sdb
    url = InvitationLink.from_dict(body).url
    if not url.startswith(prefix):
        return bad_req()
    x = url[len(prefix):].strip("/")
    if len(x) != 8:
        return bad_req()
    try:
        iid, salt = decode_slug(x)
    except binascii.Error:
        return bad_req()
    async with sdb.connection() as conn:
        inv = await conn.fetch_one(
            select([Invitation.account_id, Invitation.accepted, Invitation.is_active])
            .where(and_(Invitation.id == iid, Invitation.salt == salt)))
        if inv is None:
            return ResponseError(NotFoundError(
                detail="Invitation was not found.")).response
        if not inv[Invitation.is_active.key]:
            return ResponseError(ForbiddenError(
                detail="This invitation is disabled.")).response
        acc_id = inv[Invitation.account_id.key]
        is_admin = acc_id == admin_backdoor
        if is_admin:
            # create a new account for the admin user
            acc_id = await conn.execute(
                insert(Account).values(Account().create_defaults().explode()))
            if acc_id >= admin_backdoor:
                await conn.execute(delete(Account).where(Account.id == acc_id))
                return ResponseError(GenericError(
                    type="/errors/LockedError",
                    title=HTTPStatus.LOCKED.phrase,
                    status=HTTPStatus.LOCKED,
                    detail="Invitation was not found.")).response
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
            values = {Invitation.accepted.key: inv[Invitation.accepted.key] + 1}
            await conn.execute(update(Invitation).where(Invitation.id == iid).values(values))
        user = await (await request.user()).load_accounts(conn)
    return response(InvitedUser(account=acc_id, user=user))


async def check_invitation(request: AthenianWebRequest, body: dict) -> web.Response:
    """Given an invitation URL, get its type (admin or regular account member) and find whether \
    it is enabled or disabled."""
    url = InvitationLink.from_dict(body).url
    result = InvitationCheckResult(valid=False)
    if not url.startswith(prefix):
        return response(result)
    x = url[len(prefix):].strip("/")
    if len(x) != 8:
        return response(result)
    try:
        iid, salt = decode_slug(x)
    except binascii.Error:
        return response(result)
    inv = await request.sdb.fetch_one(
        select([Invitation.account_id, Invitation.is_active])
        .where(and_(Invitation.id == iid, Invitation.salt == salt)))
    if inv is None:
        return response(result)
    result.valid = True
    result.active = inv[Invitation.is_active.key]
    types = [InvitationCheckResult.INVITATION_TYPE_REGULAR,
             InvitationCheckResult.INVITATION_TYPE_ADMIN]
    result.type = types[inv[Invitation.account_id.key] == admin_backdoor]
    return response(result)
