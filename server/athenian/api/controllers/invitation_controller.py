import base64
import binascii
import os
from random import randint
import struct

from aiohttp import web
import pyffx
from sqlalchemy import and_, insert, select, update

from athenian.api.controllers.response import response, ResponseError
from athenian.api.models.state.models import Invitation, UserAccount
from athenian.api.models.web import BadRequestError, ForbiddenError, NotFoundError
from athenian.api.models.web.accepted_invitation import AcceptedInvitation
from athenian.api.models.web.invitation_link import InvitationLink
from athenian.api.request import AthenianWebRequest


ikey = os.getenv("ATHENIAN_INVITATION_KEY")
prefix = "https://app.athenian.co/i/"


async def gen_invitation(request: AthenianWebRequest, id: int) -> web.Response:
    """Generate a new regular member invitation URL."""
    sdb = request.sdb
    status = await sdb.fetch_one(
        select([UserAccount.is_admin])
        .where(and_(UserAccount.user_id == request.user.id, UserAccount.account_id == id)))
    if status is None:
        return ResponseError(NotFoundError(
            detail="User %s is not in the account %d" % (request.user.id, id))).response
    if not status[UserAccount.is_admin.key]:
        return ResponseError(ForbiddenError(
            detail="User %s is not an admin of the account %d" % (request.user.id, id))).response
    existing = await sdb.fetch_one(
        select([Invitation.id, Invitation.salt])
        .where(and_(Invitation.is_active, Invitation.account_id == id)))
    if existing is not None:
        invitation_id = existing[Invitation.id.key]
        salt = existing[Invitation.salt.key]
    else:
        # create a new invitation
        salt = randint(0, (1 << 16) - 1)  # 0:65535 - 2 bytes
        inv = Invitation(salt=salt, account_id=id, created_by=request.user.id)
        inv.create_defaults()
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
    url = AcceptedInvitation.from_dict(body).origin
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
            select([Invitation.account_id, Invitation.accepted])
            .where(and_(Invitation.id == iid, Invitation.salt == salt, Invitation.is_active)))
        if inv is None:
            return ResponseError(NotFoundError(
                detail="Invitation was not found or is inactive.")).response
        acc = inv[Invitation.account_id.key]
        status = await conn.fetch_one(select([UserAccount.is_admin])
                                      .where(and_(UserAccount.user_id == request.user.id,
                                                  UserAccount.account_id == acc)))
        if status is None:
            # create the user<>account record if it does not exist
            user = UserAccount(user_id=request.user.id, account_id=acc)
            user.create_defaults()
            await conn.execute(insert(UserAccount).values(user.explode(with_primary_keys=True)))
            akey = Invitation.accepted.key
            await conn.execute(update(Invitation)
                               .where(Invitation.id == iid)
                               .values(**{akey: inv[akey] + 1}))
        await request.user.load_accounts(conn)
    return response(request.user)
