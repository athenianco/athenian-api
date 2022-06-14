import base64
from collections import defaultdict
from datetime import timezone
from typing import Any, Sequence, Set, Tuple

import aiohttp.web
from sqlalchemy import insert, select

from athenian.api import ffx
from athenian.api.async_utils import gather
from athenian.api.auth import Auth0, disable_default_user
from athenian.api.db import DatabaseLike
from athenian.api.models.state.models import Share, UserAccount
from athenian.api.models.web import BadRequestError, NotFoundError, Share as WebShare
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response
from athenian.api.tracing import sentry_span


@disable_default_user
async def save_share(request: AthenianWebRequest, body: Any) -> aiohttp.web.Response:
    """Save the state of UI views and return a reference."""
    if body is None:
        raise ResponseError(BadRequestError("Saved data must not be null"))
    obj_id = await request.sdb.execute(
        insert(Share).values(
            Share(
                created_by=request.uid,
                data=body,
            )
            .create_defaults()
            .explode(),
        ),
    )
    finseq = _encode_share_id(obj_id, request.app["auth"])
    return aiohttp.web.json_response(finseq)


def _encode_share_id(obj_id: int, auth: Auth0) -> str:
    encseq = ffx.encrypt(obj_id.to_bytes(8, "big"), auth.key.encode())
    finseq = base64.b32encode(bytes.fromhex(encseq)).lower().decode()
    return finseq.replace("o", "8").replace("l", "9").rstrip("=")


@sentry_span
async def _fetch_user_accounts(uids: Sequence[str], sdb: DatabaseLike) -> Tuple[Set[int]]:
    rows = await sdb.fetch_all(
        select([UserAccount.user_id, UserAccount.account_id]).where(UserAccount.user_id.in_(uids)),
    )
    accs = defaultdict(list)
    for row in rows:
        accs[row[0]].append(row[1])
    return tuple(set(accs[uid]) for uid in uids)


async def get_share(request: AthenianWebRequest, id: str) -> aiohttp.web.Response:
    """Load the previously saved state of the UI views."""
    sdb = request.sdb
    try:
        b32 = id.replace("8", "o").replace("9", "l").encode().upper() + b"==="
        encseq = base64.b32decode(b32).hex()
        decseq = ffx.decrypt(encseq, request.app["auth"].key.encode())
        obj_id = int.from_bytes(decseq, "big")
        if obj_id < 1 or obj_id > (1 << 63):
            raise ValueError
    except Exception:
        raise ResponseError(BadRequestError("Invalid share identifier")) from None
    obj = await sdb.fetch_one(select([Share]).where(Share.id == obj_id))
    msg_404 = "Share does not exist or access denied"
    if obj is None:
        raise ResponseError(NotFoundError(msg_404))
    (my_accounts, your_accounts), you = await gather(
        _fetch_user_accounts((request.uid, obj[Share.created_by.name]), sdb),
        request.app["auth"].get_user(obj[Share.created_by.name]),
    )
    if not my_accounts.intersection(your_accounts):
        raise ResponseError(NotFoundError(msg_404))
    share = WebShare(
        author=you.name,
        created=obj[Share.created_at.name],
        data=obj[Share.data.name],
    )
    if sdb.url.dialect == "sqlite":
        share.created = share.created.replace(tzinfo=timezone.utc)
    return model_response(share)
