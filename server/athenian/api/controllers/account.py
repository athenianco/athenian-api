import struct
from typing import Optional, Union

import aiomcache
import databases.core
from sqlalchemy import select

from athenian.api.cache import gen_cache_key
from athenian.api.models.state.models import Account
from athenian.api.models.web import NoSourceDataError
from athenian.api.response import ResponseError


async def get_installation_id(account: int,
                              sdb_conn: Union[databases.Database, databases.core.Connection],
                              cache: Optional[aiomcache.Client],
                              ) -> int:
    """Fetch the Athenian metadata installation ID for the given account ID."""
    cache_key = None
    iid = None
    if cache is not None:
        cache_key = gen_cache_key("installation_id|%d", account)
        iid_bin = await cache.get(cache_key)
        if iid_bin is not None:
            iid = struct.unpack("!q", iid_bin)[0]
    if iid is None:
        iid = await sdb_conn.fetch_val(
            select([Account.installation_id]).where(Account.id == account))
        if iid is None:
            raise ResponseError(NoSourceDataError(
                detail="The account installation has not finished yet."))
        if cache is not None:
            # the TTL is huge because this relation will never change and is requested frequently
            await cache.set(cache_key, struct.pack("!q", iid), exptime=365 * 24 * 3600)
    return iid
