import struct
from typing import Optional, Union

import aiomcache
import databases.core
from sqlalchemy import select

from athenian.api.cache import cached
from athenian.api.models.state.models import Account
from athenian.api.models.web import NoSourceDataError
from athenian.api.response import ResponseError


@cached(
    # the TTL is huge because this relation will never change and is requested frequently
    exptime=365 * 24 * 3600,
    serialize=lambda iid: struct.pack("!q", iid),
    deserialize=lambda buf: struct.unpack("!q", buf)[0],
    key=lambda account, **_: (account,),
)
async def get_installation_id(account: int,
                              sdb_conn: Union[databases.Database, databases.core.Connection],
                              cache: Optional[aiomcache.Client],
                              ) -> int:
    """Fetch the Athenian metadata installation ID for the given account ID."""
    iid = await sdb_conn.fetch_val(
        select([Account.installation_id]).where(Account.id == account))
    if iid is None:
        raise ResponseError(NoSourceDataError(
            detail="The account installation has not finished yet."))
    return iid
