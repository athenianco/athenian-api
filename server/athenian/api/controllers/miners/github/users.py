import marshal
from typing import Collection, List, Optional

import aiomcache
import databases
from sqlalchemy import select

from athenian.api.cache import cached
from athenian.api.models.metadata.github import User


async def mine_users(logins: Collection[str], db: databases.Database,
                     fields: Optional[Collection[str]] = None,
                     cache: Optional[aiomcache.Client] = None) -> List[dict]:
    users = await _mine_users_all_fields(logins, db, cache=cache)
    if fields:
        users = [{f: dict(u)[f] for f in fields} for u in users]

    return users


@cached(
    exptime=5 * 60,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda logins, **_: (",".join(sorted(logins))),
)
async def _mine_users_all_fields(logins: Collection[str], db: databases.Database,
                                 cache: Optional[aiomcache.Client] = None) -> List[dict]:
    return await db.fetch_all(select([User]).where(User.login.in_(logins)))
