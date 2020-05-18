import pickle
from typing import Any, Collection, List, Mapping, Optional

import aiomcache
import databases
from sqlalchemy import select

from athenian.api.cache import cached
from athenian.api.models.metadata.github import User


@cached(
    exptime=60 * 60,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda logins, **_: (",".join(sorted(logins)),),
)
async def mine_users(logins: Collection[str],
                     db: databases.Database,
                     cache: Optional[aiomcache.Client]) -> List[Mapping[str, Any]]:
    """Fetch details about each GitHub user in the given list of `logins`."""
    return [dict(u) for u in await db.fetch_all(select([User]).where(User.login.in_(logins)))]
