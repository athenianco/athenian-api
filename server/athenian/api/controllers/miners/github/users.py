import marshal
import pickle
from typing import Any, Collection, List, Mapping, Optional, Tuple

import aiomcache
import databases
from sqlalchemy import select

from athenian.api.cache import cached
from athenian.api.models.metadata.github import User
from athenian.api.tracing import sentry_span


@sentry_span
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


@sentry_span
@cached(
    exptime=60 * 60,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda logins, **_: (",".join(sorted(logins)),),
)
async def mine_user_avatars(logins: Collection[str],
                            db: databases.Database,
                            cache: Optional[aiomcache.Client],
                            prefix="",
                            ) -> List[Tuple[str, str]]:
    """Fetch the user profile picture URL for each login."""
    rows = await db.fetch_all(select([User.login, User.avatar_url])
                              .where(User.login.in_(logins)))
    return [(prefix + u[0], u[1]) for u in rows]
