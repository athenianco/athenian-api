import marshal
import pickle
from typing import Any, Collection, Iterable, List, Mapping, Optional, Tuple

import aiomcache
import databases
from sqlalchemy import and_, select

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
                     meta_ids: Tuple[int, ...],
                     mdb: databases.Database,
                     cache: Optional[aiomcache.Client]) -> List[Mapping[str, Any]]:
    """Fetch details about each GitHub user in the given list of `logins`."""
    return [dict(u) for u in await mdb.fetch_all(
        select([User]).where(and_(User.login.in_(logins), User.acc_id.in_(meta_ids))))]


@sentry_span
@cached(
    exptime=60 * 60,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda logins, prefix="", **_: (",".join(sorted(logins)), prefix),
)
async def mine_user_avatars(logins: Iterable[str],
                            meta_ids: Tuple[int, ...],
                            mdb: databases.Database,
                            cache: Optional[aiomcache.Client],
                            prefix="",
                            ) -> List[Tuple[str, str]]:
    """Fetch the user profile picture URL for each login."""
    rows = await mdb.fetch_all(select([User.login, User.avatar_url])
                               .where(and_(User.login.in_(logins), User.acc_id.in_(meta_ids))))
    return [(prefix + u[0], u[1]) for u in rows]
