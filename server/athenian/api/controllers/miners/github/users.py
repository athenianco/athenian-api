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
    version=2,
)
async def mine_users(logins: Collection[str],
                     meta_ids: Tuple[int, ...],
                     mdb: databases.Database,
                     cache: Optional[aiomcache.Client],
                     ) -> List[Mapping[str, Any]]:
    """Fetch details about each GitHub user in the given list of `logins`."""
    return [dict(u) for u in await mdb.fetch_all(
        select([User]).where(and_(User.login.in_(logins), User.acc_id.in_(meta_ids))))]


@sentry_span
@cached(
    exptime=60 * 60,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda logins, with_prefix, **_: (",".join(sorted(logins)), with_prefix),
)
async def mine_user_avatars(logins: Iterable[str],
                            with_prefix: bool,
                            meta_ids: Tuple[int, ...],
                            mdb: databases.Database,
                            cache: Optional[aiomcache.Client],
                            ) -> List[Tuple[str, str]]:
    """Fetch the user profile picture URL for each login."""
    selected = [User.login, User.avatar_url]
    if with_prefix:
        selected.append(User.html_url)
    rows = await mdb.fetch_all(select(selected)
                               .where(and_(User.login.in_(logins),
                                           User.acc_id.in_(meta_ids))))
    if not with_prefix:
        return [(u[User.login.key], u[User.avatar_url.key]) for u in rows]
    return [(u[User.html_url.key].split("/", 2)[2], u[User.avatar_url.key]) for u in rows]
