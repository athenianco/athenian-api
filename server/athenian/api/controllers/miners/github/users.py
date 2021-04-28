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
                     cache: Optional[aiomcache.Client],
                     ) -> List[Mapping[str, Any]]:
    """
    Fetch details about each GitHub user in the given list of `logins`.

    There can be duplicates when there are users of different types.
    """
    rows = await mdb.fetch_all(
        select([User.node_id, User.email, User.login, User.name, User.html_url, User.avatar_url])
        .where(and_(User.login.in_(logins), User.acc_id.in_(meta_ids)))
        .order_by(User.type))  # BOT -> MANNEQUIN -> ORGANIZATION -> USER
    return [dict(row) for row in rows]


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
    return [(u[User.html_url.key].split("://", 1)[1], u[User.avatar_url.key]) for u in rows]
