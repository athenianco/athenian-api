from enum import auto, IntEnum
import marshal
import pickle
from typing import Any, Collection, Iterable, List, Mapping, Optional, Tuple, Union

import aiomcache
import morcilla
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
                     mdb: morcilla.Database,
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


class UserAvatarKeys(IntEnum):
    """User avatar identifier kind."""

    LOGIN = auto()
    PREFIXED_LOGIN = auto()
    NODE = auto()


@sentry_span
async def mine_user_avatars(logins: Iterable[str],
                            keys: UserAvatarKeys,
                            meta_ids: Tuple[int, ...],
                            mdb: morcilla.Database,
                            cache: Optional[aiomcache.Client],
                            ) -> List[Tuple[Union[str, int], str]]:
    """Fetch the user profile picture URL for each login."""
    tuples = await _mine_user_avatars(logins, meta_ids, mdb, cache)
    return [(
        node
        if keys == UserAvatarKeys.NODE
        else (
            prefixed_login
            if keys == UserAvatarKeys.PREFIXED_LOGIN
            else prefixed_login.rsplit("/", 1)[1]
        ),
        avatar,
    ) for node, prefixed_login, avatar in tuples]


@sentry_span
@cached(
    exptime=60 * 60,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda logins, **_: (",".join(sorted(logins)),),
)
async def _mine_user_avatars(logins: Iterable[str],
                             meta_ids: Tuple[int, ...],
                             mdb: morcilla.Database,
                             cache: Optional[aiomcache.Client],
                             ) -> List[Tuple[int, str, str]]:
    rows = await mdb.fetch_all(select([User.node_id, User.html_url, User.avatar_url])
                               .where(and_(User.login.in_(logins),
                                           User.acc_id.in_(meta_ids))))
    return [(u[User.node_id.name],
             u[User.html_url.name].split("://", 1)[1],
             u[User.avatar_url.name])
            for u in rows]
