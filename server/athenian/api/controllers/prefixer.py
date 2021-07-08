import asyncio
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple

import aiomcache
from sqlalchemy import and_, select

from athenian.api.async_utils import gather
from athenian.api.cache import cached
from athenian.api.db import DatabaseLike
from athenian.api.models.metadata.github import Repository, User
from athenian.api.typing_utils import dataclass


@dataclass(slots=True, frozen=True)
class Prefixer:
    """
    Prepend service prefixes to repository and user names.

    For example, turn athenianco/athenian-api to github.com/athenianco/athenian-api.
    """

    do_not_construct_me_directly: None
    repo_node_map: Dict[str, str]
    repo_name_map: Dict[str, str]
    user_node_to_prefixed_login: Dict[str, str]
    user_login_to_prefixed_login: Dict[str, str]
    user_node_to_login: Dict[str, str]
    user_login_to_node: Dict[str, str]

    @staticmethod
    @cached(
        exptime=5 * 60,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda meta_ids, **_: (",".join(str(i) for i in meta_ids),),
    )
    async def load(meta_ids: Iterable[int],
                   mdb: DatabaseLike,
                   cache: Optional[aiomcache.Client],
                   ) -> "Prefixer":
        """Create a Prefixer for all repositories and users of the given metadata account IDs."""
        repo_rows, user_rows = await gather(
            mdb.fetch_all(
                select([Repository.node_id, Repository.full_name, Repository.html_url])
                .where(and_(Repository.acc_id.in_(meta_ids),
                            Repository.full_name.isnot(None))),
            ),
            mdb.fetch_all(
                select([User.node_id, User.login, User.html_url])
                .where(and_(User.acc_id.in_(meta_ids),
                            User.login.isnot(None))),
            ),
            op="Prefixer",
        )

        def strip_proto(url: str) -> str:
            return url.split("://", 1)[1]

        repo_node_map = {
            r[Repository.node_id.key]: strip_proto(r[Repository.html_url.key]) for r in repo_rows
        }
        repo_name_map = {
            r[Repository.full_name.key]: strip_proto(r[Repository.html_url.key]) for r in repo_rows
        }
        user_node_to_prefixed_login = {
            r[User.node_id.key]: strip_proto(r[User.html_url.key]) for r in user_rows
        }
        user_login_to_prefixed_login = {
            r[User.login.key]: strip_proto(r[User.html_url.key]) for r in user_rows
        }
        user_node_to_login = {
            r[User.node_id.key]: r[User.login.key] for r in user_rows
        }
        user_login_to_node = {
            r[User.login.key]: r[User.node_id.key] for r in user_rows
        }
        return Prefixer(None,
                        repo_node_map=repo_node_map,
                        repo_name_map=repo_name_map,
                        user_node_to_prefixed_login=user_node_to_prefixed_login,
                        user_login_to_prefixed_login=user_login_to_prefixed_login,
                        user_node_to_login=user_node_to_login,
                        user_login_to_node=user_login_to_node)

    @staticmethod
    def schedule_load(meta_ids: Tuple[int, ...],
                      mdb: DatabaseLike,
                      cache: Optional[aiomcache.Client],
                      ) -> "PrefixerPromise":
        """Postponse the Prefixer initialization so that it can load asynchronously in \
        the background."""
        task = asyncio.create_task(Prefixer.load(meta_ids, mdb, cache))
        return PrefixerPromise(None, task=task)

    def as_promise(self) -> "PrefixerPromise":
        """Convert self to a held promise."""
        promise = PrefixerPromise(None, task=None)
        promise._prefixer = self
        return promise

    def resolve_repo_nodes(self, repo_node_ids: Iterable[str]) -> List[str]:
        """Lookup each repository node ID in repo_node_map."""
        return [self.repo_node_map[node_id] for node_id in repo_node_ids]

    def prefix_repo_names(self, repo_names: Iterable[str]) -> List[str]:
        """Lookup each repository full name in repo_name_map."""
        return [self.repo_name_map[name] for name in repo_names]

    def resolve_user_nodes(self, user_node_ids: Iterable[str]) -> List[str]:
        """Lookup each user node ID in user_node_to_prefixed_login."""
        return [self.user_node_to_prefixed_login[node_id] for node_id in user_node_ids]

    def prefix_user_logins(self, user_logins: Iterable[str]) -> List[str]:
        """Lookup each user login in user_login_to_prefixed_login."""
        return [self.user_login_to_prefixed_login[name] for name in user_logins]


class PrefixerPromise:
    """
    Lazy loading wrapper around Prefixer.

    Usage:
        >>> promise = Prefixer.schedule_load(meta_ids, mdb)
        >>> # ...
        >>> prefixer = await promise.load()
    """

    def __init__(self,
                 do_not_call_me_directly: Any, *,
                 task: Optional[asyncio.Task]):
        """Initialize a new instance of PrefixerPromise. The user is not supposed to call this \
        constructor directly."""
        self._task = task
        self._prefixer = None

    async def load(self) -> Prefixer:
        """Block until the referenced Prefixer loads and return it."""
        if self._prefixer is None:
            assert self._task is not None
            await self._task
            self._prefixer = self._task.result()
            self._task = None
        return self._prefixer

    def cancel(self):
        """Stop and delete the task to load the Prefixer."""
        if self._task is not None:
            self._task.cancel()
            self._task = None
