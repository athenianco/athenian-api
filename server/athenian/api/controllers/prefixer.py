import asyncio
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from sqlalchemy import and_, select

from athenian.api.async_utils import gather
from athenian.api.models.metadata.github import Repository, User
from athenian.api.typing_utils import DatabaseLike


class Prefixer:
    """
    Prepend service prefixes to repository and user names.

    For example, turn athenianco/athenian-api to github.com/athenianco/athenian-api.
    """

    def __init__(self,
                 do_not_call_me_directly: Any, *,
                 repo_node_map: Dict[str, str],
                 repo_name_map: Dict[str, str],
                 user_node_map: Dict[str, str],
                 user_login_map: Dict[str, str]):
        """
        Initialize a new instance of Prefixer class.

        This constructor is not supposed to be called directly by the user.
        """
        self._repo_node_map = repo_node_map
        self._repo_name_map = repo_name_map
        self._user_node_map = user_node_map
        self._user_login_map = user_login_map

    @property
    def repo_node_map(self) -> Mapping[str, str]:
        """Return mapping from repository node IDs to prefixed names with owners."""
        return self._repo_node_map

    @property
    def repo_name_map(self) -> Mapping[str, str]:
        """Return mapping from repository names to prefixed names with owners."""
        return self._repo_name_map

    @property
    def user_node_map(self) -> Mapping[str, str]:
        """Return mapping from user node IDs to prefixed logins."""
        return self._user_node_map

    @property
    def user_login_map(self) -> Mapping[str, str]:
        """Return mapping from user logins to prefixed logins."""
        return self._user_login_map

    @staticmethod
    async def load(meta_ids: Iterable[int], mdb: DatabaseLike) -> "Prefixer":
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
        user_node_map = {
            r[User.node_id.key]: strip_proto(r[User.html_url.key]) for r in user_rows
        }
        user_login_map = {
            r[User.login.key]: strip_proto(r[User.html_url.key]) for r in user_rows
        }
        return Prefixer(None,
                        repo_node_map=repo_node_map,
                        repo_name_map=repo_name_map,
                        user_node_map=user_node_map,
                        user_login_map=user_login_map)

    @staticmethod
    def schedule_load(meta_ids: Tuple[int, ...], mdb: DatabaseLike) -> "PrefixerPromise":
        """Postponse the Prefixer initialization so that it can load asynchronously in \
        the background."""
        task = asyncio.create_task(Prefixer.load(meta_ids, mdb))
        return PrefixerPromise(None, task=task)

    def as_promise(self) -> "PrefixerPromise":
        """Convert self to a held promise."""
        promise = PrefixerPromise(None, task=None)
        promise._prefixer = self
        return promise

    def resolve_repo_nodes(self, repo_node_ids: Iterable[str]) -> List[str]:
        """Lookup each repository node ID in repo_node_map."""
        return [self._repo_node_map[node_id] for node_id in repo_node_ids]

    def prefix_repo_names(self, repo_names: Iterable[str]) -> List[str]:
        """Lookup each repository full name in repo_name_map."""
        return [self.repo_name_map[name] for name in repo_names]

    def resolve_user_nodes(self, user_node_ids: Iterable[str]) -> List[str]:
        """Lookup each user node ID in user_node_map."""
        return [self._user_node_map[node_id] for node_id in user_node_ids]

    def prefix_user_logins(self, user_logins: Iterable[str]) -> List[str]:
        """Lookup each user login in user_login_map."""
        return [self.user_login_map[name] for name in user_logins]


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
