from typing import Optional, Set

import aiomcache

from athenian.api.models.metadata import PREFIXES
from athenian.api.typing_utils import DatabaseLike


class AccessChecker:
    """Interface for all repository access checkers."""

    CACHE_TTL = 60 * 60  # 1 hour
    SERVICE = ""

    def __init__(self,
                 account: int,
                 sdb_conn: DatabaseLike,
                 mdb_conn: DatabaseLike,
                 cache: Optional[aiomcache.Client],
                 cache_ttl=CACHE_TTL):
        """
        Initialize a new instance of AccessChecker.

        You need to await load() to prepare for check()-ing.
        """
        self.account = account
        self.sdb = sdb_conn
        self.mdb = mdb_conn
        self.cache = cache
        self.cache_ttl = cache_ttl
        self._installed_repos = set()

    def installed_repos(self, with_prefix: bool = True) -> Set[str]:
        """Get the currently installed repository names."""
        prefix = PREFIXES[self.SERVICE] if with_prefix else ""
        return {f"{prefix}{r}" for r in self._installed_repos}

    async def load(self) -> "AccessChecker":
        """Fetch the list of accessible repositories."""
        raise NotImplementedError

    async def check(self, repos: Set[str]) -> Set[str]:
        """Return repositories which do not belong to the metadata installation.

        The names must be *without* the service prefix.
        """
        raise NotImplementedError
