from typing import Optional, Set, Union

import aiomcache
import databases.core


class AccessChecker:
    """Interface for all repository access checkers."""

    CACHE_TTL = 60 * 60  # 1 hour

    def __init__(self,
                 account: int,
                 sdb_conn: Union[databases.Database, databases.core.Connection],
                 mdb_conn: Union[databases.Database, databases.core.Connection],
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

    async def load(self) -> "AccessChecker":
        """Fetch the list of accessible repositories."""
        raise NotImplementedError

    async def check(self, repos: Set[str]) -> Set[str]:
        """Return repositories which do not belong to the metadata installation.

        The names must be *without* the service prefix.
        """
        raise NotImplementedError
