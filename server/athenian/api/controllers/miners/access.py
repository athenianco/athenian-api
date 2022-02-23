from typing import Dict, KeysView, Optional, Set, Tuple, Union

import aiomcache

from athenian.api.db import DatabaseLike


class AccessChecker:
    """Interface for all repository access checkers."""

    CACHE_TTL = 60  # 1 minute

    def __init__(self,
                 account: int,
                 metadata_ids: Tuple[int, ...],
                 sdb_conn: DatabaseLike,
                 mdb_conn: DatabaseLike,
                 cache: Optional[aiomcache.Client],
                 cache_ttl=CACHE_TTL):
        """
        Initialize a new instance of AccessChecker.

        You need to await load() to prepare for check()-ing.
        """
        assert len(metadata_ids) > 0
        self.account = account
        self.metadata_ids = metadata_ids
        self.sdb = sdb_conn
        self.mdb = mdb_conn
        self.cache = cache
        self.cache_ttl = cache_ttl
        self._installed_repos = {}

    @property
    def installed_repos(self) -> Dict[str, int]:
        """Get the currently installed repository names *without* the service prefix mapped to \
        node IDs."""
        return self._installed_repos

    async def load(self) -> "AccessChecker":
        """Fetch the list of accessible repositories."""
        raise NotImplementedError

    async def check(self, repos: Union[Set[str], KeysView[str]]) -> Set[str]:
        """Return repositories which do not belong to the metadata installation.

        The names must be *without* the service prefix.
        """
        raise NotImplementedError
