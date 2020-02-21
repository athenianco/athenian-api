import marshal
from typing import Optional, Set, Union

import aiomcache
import databases.core
from sqlalchemy import select

from athenian.api.cache import gen_cache_key
from athenian.api.controllers.account import get_installation_id
from athenian.api.models.metadata.github import InstallationRepo


class AccessChecker:
    """
    Stateful repository access checker.

    We load the repositories belonging to the bound GitHub installation and rule them out from
    the checked set.
    """

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

    async def load(self) -> None:
        """Fetch the list of accessible repositories."""
        iid = await get_installation_id(self.account, self.sdb, self.cache)
        cache_key = None
        installed_repos = None
        if self.cache is not None:
            cache_key = gen_cache_key("installation_repositories|%d", iid)
            installed_repos_blob = await self.cache.get(cache_key)
            if installed_repos_blob is not None:
                installed_repos = marshal.loads(installed_repos_blob)
        if installed_repos is None:
            installed_repos_db = await self.mdb.fetch_all(
                select([InstallationRepo.repo_full_name])
                .where(InstallationRepo.install_id == iid))
            key = InstallationRepo.repo_full_name.key
            installed_repos = {("github.com/" + r[key]) for r in installed_repos_db}
            if self.cache is not None:
                await self.cache.set(cache_key, marshal.dumps(installed_repos),
                                     exptime=self.cache_ttl)
        self._installed_repos = installed_repos

    async def check(self, repos: Set[str]) -> Set[str]:
        """Return repositories which do not belong to the metadata installation."""
        assert isinstance(repos, set)
        return repos - self._installed_repos
