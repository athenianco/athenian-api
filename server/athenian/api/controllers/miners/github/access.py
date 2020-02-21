import marshal
from typing import Set

from sqlalchemy import select

from athenian.api.cache import gen_cache_key
from athenian.api.controllers.account import get_installation_id
from athenian.api.controllers.miners.access import AccessChecker
from athenian.api.models.metadata.github import InstallationRepo


class GitHubAccessChecker(AccessChecker):
    """
    Stateful repository access checker.

    We load the repositories belonging to the bound GitHub installation and rule them out from
    the checked set.
    """

    async def load(self) -> "AccessChecker":
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
        return self

    async def check(self, repos: Set[str]) -> Set[str]:
        """Return repositories which do not belong to the metadata installation."""
        assert isinstance(repos, set)
        return repos - self._installed_repos
