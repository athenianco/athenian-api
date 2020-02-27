import marshal
from typing import Set

from sqlalchemy import select

from athenian.api.cache import cached
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
        self._installed_repos = await self._fetch_installed_repos(iid)
        return self

    @cached(
        exptime=lambda self, **_: self.cache_ttl,
        serialize=marshal.dumps,
        deserialize=marshal.loads,
        key=lambda iid, **_: (iid,),
        cache=lambda self, **_: self.cache,
    )
    async def _fetch_installed_repos(self, iid: int) -> Set[str]:
        installed_repos_db = await self.mdb.fetch_all(
            select([InstallationRepo.repo_full_name])
            .where(InstallationRepo.install_id == iid))
        key = InstallationRepo.repo_full_name.key
        return {r[key] for r in installed_repos_db}

    async def check(self, repos: Set[str]) -> Set[str]:
        """Return repositories which do not belong to the metadata installation.

        The names must be *without* the service prefix.
        """
        assert isinstance(repos, set)
        return repos - self._installed_repos
