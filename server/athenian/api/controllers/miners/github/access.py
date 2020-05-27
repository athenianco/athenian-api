import marshal
from typing import Iterable, Set

from sqlalchemy import select

from athenian.api.cache import cached
from athenian.api.controllers.account import get_installation_ids
from athenian.api.controllers.miners.access import AccessChecker
from athenian.api.models.metadata.github import InstallationRepo
from athenian.api.tracing import sentry_span


class GitHubAccessChecker(AccessChecker):
    """
    Stateful repository access checker.

    We load the repositories belonging to the bound GitHub installation and rule them out from
    the checked set.
    """

    SERVICE = "github"

    @sentry_span
    async def load(self) -> "AccessChecker":
        """Fetch the list of accessible repositories."""
        iids = await get_installation_ids(self.account, self.sdb, self.cache)
        self._installed_repos = await self._fetch_installed_repos(iids)
        return self

    @cached(
        exptime=lambda self, **_: self.cache_ttl,
        serialize=marshal.dumps,
        deserialize=marshal.loads,
        key=lambda iids, **_: tuple(iids),
        cache=lambda self, **_: self.cache,
    )
    async def _fetch_installed_repos(self, iids: Iterable[int]) -> Set[str]:
        installed_repos_db = await self.mdb.fetch_all(
            select([InstallationRepo.repo_full_name])
            .where(InstallationRepo.install_id.in_(iids)))
        key = InstallationRepo.repo_full_name.key
        return {r[key] for r in installed_repos_db}

    async def check(self, repos: Set[str]) -> Set[str]:
        """Return repositories which do not belong to the metadata installation.

        The names must be *without* the service prefix.
        """
        assert isinstance(repos, set)
        return repos - self._installed_repos
