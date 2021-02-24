import marshal
from typing import Dict, Iterable, Set

from sqlalchemy import select

from athenian.api.cache import cached
from athenian.api.controllers.miners.access import AccessChecker
from athenian.api.models.metadata.github import AccountRepository
from athenian.api.tracing import sentry_span


class GitHubAccessChecker(AccessChecker):
    """
    Stateful repository access checker.

    We load the repositories belonging to the bound GitHub installation and rule them out from
    the checked set.

    Do not use this to load all the repos for the account! get_account_repositories() instead.
    """

    SERVICE = "github"

    @sentry_span
    async def load(self) -> "AccessChecker":
        """Fetch the list of accessible repositories."""
        self._installed_repos = await self._fetch_installed_repos(self.metadata_ids)
        return self

    @cached(
        exptime=lambda self, **_: self.cache_ttl,
        serialize=marshal.dumps,
        deserialize=marshal.loads,
        key=lambda metadata_ids, **_: tuple(metadata_ids),
        cache=lambda self, **_: self.cache,
    )
    async def _fetch_installed_repos(self, metadata_ids: Iterable[int]) -> Dict[str, str]:
        rows = await self.mdb.fetch_all(
            select([AccountRepository.repo_full_name, AccountRepository.repo_node_id])
            .where(AccountRepository.acc_id.in_(metadata_ids)))
        name_key = AccountRepository.repo_full_name.key
        node_key = AccountRepository.repo_node_id.key
        return {r[name_key]: r[node_key] for r in rows}

    async def check(self, repos: Set[str]) -> Set[str]:
        """Return repositories which do not belong to the metadata installation.

        The names must be *without* the service prefix.
        """
        assert isinstance(repos, set)
        return repos - self._installed_repos.keys()
