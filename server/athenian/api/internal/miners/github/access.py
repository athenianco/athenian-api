from itertools import chain
import marshal
from typing import Iterable, KeysView, Optional

from sqlalchemy import and_, join, select

from athenian.api.async_utils import gather
from athenian.api.cache import CancelCache, cached
from athenian.api.internal.miners.access import AccessChecker
from athenian.api.models.metadata.github import AccountRepository, NodeRepository
from athenian.api.tracing import sentry_span


class GitHubAccessChecker(AccessChecker):
    """
    Stateful repository access checker.

    We load the repositories belonging to the bound GitHub installation and rule them out from
    the checked set.

    Do not use this to load all the repos for the account! get_account_repositories() instead.
    """

    @sentry_span
    async def load(self) -> "AccessChecker":
        """Fetch the list of accessible repositories."""
        self._installed_repos = await self._fetch_installed_repos_cached(self.metadata_ids)
        return self

    def _postprocess_fetch_installed_repos(
        result: dict[str, int],
        override_result: Optional[dict[str, int]] = None,
        **_,
    ) -> dict[str, int]:
        if override_result is not None:
            raise CancelCache()
        return result

    @cached(
        exptime=lambda self, **_: self.cache_ttl,
        serialize=marshal.dumps,
        deserialize=marshal.loads,
        key=lambda metadata_ids, **_: tuple(metadata_ids),
        postprocess=_postprocess_fetch_installed_repos,
        cache=lambda self, **_: self.cache,
    )
    async def _fetch_installed_repos_cached(
        self,
        metadata_ids: Iterable[int],
        override_result: Optional[dict[str, int]] = None,
    ) -> dict[str, int]:
        __tracebackhide__ = True  # noqa: F841
        if override_result is not None:
            return override_result
        return await self._fetch_installed_repos(metadata_ids)

    _postprocess_fetch_installed_repos = staticmethod(_postprocess_fetch_installed_repos)

    async def _fetch_installed_repos(self, metadata_ids: Iterable[int]) -> dict[str, int]:
        return dict(
            chain(
                *(
                    await gather(
                        self.mdb.fetch_all(
                            select(
                                AccountRepository.repo_full_name, AccountRepository.repo_graph_id,
                            ).where(AccountRepository.acc_id.in_(metadata_ids)),
                        ),
                        # workaround DEV-4725
                        self.mdb.fetch_all(
                            select(NodeRepository.name_with_owner, NodeRepository.node_id)
                            .select_from(
                                join(
                                    NodeRepository,
                                    AccountRepository,
                                    and_(
                                        NodeRepository.acc_id == AccountRepository.acc_id,
                                        NodeRepository.node_id == AccountRepository.repo_graph_id,
                                    ),
                                ),
                            )
                            .where(NodeRepository.acc_id.in_(metadata_ids)),
                        ),
                    )
                ),
            ),
        )

    async def check(self, repos: set[str] | KeysView[str]) -> set[str]:
        """Return repositories which do not belong to the metadata installation.

        The names must be *without* the service prefix.
        """
        assert isinstance(repos, (set, KeysView))
        if diff := repos - self._installed_repos.keys():
            updated_repos = await self._fetch_installed_repos(self.metadata_ids)
            if self._installed_repos != updated_repos:
                self._installed_repos |= updated_repos
                await self._fetch_installed_repos_cached(self.metadata_ids, self._installed_repos)
                diff -= updated_repos.keys()
        return diff
