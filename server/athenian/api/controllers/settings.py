from enum import IntEnum
from functools import lru_cache
import re
from typing import Any, Callable, Collection, Coroutine, Dict, List, Optional, Set

import aiomcache
import databases
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
from sqlalchemy import and_, delete, insert, select
from sqlalchemy.sql import Select

from athenian.api.controllers.account import get_account_repositories
from athenian.api.controllers.reposet import resolve_repos
from athenian.api.models.state.models import ReleaseSetting
from athenian.api.models.web import InvalidRequestError, ReleaseMatchStrategy
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError
from athenian.api.typing_utils import dataclass

# rejected: PR was closed without merging.
# force_push_drop: commit history was overwritten and the PR's merge commit no longer exists.
ReleaseMatch = IntEnum("ReleaseMatch", {"rejected": -2,
                                        "force_push_drop": -1,
                                        ReleaseMatchStrategy.BRANCH: 0,
                                        ReleaseMatchStrategy.TAG: 1,
                                        ReleaseMatchStrategy.TAG_OR_BRANCH: 2,
                                        ReleaseMatchStrategy.EVENT: 3})
ReleaseMatch.__doc__ = """Enumeration of supported release matching strategies."""

default_branch_alias = "{{default}}"


@dataclass(slots=True, repr=False, frozen=True)
class ReleaseMatchSetting:
    """Internal representation of the repository release match setting."""

    branches: str
    tags: str
    match: ReleaseMatch

    def __str__(self) -> str:
        """Return the compact string representation of the object."""
        return '{"branches": "%s", "tags": "%s", "match": "%s"}' % (
            self.branches, self.tags, self.match.name)

    def __repr__(self) -> str:
        """Return the Python representation of the object."""
        return 'ReleaseMatchSetting(branches="%s", tags="%s", match=Match["%s"])' % (
            self.branches, self.tags, self.match.name)

    def __lt__(self, other: "ReleaseMatchSetting") -> bool:
        """Implement self < other to become sortable."""
        if self.match != other.match:
            return self.match < other.match
        if self.tags != other.tags:
            return self.tags < other.tags
        return self.branches < other.branches


class ReleaseSettings:
    """Mapping from prefixed repository full names to their release settings."""

    def __init__(self, map_prefixed: Dict[str, ReleaseMatchSetting]):
        """Initialize a new instance of ReleaseSettings class."""
        self._map_prefixed = map_prefixed
        self._map_native = {k.split("/", 1)[1]: v for k, v in map_prefixed.items()}
        self._coherence = dict(zip(self._map_native, self._map_prefixed))

    def __repr__(self) -> str:
        """Implement repr()."""
        return "ReleaseSettings(%r)" % self._map_prefixed

    def copy(self) -> "ReleaseSettings":
        """Shallow copy the settings."""
        return ReleaseSettings(self._map_prefixed.copy())

    @property
    def prefixed(self) -> Dict[str, ReleaseMatchSetting]:
        """View the release settings with repository name prefixes."""
        return self._map_prefixed

    @property
    def native(self) -> Dict[str, ReleaseMatchSetting]:
        """View the release settings without repository name prefixes."""
        return self._map_native

    def prefixed_for_native(self, name_without_prefix: str) -> str:
        """Return the prefixed repository name for an unprefixed name."""
        return self._coherence[name_without_prefix]

    def set_by_native(self, name_without_prefix: str, value: ReleaseMatchSetting) -> None:
        """Update release settings given a repository name without prefix."""
        self._map_prefixed[self.prefixed_for_native(name_without_prefix)] = \
            self._map_native[name_without_prefix] = value

    def set_by_prefixed(self, name_with_prefix: str, value: ReleaseMatchSetting) -> None:
        """Update release settings given a repository name with prefix."""
        self._map_prefixed[name_with_prefix] = \
            self._map_native[name_with_prefix.split("/", 1)[1]] = value


class Settings:
    """User's settings."""

    def __init__(self,
                 do_not_call_me_directly: Any, *,
                 account: int,
                 user_id: Optional[str],
                 login: Optional[Callable[[], Coroutine[None, None, str]]],
                 sdb: databases.Database,
                 mdb: databases.Database,
                 cache: Optional[aiomcache.Client],
                 slack: Optional[SlackWebClient]):
        """Initialize a new instance of Settings class."""
        self._account = account
        self._user_id = user_id
        self._login = login
        assert isinstance(sdb, databases.Database)
        self._sdb = sdb
        assert isinstance(mdb, databases.Database)
        self._mdb = mdb
        self._cache = cache
        self._slack = slack

    @classmethod
    def from_account(cls,
                     account: int,
                     sdb: databases.Database,
                     mdb: databases.Database,
                     cache: Optional[aiomcache.Client],
                     slack: Optional[SlackWebClient]):
        """Create a new Settings class instance in readonly mode given the account ID."""
        return Settings(None,
                        account=account,
                        user_id=None,
                        login=None,
                        sdb=sdb,
                        mdb=mdb,
                        cache=cache,
                        slack=slack)

    @classmethod
    def from_request(cls, request: AthenianWebRequest, account: int) -> "Settings":
        """Create a new Settings class instance in readwrite mode from the request object and \
        the account ID."""
        async def login_loader() -> str:
            return (await request.user()).login

        return Settings(None,
                        account=account,
                        user_id=request.uid,
                        login=login_loader,
                        sdb=request.sdb,
                        mdb=request.mdb,
                        cache=request.cache,
                        slack=request.app["slack"])

    @staticmethod
    @lru_cache(1024)
    def _cached_release_settings_sql(account: int, repos: Collection[str]) -> Select:
        return select([ReleaseSetting]).where(and_(ReleaseSetting.account_id == account,
                                                   ReleaseSetting.repository.in_(repos)))

    async def list_release_matches(self, repos: Optional[Collection[str]] = None,
                                   ) -> ReleaseSettings:
        """
        List the current release matching settings for all related repositories.

        Repository names must be prefixed!
        If `repos` is None, we load all the repositories belonging to the account.
        """
        async with self._sdb.connection() as conn:
            if repos is None:
                repos = await get_account_repositories(self._account, True, conn)
            repos = frozenset(repos)
            rows = await conn.fetch_all(self._cached_release_settings_sql(self._account, repos))
            settings = []
            loaded = set()
            for row in rows:
                repo = row[ReleaseSetting.repository.name]
                loaded.add(repo)
                settings.append((
                    repo,
                    ReleaseMatchSetting(
                        branches=row[ReleaseSetting.branches.name],
                        tags=row[ReleaseSetting.tags.name],
                        match=ReleaseMatch(row[ReleaseSetting.match.name]),
                    )))
            for repo in repos:
                if repo not in loaded:
                    settings.append((
                        repo,
                        ReleaseMatchSetting(
                            branches=default_branch_alias,
                            tags=".*",
                            match=ReleaseMatch.tag_or_branch,
                        )))
            settings.sort()
            settings = dict(settings)
        return ReleaseSettings(settings)

    async def set_release_matches(self,
                                  repos: List[str],
                                  branches: str,
                                  tags: str,
                                  match: ReleaseMatch,
                                  ) -> Set[str]:
        """Set the release matching rule for a list of repositories."""
        for propname, s in (("branches", ReleaseMatch.branch), ("tags", ReleaseMatch.tag)):
            propval = locals()[propname]
            if match in (s, ReleaseMatch.tag_or_branch) and not propval:
                raise ResponseError(InvalidRequestError(
                    "." + propname,
                    detail='Value may not be empty given "match" = "%s"' % match.name),
                )
            try:
                re.compile(propval)
            except re.error as e:
                raise ResponseError(InvalidRequestError(
                    "." + propname,
                    detail="Invalid regular expression: %s" % e),
                ) from None
        if not branches:
            branches = default_branch_alias
        if not tags:
            tags = ".*"
        async with self._sdb.connection() as conn:
            repos, _ = await resolve_repos(
                repos, self._account, self._user_id, self._login,
                conn, self._mdb, self._cache, self._slack, strip_prefix=False)
            values = [ReleaseSetting(repository=r,
                                     account_id=self._account,
                                     branches=branches,
                                     tags=tags,
                                     match=match,
                                     ).create_defaults().explode(with_primary_keys=True)
                      for r in repos]
            query = insert(ReleaseSetting).prefix_with("OR REPLACE", dialect="sqlite")
            async with conn.transaction():
                if self._sdb.url.dialect != "sqlite":
                    await conn.execute(delete(ReleaseSetting).where(and_(
                        ReleaseSetting.account_id == self._account,
                        ReleaseSetting.repository.in_(repos),
                    )))
                await conn.execute_many(query, values)
        return repos
