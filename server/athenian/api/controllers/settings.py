from dataclasses import dataclass
from enum import IntEnum
import re
from typing import Collection, Dict, List, Optional, Set

import aiomcache
import databases
import slack
from sqlalchemy import and_, delete, insert, select

from athenian.api import ResponseError
from athenian.api.controllers.account import get_user_account_status
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.controllers.reposet import resolve_repos
from athenian.api.models.state.models import ReleaseSetting
from athenian.api.models.web import InvalidRequestError, ReleaseMatchStrategy
from athenian.api.request import AthenianWebRequest

Match = IntEnum("Match", {ReleaseMatchStrategy.BRANCH: 0,
                          ReleaseMatchStrategy.TAG: 1,
                          ReleaseMatchStrategy.TAG_OR_BRANCH: 2})
Match.__doc__ = """Supported release matching strategies."""

default_branch_alias = "{{default}}"


@dataclass(frozen=True)
class ReleaseMatchSetting:
    """Internal representation of the repository release match setting."""

    branches: str
    tags: str
    match: Match

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


class Settings:
    """User's settings."""

    def __init__(self,
                 account: int,
                 user_id: str,
                 native_user_id: str,
                 sdb: databases.Database,
                 mdb: databases.Database,
                 cache: Optional[aiomcache.Client],
                 slack: Optional[slack.WebClient]):
        """Initialize a new instance of Settings class."""
        self._account = account
        self._user_id = user_id
        self._native_user_id = native_user_id
        assert isinstance(sdb, databases.Database)
        self._sdb = sdb
        assert isinstance(mdb, databases.Database)
        self._mdb = mdb
        self._cache = cache
        self._slack = slack

    @classmethod
    def from_request(cls, request: AthenianWebRequest, account: int) -> "Settings":
        """Create a new Settings class instance from the request object and the account ID."""
        return Settings(
            account=account, user_id=request.uid, native_user_id=request.native_uid,
            sdb=request.sdb, mdb=request.mdb, cache=request.cache, slack=request.app["slack"])

    async def list_release_matches(self, repos: Optional[Collection[str]] = None,
                                   ) -> Dict[str, ReleaseMatchSetting]:
        """List the current release matching settings for all related repositories."""
        async with self._sdb.connection() as conn:
            await get_user_account_status(self._user_id, self._account, conn, self._cache)
            if repos is None:
                repos = set()
                for cls in access_classes.values():
                    repos.update((await cls(self._account, conn, self._mdb, self._cache).load())
                                 .installed_repos())
            rows = await conn.fetch_all(
                select([ReleaseSetting]).where(and_(ReleaseSetting.account_id == self._account,
                                                    ReleaseSetting.repository.in_(repos))))
            settings = []
            loaded = set()
            for row in rows:
                repo = row[ReleaseSetting.repository.key]
                loaded.add(repo)
                settings.append((
                    repo,
                    ReleaseMatchSetting(
                        branches=row[ReleaseSetting.branches.key],
                        tags=row[ReleaseSetting.tags.key],
                        match=Match(row[ReleaseSetting.match.key]),
                    )))
            for repo in repos:
                if repo not in loaded:
                    settings.append((
                        repo,
                        ReleaseMatchSetting(
                            branches=default_branch_alias,
                            tags=".*",
                            match=Match.tag_or_branch,
                        )))
            settings.sort()
            settings = dict(settings)
        return settings

    async def set_release_matches(self,
                                  repos: List[str],
                                  branches: str,
                                  tags: str,
                                  match: Match,
                                  ) -> Set[str]:
        """Set the release matching rule for a list of repositories."""
        for propname, s in (("branches", Match.branch), ("tags", Match.tag)):
            propval = locals()[propname]
            if match in (s, Match.tag_or_branch) and not propval:
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
            # check that the user belongs to the account
            await get_user_account_status(self._user_id, self._account, conn, self._cache)
            repos = await resolve_repos(
                repos, self._account, self._user_id, self._native_user_id,
                conn, self._mdb, self._cache, self._slack, strip_prefix=False)
            values = [ReleaseSetting(repository=r,
                                     account_id=self._account,
                                     branches=branches,
                                     tags=tags,
                                     match=match,
                                     ).create_defaults().explode(with_primary_keys=True)
                      for r in repos]
            query = insert(ReleaseSetting).prefix_with("OR REPLACE", dialect="sqlite")
            if self._sdb.url.dialect != "sqlite":
                await conn.execute(delete(ReleaseSetting).where(and_(
                    ReleaseSetting.account_id == self._account,
                    ReleaseSetting.repository.in_(repos),
                )))
            await conn.execute_many(query, values)
        return repos
