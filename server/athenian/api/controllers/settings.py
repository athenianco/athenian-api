from enum import IntEnum
import re
from typing import Collection, Dict, List, Optional, Set

import aiomcache
import databases
from sqlalchemy import and_, delete, insert, select

from athenian.api import ResponseError
from athenian.api.controllers.account import get_user_account_status
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.controllers.reposet import resolve_repos
from athenian.api.models.state.models import ReleaseSetting
from athenian.api.models.web import ForbiddenError, InvalidRequestError, \
    ReleaseMatchSetting as WebReleaseMatchSetting, ReleaseMatchStrategy
from athenian.api.request import AthenianWebRequest

Match = IntEnum("Match", {ReleaseMatchStrategy.BRANCH: 0,
                          ReleaseMatchStrategy.TAG: 1,
                          ReleaseMatchStrategy.TAG_OR_BRANCH: 2})
Match.__doc__ = """Supported release matching strategies."""

default_branch_alias = "{{default}}"


class Settings:
    """User's settings."""

    def __init__(self,
                 account: int,
                 user_id: str,
                 native_user_id: str,
                 sdb: databases.Database,
                 mdb: databases.Database,
                 cache: Optional[aiomcache.Client]):
        """Initialize a new instance of Settings class."""
        self._account = account
        self._user_id = user_id
        self._native_user_id = native_user_id
        assert isinstance(sdb, databases.Database)
        self._sdb = sdb
        assert isinstance(mdb, databases.Database)
        self._mdb = mdb
        self._cache = cache

    @classmethod
    def from_request(cls, request: AthenianWebRequest, account: int) -> "Settings":
        """Create a new Settings class instance from the request object and the account ID."""
        return Settings(
            account=account, user_id=request.uid, native_user_id=request.native_uid,
            sdb=request.sdb, mdb=request.mdb, cache=request.cache)

    async def list_release_matches(self, repos: Optional[Collection[str]] = None,
                                   ) -> Dict[str, WebReleaseMatchSetting]:
        """List the current release matching settings for all related repositories."""
        async with self._sdb.connection() as conn:
            await get_user_account_status(self._user_id, self._account, conn, self._cache)
            if repos is None:
                repos = set()
                for cls in access_classes.values():
                    repos.update((await cls(self._account, conn, self._mdb, self._cache).load())
                                 .installed_repos())
            settings: Dict[str, WebReleaseMatchSetting] = {}
            rows = await conn.fetch_all(
                select([ReleaseSetting]).where(and_(ReleaseSetting.account_id == self._account,
                                                    ReleaseSetting.repository.in_(repos))))
            for row in rows:
                settings[row[ReleaseSetting.repository.key]] = WebReleaseMatchSetting(
                    branches=row[ReleaseSetting.branches.key],
                    tags=row[ReleaseSetting.tags.key],
                    match=Match(row[ReleaseSetting.match.key]).name,
                )
            for repo in repos:
                if repo not in settings:
                    settings[repo] = WebReleaseMatchSetting(
                        branches=default_branch_alias,
                        tags=".*",
                        match=Match.tag_or_branch.name,
                    )
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
            if not await get_user_account_status(
                    self._user_id, self._account, conn, self._cache):
                raise ResponseError(ForbiddenError(
                    detail="User %s is not an admin of %d" % (self._user_id, self._account)))
            repos = await resolve_repos(
                repos, self._account, self._user_id, self._native_user_id,
                conn, self._mdb, self._cache, strip_prefix=False)
            values = [ReleaseSetting(repository=r,
                                     account_id=self._account,
                                     branches=branches,
                                     tags=tags,
                                     match=match,
                                     ).create_defaults().explode(with_primary_keys=True)
                      for r in repos]
            query = insert(ReleaseSetting).prefix_with("OR REPLACE", dialect="sqlite")
            if self._sdb.url.dialect != "sqlite":
                await conn.execute(delete([ReleaseSetting]).where(and_(
                    ReleaseSetting.account_id == self._account,
                    ReleaseSetting.repository.in_(repos),
                )))
            await conn.execute_many(query, values)
        return repos
