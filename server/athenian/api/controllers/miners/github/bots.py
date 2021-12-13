import asyncio
import pickle
from time import time
from typing import FrozenSet, Optional

import aiomcache
import morcilla
from sqlalchemy import and_, select

from athenian.api.cache import cached, middle_term_exptime, short_term_exptime
from athenian.api.models.metadata.github import Bot
from athenian.api.models.state.models import Team


class Bots:
    """Lazy loader of the set of bot logins."""

    def __init__(self):
        """Initialize a new instance of the Bots class."""
        self._bots = None  # type: Optional[FrozenSet[str]]
        self._timestamp = time()
        self._lock = None  # type: Optional[asyncio.Lock]

    async def _fetch(self, mdb: morcilla.Database) -> None:
        self._bots = frozenset(r[0] for r in await mdb.fetch_all(select([Bot.login])))
        self._timestamp = time()

    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda account, **_: (account,),
    )
    async def __call__(self,
                       account: int,
                       mdb: morcilla.Database,
                       sdb: morcilla.Database,
                       cache: Optional[aiomcache.Client],
                       ) -> FrozenSet[str]:
        """
        Return the bot logins.

        There are two parts: global bots in mdb and local bots in the Bots team in sdb.
        """
        if self._bots is None or time() - self._timestamp >= middle_term_exptime:
            if self._lock is None:
                # we don't run multi-threaded
                self._lock = asyncio.Lock()
            async with self._lock:
                if self._bots is None or time() - self._timestamp >= middle_term_exptime:
                    await self._fetch(mdb)
        extra = await sdb.fetch_val(select([Team.members]).where(and_(
            Team.owner_id == account,
            Team.name == Team.BOTS,
        )))
        if extra is None:
            return self._bots
        return self._bots.union(u.rsplit("/", 1)[1] for u in extra)


bots = Bots()
del Bots  # yes, don't allow to use it directly
