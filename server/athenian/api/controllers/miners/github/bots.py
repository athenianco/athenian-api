import asyncio
from time import time
from typing import Optional, Set

import databases
from sqlalchemy import select

from athenian.api.cache import middle_term_expire
from athenian.api.models.metadata.github import Bot


class Bots:
    """Lazy loader of the set of bot logins."""

    def __init__(self):
        """Initialize a new instance of the Bots class."""
        self._bots = set()
        self._timestamp = time()
        self._lock = None  # type: Optional[asyncio.Lock]

    async def _fetch(self, mdb: databases.Database) -> None:
        self._bots = {r[0] for r in await mdb.fetch_all(select([Bot.login]))}
        self._timestamp = time()

    async def __call__(self, mdb: databases.Database) -> Set[str]:
        """Return the bot logins."""
        if self._bots and time() - self._timestamp < middle_term_expire:
            # fast path to avoid acquiring the lock
            return self._bots
        if self._lock is None:
            # we don't run multi-threaded
            self._lock = asyncio.Lock()
        async with self._lock:
            if not self._bots or time() - self._timestamp >= middle_term_expire:
                await self._fetch(mdb)
        return self._bots


bots = Bots()
