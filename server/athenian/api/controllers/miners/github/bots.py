import asyncio
import pickle
from time import time
from typing import Dict, FrozenSet, Optional, Set, Tuple, Union

import aiomcache
import morcilla
from sqlalchemy import and_, select

from athenian.api.async_utils import gather
from athenian.api.cache import cached, middle_term_exptime, short_term_exptime
from athenian.api.models.metadata.github import Bot, ExtraBot
from athenian.api.models.state.models import Team


class Bots:
    """Lazy loader of the set of bot logins."""

    def __init__(self):
        """Initialize a new instance of the Bots class."""
        self._bots = None  # type: Optional[Dict[int, Union[Set[str], FrozenSet[str]]]]
        self._timestamp = time()
        self._lock = None  # type: Optional[asyncio.Lock]

    async def _fetch(self, mdb: morcilla.Database) -> None:
        rows, extra = await gather(
            mdb.fetch_all(select([Bot.acc_id, Bot.login])),
            mdb.fetch_all(select([ExtraBot.login])),
        )
        self._bots = bots = {}
        for row in rows:
            bots.setdefault(row[Bot.acc_id.name], set()).add(row[Bot.login.name])
        bots[0] = frozenset(r[0] for r in extra)
        self._timestamp = time()

    async def _ensure_fetched(self, mdb: morcilla.Database) -> None:
        if self._bots is None or time() - self._timestamp >= middle_term_exptime:
            if self._lock is None:
                # we don't run multi-threaded
                self._lock = asyncio.Lock()
            async with self._lock:
                if self._bots is None or time() - self._timestamp >= middle_term_exptime:
                    await self._fetch(mdb)

    async def extra(self, mdb: morcilla.Database) -> FrozenSet[str]:
        """Return additional bots from "github_bots_extra" table."""
        await self._ensure_fetched(mdb)
        return self._bots[0]

    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda account, **_: (account,),
    )
    async def __call__(self,
                       account: int,
                       meta_ids: Tuple[int, ...],
                       mdb: morcilla.Database,
                       sdb: morcilla.Database,
                       cache: Optional[aiomcache.Client],
                       ) -> FrozenSet[str]:
        """
        Return the bot logins.

        There are two parts: global bots in mdb and local bots in the Bots team in sdb.
        """
        await self._ensure_fetched(mdb)
        team = (await sdb.fetch_val(select([Team.members]).where(and_(
            Team.owner_id == account,
            Team.name == Team.BOTS,
        )))) or []
        bots = frozenset(self._bots[0].union(
            (u.rsplit("/", 1)[1] for u in team),
            *(self._bots.get(mid, set()) for mid in meta_ids),
        ))
        return bots


bots = Bots()
del Bots  # yes, don't allow to use it directly
