from __future__ import annotations

import asyncio
import pickle
from time import time
from typing import Optional

import aiomcache
import morcilla
from sqlalchemy import and_, select

from athenian.api.async_utils import gather
from athenian.api.cache import cached, middle_term_exptime, short_term_exptime
from athenian.api.models.metadata.github import Bot, ExtraBot, User
from athenian.api.models.state.models import Team


class Bots:
    """Lazy loader of the set of bot logins."""

    def __init__(self):
        """Initialize a new instance of the Bots class."""
        self._bots: Optional[dict[int, frozenset[str]]] = None
        self._timestamp = time()
        self._lock: Optional[asyncio.Lock] = None

    async def _fetch(self, mdb: morcilla.Database) -> None:
        rows, extra = await gather(
            mdb.fetch_all(select([Bot.acc_id, Bot.login])),
            mdb.fetch_all(select([ExtraBot.login])),
        )
        bots: dict[int, set[str]] = {}
        for row in rows:
            bots.setdefault(row[Bot.acc_id.name], set()).add(row[Bot.login.name])
        self._bots = {n: frozenset(v) for n, v in bots.items()}
        self._bots[0] = frozenset(r[0] for r in extra)
        self._timestamp = time()

    async def _ensure_fetched(self, mdb: morcilla.Database) -> None:
        if self._bots is None or time() - self._timestamp >= middle_term_exptime:
            if self._lock is None:
                # we don't run multi-threaded
                self._lock = asyncio.Lock()
            async with self._lock:
                if self._bots is None or time() - self._timestamp >= middle_term_exptime:
                    await self._fetch(mdb)

    async def extra(self, mdb: morcilla.Database) -> frozenset[str]:
        """Return additional bots from "github_bots_extra" table."""
        await self._ensure_fetched(mdb)
        assert self._bots is not None
        return self._bots[0]

    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda account, **_: (account,),
    )
    async def __call__(
        self,
        account: int,
        meta_ids: tuple[int, ...],
        mdb: morcilla.Database,
        sdb: morcilla.Database,
        cache: Optional[aiomcache.Client],
    ) -> frozenset[str]:
        """
        Return the bot logins.

        There are two parts: global bots in mdb and local bots in the Bots team in sdb.
        """
        await self._ensure_fetched(mdb)
        assert self._bots is not None
        team = (
            await sdb.fetch_val(
                select([Team.members]).where(
                    and_(
                        Team.owner_id == account,
                        Team.name == Team.BOTS,
                    ),
                ),
            )
        ) or []
        team_logins = await mdb.fetch_all(
            select([User.login]).where(
                and_(
                    User.acc_id.in_(meta_ids),
                    User.node_id.in_(team),
                ),
            ),
        )
        bots = frozenset(
            self._bots[0].union(
                (r[0] for r in team_logins),
                *(self._bots.get(mid, set()) for mid in meta_ids),
            ),
        )
        return bots


bots = Bots()
del Bots  # yes, don't allow to use it directly
