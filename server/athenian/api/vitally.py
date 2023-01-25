import asyncio
from datetime import datetime, timedelta
from random import random
from typing import Optional

from sqlalchemy import select

from athenian.api.db import Database, Row
from athenian.api.models.persistentdata.models import VitallyAccount


class VitallyAccounts:
    """Maintain the fresh mapping from account ID to the details stored in Vitally."""

    VAR_NAME = "vitally_accounts"

    def __init__(
        self,
        rdb: Optional[Database] = None,
        refresh_interval: Optional[timedelta] = None,
        jitter: Optional[timedelta] = None,
    ):
        """Initialize a new instance of VitallyAccounts."""
        if refresh_interval is None:
            refresh_interval = timedelta(seconds=3600 + 600)
        self.refresh_interval = refresh_interval
        if jitter is None:
            jitter = timedelta(seconds=1)
        self.jitter = jitter
        self.rdb = rdb
        self._accounts = {}
        self._task: Optional[asyncio.Task] = None

    @property
    def accounts(self) -> dict[int, Row]:
        """Return the mapping account ID -> Vitally info."""
        return self._accounts

    async def boot(self) -> None:
        """Start regular fetches."""
        if self._task is not None:
            now = datetime.now()
            deadline = (
                now.replace(minute=0, second=0, microsecond=0)
                + self.refresh_interval
                + random() * self.jitter
            )
            await asyncio.sleep((deadline - now).total_seconds())
        await self.refresh()
        self._task = asyncio.create_task(self.boot(), name="refresh Vitally accounts")

    def shutdown(self):
        """Stop regular fetches."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def refresh(self) -> None:
        """Update the current mapping."""
        rows = await self.rdb.fetch_all(select(VitallyAccount))
        acc_col = VitallyAccount.account_id.name
        self._accounts = {r[acc_col]: r for r in rows}
