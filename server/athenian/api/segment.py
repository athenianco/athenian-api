from datetime import datetime, timezone
import logging
from typing import Any, Dict

import aiohttp

from athenian.api import metadata
from athenian.api.cache import cached, max_exptime
from athenian.api.models.web import Model
from athenian.api.request import AthenianWebRequest


class SegmentClient:
    """
    Segment.io client tracker.

    Docs: https://segment.com/docs/connections/sources/catalog/libraries/server/http-api/
    """

    log = logging.getLogger(f"{metadata.__package__}.SegmentClient")
    url = "https://api.segment.io/v1"

    def __init__(self, key: str):
        """Initialize a new isntance of SegmentClient."""
        self._session = aiohttp.ClientSession()
        self._key = key
        self.log.info("Enabled tracking user actions")

    async def submit(self, request: AthenianWebRequest) -> None:
        """Ensure that the user is identified and track another API call."""
        if getattr(request, "god_id", request.uid) != request.uid or request.is_default_user:
            return
        await self._identify(request)
        await self._track(request)

    @cached(
        exptime=max_exptime,
        serialize=lambda _: b"\x01",
        deserialize=lambda _: b"\x01",
        key=lambda request, **_: (request.uid,),
        cache=lambda request, **_: request.cache,
        refresh_on_access=True,
    )
    async def _identify(self, request: AthenianWebRequest) -> None:
        user = await request.user()
        data = {
            "userId": user.id,
            "traits": {
                "name": user.name,
                "email": user.email,
                "accounts": Model.serialize(user.accounts),
                "login": user.login,
            },
            **self._common_data(),
        }
        await self._post(data, "identify")

    async def _track(self, request: AthenianWebRequest) -> None:
        data = {
            "userId": request.uid,
            "event": request.path,
            "properties": {
                **{k.lower(): v for k, v in request.headers.items()
                   if k.lower() not in ("authorization", "x-api-key")},
                "account": request.account,
            },
            **self._common_data(),
        }
        await self._post(data, "track")

    @staticmethod
    def _common_data() -> Dict[str, Any]:
        return {
            "context": {
                "version": metadata.__version__,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _post(self, data: Dict[str, Any], endpoint: str) -> None:
        async with self._session.post(f"{self.url}/{endpoint}",
                                      auth=aiohttp.BasicAuth(self._key, ""),
                                      json=data) as response:
            if response.status != 200:
                self.log.error("Failed to %s in Segment: HTTP %d: %s",
                               endpoint, response.status, await response.text())
            else:
                self.log.debug("%s %s", endpoint, data)

    async def close(self) -> None:
        """Shut down the client."""
        await self._session.close()
