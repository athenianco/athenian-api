from datetime import datetime, timezone
from itertools import chain
import logging
from typing import Any, Dict

import aiohttp
from flogging import flogging
from sqlalchemy import distinct, func, select

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.cache import CancelCache, cached, max_exptime
from athenian.api.internal.account import get_metadata_account_ids_or_empty
from athenian.api.models.metadata.github import NodeRepository
from athenian.api.models.state.models import UserAccount
from athenian.api.request import AthenianWebRequest

flogging.trailing_dot_exceptions.add("charset_normalizer")


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

    def _check_identify_full(result: bool, **_) -> bool:
        if not result:
            raise CancelCache()
        return True

    @cached(
        exptime=max_exptime,
        serialize=lambda full: b"\x01" if full else b"\x00",
        deserialize=lambda byte: byte != b"\x00",
        postprocess=_check_identify_full,
        key=lambda request, **_: (request.uid,),
        cache=lambda request, **_: request.cache,
        refresh_on_access=True,
    )
    async def _identify(self, request: AthenianWebRequest) -> bool:
        tasks = [
            request.user(),
            request.sdb.fetch_all(
                select([UserAccount.account_id]).where(UserAccount.user_id == request.uid)
            ),
        ]
        user, accounts = await gather(*tasks)
        accounts = [r[0] for r in accounts]
        tasks = [
            get_metadata_account_ids_or_empty(acc, request.sdb, request.cache) for acc in accounts
        ]
        meta_ids = list(chain.from_iterable(await gather(*tasks)))
        orgs = await request.mdb.fetch_all(
            # we could use SPLIT_PART but it does not work in SQLite
            select(
                [
                    distinct(
                        func.substr(
                            NodeRepository.name_with_owner,
                            1,  # SQL strings are 1-based
                            func.length(NodeRepository.name_with_owner)
                            - func.length(NodeRepository.name)
                            - 1,
                        )
                    )
                ]
            ).where(NodeRepository.acc_id.in_(meta_ids))
        )
        orgs = [org[0] for org in orgs]
        data = {
            "userId": user.id,
            "traits": {
                "name": user.name,
                "email": user.email,
                "accounts": accounts,
                "organizations": orgs,
                "login": user.login,
            },
            **self._common_data(),
        }
        await self._post(data, "identify")
        return bool(meta_ids)

    _check_identify_full = staticmethod(_check_identify_full)

    async def _track(self, request: AthenianWebRequest) -> None:
        data = {
            "userId": request.uid,
            "event": request.path,
            "properties": {
                **{
                    k.lower(): v
                    for k, v in request.headers.items()
                    if k.lower() not in ("authorization", "x-api-key")
                },
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
        async with self._session.post(
            f"{self.url}/{endpoint}", auth=aiohttp.BasicAuth(self._key, ""), json=data
        ) as response:
            if response.status != 200:
                self.log.error(
                    "Failed to %s in Segment: HTTP %d: %s",
                    endpoint,
                    response.status,
                    await response.text(),
                )
            else:
                self.log.debug("%s %s", endpoint, data)

    async def close(self) -> None:
        """Shut down the client."""
        await self._session.close()
