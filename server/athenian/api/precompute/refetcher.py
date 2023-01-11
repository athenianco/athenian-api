from datetime import datetime, timezone
import json
import logging
import os
from typing import Iterable

import aiohttp
from gcloud.aio.pubsub import PublisherClient, PubsubMessage

from athenian.api import metadata


class Refetcher:
    """Sender of metadata refetch requests to PubSub."""

    log = logging.getLogger(f"{metadata.__package__}.Refetcher")

    def __init__(self, topic: str | None, meta_ids: tuple[int, ...]):
        """Initialize a new instance of Refetcher class."""
        self._meta_ids = meta_ids
        if topic:
            self._session = aiohttp.ClientSession()
            self._client = PublisherClient(session=self._session)
            self._topic = self._client.topic_path(
                os.getenv("GOOGLE_CLOUD_PROJECT", os.getenv("GOOGLE_PROJECT")), topic,
            )
        else:
            self._session = self._client = self._topic = None

    async def close(self) -> None:
        """Free any resources in use by the instance."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def submit_commits(self, nodes: Iterable[int], noraise: bool = False) -> None:
        """Send commit node IDs to refetch."""
        await self._submit_noraise(nodes, "Commit", noraise)

    async def _submit_noraise(self, nodes: Iterable[int], node_type: str, noraise: bool) -> None:
        try:
            await self._submit(nodes, node_type)
        except Exception as e:
            if noraise:
                self.log.exception("while requesting to heal %s of type %s", nodes, node_type)
            else:
                raise e from None

    async def _submit(self, nodes: Iterable[int], node_type: str) -> None:
        if self._session is None:
            return
        ts = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        messages = [
            PubsubMessage(
                json.dumps(
                    {
                        "athenian_acc_id": acc_id,
                        "flags": ["recursive", "force", "important"],
                        "event_id": f"{ts}_acc_{acc_id}_heal_{node_type}",
                        "gids": [f"{node}:{node_type}" for node in nodes],
                    },
                ),
            )
            for acc_id in self._meta_ids
        ]
        if not messages:
            return
        self.log.info("requesting to heal: %s", messages)
        self.log.info("pubsub response: %s", await self._client.publish(self._topic, messages))
