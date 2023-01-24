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
    VAR_NAME = "refetcher"

    def __init__(self, topic: str | None, meta_ids: tuple[int, ...] = ()):
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

    def specialize(self, meta_ids: tuple[int, ...]) -> "Refetcher":
        """Clone the current instance with different metadata account IDs."""
        clone = object.__new__(Refetcher)
        clone._meta_ids = meta_ids
        clone._session = self._session
        clone._client = self._client
        clone._topic = self._topic
        return clone

    async def close(self) -> None:
        """Free any resources in use by the instance."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def submit_commits(self, nodes: Iterable[int], noraise: bool = False) -> None:
        """Send commit node IDs to refetch."""
        await self._submit_noraise(nodes, "Commit", [], noraise)

    async def submit_org_members(self, nodes: Iterable[int], noraise: bool = False) -> None:
        """Send organization node IDs to refetch the members."""
        await self._submit_noraise(nodes, "Organization", ["membersWithRole"], noraise)

    async def _submit_noraise(
        self,
        nodes: Iterable[int],
        node_type: str,
        fields: list[str],
        noraise: bool,
    ) -> None:
        try:
            await self._submit(nodes, node_type, fields)
        except Exception as e:
            if noraise:
                self.log.exception("while requesting to heal %s of type %s", nodes, node_type)
            else:
                raise e from None

    async def _submit(self, nodes: Iterable[int], node_type: str, fields: list[str]) -> None:
        if self._session is None or not (nodes := [f"{node}:{node_type}" for node in nodes]):
            return
        if not self._meta_ids:
            self.log.warning("attempted to heal using an unspecialized instance")
            return
        ts = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        fields = {"fields": fields} if fields else {}
        messages = [
            PubsubMessage(
                json.dumps(
                    {
                        "athenian_acc_id": acc_id,
                        "flags": ["recursive", "force", "important"],
                        "event_id": f"{ts}_acc_{acc_id}_heal_{node_type}",
                        "gids": nodes,
                        **fields,
                    },
                ),
            )
            for acc_id in self._meta_ids
        ]
        for acc_id in self._meta_ids:
            self.log.info(
                "requesting to heal %s_acc_%s_heal_%s : %s", ts, acc_id, node_type, nodes,
            )
        self.log.info("pubsub response: %s", await self._client.publish(self._topic, messages))
