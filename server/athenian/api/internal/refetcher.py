from datetime import datetime, timezone
import json
import logging
import os
from typing import Iterable

import aiohttp
from gcloud.aio.pubsub import PublisherClient, PubsubMessage
from ghid import ghid

from athenian.api import metadata


class Refetcher:
    """Sender of metadata refetch requests to PubSub."""

    log = logging.getLogger(f"{metadata.__package__}.Refetcher")
    VAR_NAME = "refetcher"

    def __init__(self, topic: str | None, meta_ids: tuple[int, ...] = ()):
        """Initialize a new instance of Refetcher class."""
        self._meta_ids = meta_ids
        self._topic_name = topic
        self._session = self._client = self._topic = None

    def specialize(self, meta_ids: tuple[int, ...]) -> "Refetcher":
        """Clone the current instance with different metadata account IDs."""
        clone = object.__new__(Refetcher)
        clone._meta_ids = meta_ids
        clone._session = self._session
        clone._client = self._client
        clone._topic_name = self._topic_name
        clone._topic = self._topic
        return clone

    async def close(self) -> None:
        """Free any resources in use by the instance."""
        if self._session and not self._session.closed:
            await self._session.close()

        self._session = None

    async def submit_commits(self, nodes: Iterable[int], noraise: bool = False) -> None:
        """Send commit node IDs to refetch."""
        await self._submit_noraise(nodes, "Commit", [], noraise, False)

    async def submit_commit_hashes(
        self,
        hashes: Iterable[tuple[int, str]],
        noraise: bool = False,
    ) -> None:
        """Send commit node IDs to refetch."""
        ids = [ghid.EncodeV2(ghid.CommitKey(*h)) for h in hashes]
        await self._submit_noraise(ids, "Commit", [], noraise, True)

    async def submit_org_members(self, nodes: Iterable[int], noraise: bool = False) -> None:
        """Send organization node IDs to refetch the members."""
        await self._submit_noraise(nodes, "Organization", ["membersWithRole"], noraise, False)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        return self._session

    async def _get_client(self) -> PublisherClient:
        if self._client is None:
            session = await self._get_session()
            self._client = PublisherClient(session=session)

        return self._client

    async def _get_topic(self) -> PublisherClient:
        if self._topic is None:
            client = await self._get_client()
            self._topic = client.topic_path(
                os.getenv("GOOGLE_CLOUD_PROJECT", os.getenv("GOOGLE_PROJECT")),
                self._topic_name,
            )

        return self._topic

    async def _submit_noraise(
        self,
        nodes: Iterable[int],
        node_type: str,
        fields: list[str],
        noraise: bool,
        as_native_ids: bool,
    ) -> None:
        try:
            await self._submit(nodes, node_type, fields, as_native_ids)
        except Exception as e:
            if noraise:
                self.log.exception(
                    "while requesting to heal %s of type %s", nodes, node_type)
            else:
                raise e from None

    async def _submit(
        self,
        nodes: Iterable[int],
        node_type: str,
        fields: list[str],
        as_native_ids: bool,
    ) -> None:
        if not self._topic_name or not (nodes := [f"{node}:{node_type}" for node in nodes]):
            return

        if not self._meta_ids:
            self.log.warning(
                "attempted to heal using an unspecialized instance")
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
                        ("ids" if as_native_ids else "gids"): nodes,
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

        client = await self._get_client()
        topic = await self._get_topic()
        self.log.info("pubsub response: %s", await client.publish(topic, messages))
