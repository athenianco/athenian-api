import logging
from typing import Iterable, List, Tuple

import aiohttp

from athenian.api import metadata
from athenian.api.aiohttp_addons import create_aiohttp_closed_event


class MandrillError(Exception):
    """Mandrill API error."""

    def __init__(self, status: int, text: str):
        """Initialize a new instance of MandrillError."""
        super().__init__(status, text)

    @property
    def status(self) -> int:
        """Return response HTTP code."""
        return self.args[0]

    @property
    def text(self) -> str:
        """Return the raw response text."""
        return self.args[1]


class MandrillClient:
    """Mailchimp Transactional client."""

    log = logging.getLogger(f"{metadata.__package__}.Mandrill")

    def __init__(self,
                 key: str,
                 timeout: float = 10):
        """Initialize a new instance of Mandrill class."""
        self.key = key
        self._session = aiohttp.ClientSession(
            base_url="https://mandrillapp.com",
            timeout=aiohttp.ClientTimeout(total=timeout),
        )

    async def close(self):
        """Free resources and close connections associated with the object."""
        session = self._session
        all_is_lost = create_aiohttp_closed_event(session)
        await session.close()
        await all_is_lost.wait()

    async def _call(self, url: str, body: dict) -> dict:
        response = await self._session.post("/api/1.0" + url, json={
            "key": self.key,
            **body,
        })
        if response.ok:
            return await response.json()
        raise MandrillError(response.status, (await response.read()).decode("utf-8"))

    async def messages_send_template(self,
                                     template_name: str,
                                     recipients: Iterable[Tuple[str, ...]],
                                     /, **kwargs,
                                     ) -> List[bool]:
        """Fill the template with `kwargs` and send emails to the specified `recipients`."""
        response = await self._call("/messages/send-template", {
            "template_name": template_name,
            "template_content": [],
            "message": {
                "to": [
                    {"email": r[0], **({"name": r[1]} if len(r) > 1 else {})}
                    for r in recipients
                ],
                "global_merge_vars": [
                    {"name": key, "content": val} for key, val in kwargs.items()
                ],
            },
        })
        result = []
        for status in response:
            if (resolution := status["status"]) in ("rejected", "invalid"):
                reject_reason = status.get("reject_reason", "")
                self.log.error("Failed to send email to %s: %s%s",
                               status["email"],
                               resolution,
                               (" " + reject_reason) if reject_reason else "")
                result.append(False)
            else:
                result.append(True)
        return result
