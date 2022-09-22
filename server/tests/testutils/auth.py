import contextlib
from typing import Any, Optional
from unittest import mock

from athenian.api.auth import Auth0


@contextlib.contextmanager
def force_request_auth(user_id: Optional[str], headers: dict):
    """Context manager to force authentication for a given user.

    If user_id is None default user will be authenticated.
    Headers to be sent in the request are given on __enter__.

    """
    if user_id is None:
        # no mock or header needed, default user will be used in request
        yield headers
    else:
        headers = headers.copy()
        headers["Authorization"] = f"Bearer {user_id}"
        with mock_auth0():
            yield headers


def mock_auth0():
    async def fake_extract_bearer_token(self, token: str) -> dict[str, Any]:
        if token == "null":
            return {"sub": self.force_user or self._default_user_id}
        return {"sub": token}

    return mock.patch.object(Auth0, "_extract_bearer_token", new=fake_extract_bearer_token)
