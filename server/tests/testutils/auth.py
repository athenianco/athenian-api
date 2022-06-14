from typing import Any, Dict
from unittest import mock

from athenian.api.auth import Auth0


def mock_auth0():
    async def fake_extract_bearer_token(self, token: str) -> Dict[str, Any]:
        if token == "null":
            return {"sub": self.force_user or self._default_user_id}
        return {"sub": token}

    return mock.patch.object(Auth0, "_extract_bearer_token", new=fake_extract_bearer_token)
