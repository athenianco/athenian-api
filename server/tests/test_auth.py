import json
import time
from typing import Optional

from aiohttp.web_runner import GracefulExit
import pytest

from athenian.api import Auth0


class DefaultAuth0(Auth0):
    def __init__(self, whitelist, cache=None):
        super().__init__(whitelist=whitelist, default_user="github|60340680")


class FakeCache:
    def __init__(self):
        self._dict = {}

    async def get(self, key: bytes, default: Optional[bytes] = None) -> Optional[bytes]:
        assert isinstance(key, bytes)
        assert default is None or isinstance(default, bytes)
        if key not in self._dict:
            return default
        value, start, exp = self._dict[key]
        if exp < 0 or 0 < exp < time.time() - start:
            return default
        return value

    async def set(self, key: bytes, value: bytes, exptime: int = 0) -> bool:
        assert isinstance(key, bytes)
        assert isinstance(value, bytes)
        assert isinstance(exptime, int)
        self._dict[key] = value, time.time(), exptime
        return True


async def test_default_user(loop):
    auth0 = Auth0(default_user="github|60340680")
    user = await auth0.default_user()
    assert user.name == "Groundskeeper Willie"
    auth0._default_user = None
    auth0._default_user_id = "abracadabra"
    with pytest.raises(GracefulExit):
        await auth0.default_user()


async def test_cache_userinfo(loop):
    cache = FakeCache()
    auth0 = Auth0(default_user="github|60340680", cache=cache, lazy=True)
    profile = {
        "sub": "auth0|5e1f6e2e8bfa520ea5290741",
        "email": "eiso@athenian.co",
        "name": "Eiso Kant",
        "updated_at": "2020-02-01T12:00:00Z",
        "picture": "https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
    }
    await cache.set(b"whatever", json.dumps(profile).encode())
    user = await auth0._get_user_info("whatever")
    assert user.name == "Eiso Kant"
