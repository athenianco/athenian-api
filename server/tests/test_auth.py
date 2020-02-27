import pickle

from aiohttp.web_runner import GracefulExit
import pytest

from athenian.api import Auth0
from athenian.api.cache import _gen_cache_key
from athenian.api.models.web import User


class DefaultAuth0(Auth0):
    def __init__(self, whitelist, cache=None):
        super().__init__(whitelist=whitelist, default_user="github|60340680")


async def test_default_user(loop):
    auth0 = Auth0(default_user="github|60340680")
    user = await auth0.default_user()
    assert user.name == "Groundskeeper Willie"
    auth0._default_user = None
    auth0._default_user_id = "abracadabra"
    with pytest.raises(GracefulExit):
        await auth0.default_user()


async def test_cache_userinfo(cache, loop):
    auth0 = Auth0(default_user="github|60340680", cache=cache, lazy=True)
    profile = {
        "sub": "auth0|5e1f6e2e8bfa520ea5290741",
        "email": "eiso@athenian.co",
        "name": "Eiso Kant",
        "updated_at": "2020-02-01T12:00:00Z",
        "picture": "https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
    }
    user = User.from_auth0(**profile)
    await cache.set(_gen_cache_key("athenian.api.auth.Auth0._get_user_info_cached|whatever"),
                    pickle.dumps(user))
    user = await auth0._get_user_info("whatever")
    assert user.name == "Eiso Kant"
