import pickle

from aiohttp.web_runner import GracefulExit
import pytest

from athenian.api import Auth0
from athenian.api.cache import gen_cache_key
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
        "nickname": "eiso",
        "name": "Eiso Kant",
        "updated_at": "2020-02-01T12:00:00Z",
        "picture": "https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
    }
    user = User.from_auth0(**profile)
    await cache.set(gen_cache_key("athenian.api.auth.Auth0._get_user_info_cached|1|whatever"),
                    pickle.dumps(user))
    user = await auth0._get_user_info("whatever")
    assert user.name == "Eiso Kant"
    assert user.login == "eiso"


@pytest.mark.parametrize("token", [
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1FTkNPVVl6UVRZeVFqTXhSVVF5TVRGR04wWkNNVUV5UXpCR"
    "05rUTJPVEF6TnpRMlJUUXdPQSJ9.eyJpc3MiOiJodHRwczovL2F0aGVuaWFuLXByb2R1Y3Rpb24uYXV0aDAuY29tLyIsI"
    "nN1YiI6ImdpdGh1YnwyNzkzNTUxIiwiYXVkIjpbImh0dHBzOi8vYXBpLmF0aGVuaWFuLmNvIiwiaHR0cHM6Ly9hdGhlbm"
    "lhbi1wcm9kdWN0aW9uLmF1dGgwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE1OTIzMzg1NjIsImV4cCI6MTU5MjQyNDk2Miw"
    "iYXpwIjoibUk1OVFoZ1JjN2UzREdVSmR1V0NEV3d5bkdVbEhib1AiLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWls"
    "In0.QV7rzVCuioeI3u94vzbqBHLHgjjmpnfwksbt-phtmMHwmFpzAVOtuNx18VMPhUVp33-qN2ao_azAIa8lHeqVVwgjC"
    "od8-ZAHMvEUAz7clEDlWDtrj-ZAe3jTNBX-qn8Svljty4PNP4DD6Wiwr6EzA2bF3zSD9UPgBbx1Msag9TQSOjbBpd6iW3"
    "V-FOcS-7CZaTeqMwc5vnt5QdVtAPiKLSGYQTOY-D5D6Q85mDLAszqOPhwWhCm4-2sepJbHk5zyngzRWgdFlse7-ifW9Dm"
    "rY6xwOkNZBa8AUc8G6e3KUpXugDkeyp6wDzXl5uyUXoM4OKrTTSsW5wGz1FE43DXKNg",
    "abc"])
async def test_wrong_token(client, headers, token):
    headers = headers.copy()
    headers["Authorization"] = "Bearer " + token
    response = await client.request(
        method="GET", path="/v1/user", headers=headers, json={},
    )
    assert response.status == 401
