from datetime import datetime, timezone
import pickle

from aiohttp.web_runner import GracefulExit
import lz4.frame
import pytest
from sqlalchemy import insert, update

from athenian.api.auth import Auth0
from athenian.api.cache import gen_cache_key
from athenian.api.models.state.models import Account, God, UserToken
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
    user = User.from_auth0(**profile, encryption_key="athenian")
    await cache.set(
        gen_cache_key("athenian.api.auth.Auth0._get_user_info_cached|1|whatever"),
        lz4.frame.compress(pickle.dumps(user)),
    )
    user = await auth0._get_user_info("whatever")
    assert user.name == "Eiso Kant"
    assert user.login == "eiso"
    assert (
        user.email
        == "61fd2d0f938c78cb93892a60ce5e9757b749d04f5c87e8fd67f3da2d2ecbba293fea15907ea9afe710"
    )


@pytest.mark.parametrize(
    "token",
    [
        "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1FTkNPVVl6UVRZeVFqTXhSVVF"
        "5TVRGR04wWkNNVUV5UXpCR05rUTJPVEF6TnpRMlJUUXdPQSJ9.eyJpc3MiOiJodHRwczovL2"
        "F0aGVuaWFuLXByb2R1Y3Rpb24uYXV0aDAuY29tLyIsInN1YiI6ImdpdGh1YnwyNzkzNTUxIi"
        "wiYXVkIjpbImh0dHBzOi8vYXBpLmF0aGVuaWFuLmNvIiwiaHR0cHM6Ly9hdGhlbmlhbi1wcm"
        "9kdWN0aW9uLmF1dGgwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE1OTIzMzg1NjIsImV4cCI6MT"
        "U5MjQyNDk2MiwiYXpwIjoibUk1OVFoZ1JjN2UzREdVSmR1V0NEV3d5bkdVbEhib1AiLCJzY29"
        "wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIn0.QV7rzVCuioeI3u94vzbqBHLHgjjmpnfwksbt"
        "-phtmMHwmFpzAVOtuNx18VMPhUVp33-qN2ao_azAIa8lHeqVVwgjCod8-ZAHMvEUAz7clEDlW"
        "Dtrj-ZAe3jTNBX-qn8Svljty4PNP4DD6Wiwr6EzA2bF3zSD9UPgBbx1Msag9TQSOjbBpd6iW3"
        "V-FOcS-7CZaTeqMwc5vnt5QdVtAPiKLSGYQTOY-D5D6Q85mDLAszqOPhwWhCm4-2sepJbHk5"
        "zyngzRWgdFlse7-ifW9DmrY6xwOkNZBa8AUc8G6e3KUpXugDkeyp6wDzXl5uyUXoM4OKrTTS"
        "sW5wGz1FE43DXKNg",
        "abc",
    ],
)
async def test_wrong_token(client, headers, token):
    headers = headers.copy()
    headers["Authorization"] = "Bearer " + token
    response = await client.request(method="GET", path="/v1/user", headers=headers, json={})
    assert response.status == 401


async def test_set_account_from_token(client, headers, sdb):
    await sdb.execute(
        insert(UserToken).values(
            UserToken(account_id=1, user_id="auth0|5e1f6e2e8bfa520ea5290741", name="xxx")
            .create_defaults()
            .explode(),
        ),
    )
    body = {
        "date_from": "2016-01-01",
        "date_to": "2020-01-01",
    }
    headers = headers.copy()
    headers["X-API-Key"] = "AQAAAAAAAAA="
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body,
    )
    assert response.status == 200


async def test_broken_json(client, headers):
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, data=b"hola",
    )
    assert response.status == 400


async def test_account_expiration_regular(client, headers, sdb):
    await sdb.execute(update(Account).values({Account.expires_at: datetime.now(timezone.utc)}))
    body = {
        "account": 1,
        "name": "this should fail",
    }
    response = await client.request(
        method="POST", path="/v1/token/create", headers=headers, json=body,
    )
    assert response.status == 401


@pytest.mark.parametrize(
    "mapped_id, status",
    [
        ("auth0|62a1ae88b6bba16c6dbc6870", 401),
        ("auth0|5e1f6e2e8bfa520ea5290741", 200),
    ],
)
async def test_account_expiration_god(client, headers, sdb, mapped_id, status):
    await sdb.execute(update(Account).values({Account.expires_at: datetime.now(timezone.utc)}))
    await sdb.execute(
        insert(God).values(
            God(
                user_id="auth0|62a1ae88b6bba16c6dbc6870",
                mapped_id=mapped_id,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    body = {
        "account": 1,
        "name": "this should fail",
    }
    response = await client.request(
        method="POST", path="/v1/token/create", headers=headers, json=body,
    )
    assert response.status == status
