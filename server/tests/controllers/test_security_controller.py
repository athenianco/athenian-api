from datetime import datetime
import json

from aiohttp import ClientResponse
import pytest
from sqlalchemy import insert, select

from athenian.api.db import Database
from athenian.api.models.state.models import UserToken
from athenian.api.models.web import CreatedToken, ListedToken
from tests.testutils.auth import force_request_auth
from tests.testutils.db import models_insert
from tests.testutils.factory.state import UserAccountFactory
from tests.testutils.requester import Requester

_USER_ID = "github|1"


class TestCreateToken(Requester):
    path = "/v1/token/create"

    @pytest.fixture(scope="function", autouse=True)
    async def _create_user(self, sdb):
        await models_insert(sdb, UserAccountFactory(user_id=_USER_ID))

    async def _request(self, *args, user_id: str | None = _USER_ID, **kwargs) -> ClientResponse:
        with force_request_auth(user_id, self.headers) as headers:
            return await super()._request(*args, headers=headers, **kwargs)

    async def test_auth(self, client, app, sdb, eiso_user):
        body = {"account": 1, "name": "xxx"}
        res = await self.post_json(json=body, user_id="auth0|5e1f6e2e8bfa520ea5290741")
        response = CreatedToken.from_dict(res)
        assert response.id == 1
        token = response.token
        assert token == "AQAAAAAAAAA="  # unencrypted

        app._auth0._default_user_id = None
        _set_user = app._auth0._set_user

        async def _new_set_user(request, token, method):
            await _set_user(request, token, method)

            async def get_user():
                return eiso_user

            request.user = get_user

        app._auth0._set_user = _new_set_user

        headers = self.headers.copy()
        headers["X-API-Key"] = token
        response = await self.client.request(
            method="GET", path="/v1/user", headers=headers, json={},
        )
        assert response.status == 200
        res_data = await response.json()
        del res_data["updated"]
        assert res_data["id"] == "auth0|5e1f6e2e8bfa520ea5290741"

    @pytest.mark.parametrize(
        "account, name, code",
        [(2, "yyy", 200), (3, "zzz", 404), (1, "", 400), (1, "x" * 257, 400)],
    )
    async def test_nasty_input(self, sdb, account, name, code):
        await models_insert(sdb, UserAccountFactory(user_id=_USER_ID, account_id=2))
        await self.post_json(json={"account": account, "name": name}, assert_status=code)

    async def test_same_name(self):
        body = {"account": 1, "name": "xxx"}
        await self.post_json(json=body)
        await self.post_json(json=body, assert_status=409)

    async def test_default_user_forbidden(self, sdb: Database) -> None:
        body = {"account": 1, "name": "xxx"}
        await self.post_json(assert_status=403, json=body, user_id=None)


async def test_delete_token_smoke(client, headers, sdb, disable_default_user):
    body = {"account": 1, "name": "xxx"}
    response = await client.request(
        method="POST", path="/v1/token/create", headers=headers, json=body,
    )
    assert response.status == 200
    token_id = (await response.json())["id"]
    response = await client.request(
        method="DELETE", path="/v1/token/%d" % token_id, headers=headers, json={},
    )
    assert response.status == 200
    assert await response.json() == {}
    assert len(await sdb.fetch_all(select([UserToken.id]))) == 0


@pytest.mark.parametrize("token_id, code", [(20, 404), (1, 404)])
async def test_delete_token_nasty_input(client, headers, sdb, token_id, code):
    await sdb.execute(
        insert(UserToken).values(
            UserToken(account_id=3, user_id="auth0|5e1f6e2e8bfa520ea5290741", name="xxx")
            .create_defaults()
            .explode(),
        ),
    )
    response = await client.request(
        method="DELETE", path="/v1/token/%d" % token_id, headers=headers, json={},
    )
    assert response.status == code


async def test_patch_token_smoke(client, headers, sdb):
    await sdb.execute(
        insert(UserToken).values(
            UserToken(account_id=1, user_id="auth0|62a1ae88b6bba16c6dbc6870", name="xxx")
            .create_defaults()
            .explode(),
        ),
    )
    body = {
        "name": "yyy",
    }
    response = await client.request(
        method="PATCH", path="/v1/token/1", headers=headers, json=body,
    )
    assert response.status == 200
    assert json.loads((await response.read()).decode("utf-8")) == {}
    assert await sdb.fetch_val(select([UserToken.name])) == "yyy"


@pytest.mark.parametrize(
    "token_id, name, code",
    [
        (1, "xxx1", 200),
        (1, "", 400),
        (1, "x" * 257, 400),
        (1, "xxx2", 409),
        (3, "yyyy", 404),
        (10, "zzz", 404),
    ],
)
async def test_patch_token_nasty_input(client, headers, sdb, token_id, name, code):
    await sdb.execute(
        insert(UserToken).values(
            UserToken(account_id=1, user_id="auth0|62a1ae88b6bba16c6dbc6870", name="xxx1")
            .create_defaults()
            .explode(),
        ),
    )
    await sdb.execute(
        insert(UserToken).values(
            UserToken(account_id=1, user_id="auth0|62a1ae88b6bba16c6dbc6870", name="xxx2")
            .create_defaults()
            .explode(),
        ),
    )
    await sdb.execute(
        insert(UserToken).values(
            UserToken(account_id=3, user_id="auth0|5e1f6e2e8bfa520ea5290741", name="xxx3")
            .create_defaults()
            .explode(),
        ),
    )
    body = {
        "name": name,
    }
    response = await client.request(
        method="PATCH", path="/v1/token/%d" % token_id, headers=headers, json=body,
    )
    assert response.status == code


async def test_list_tokens_smoke(client, headers, sdb):
    response = await client.request(method="GET", path="/v1/tokens/1", headers=headers, json={})
    assert response.status == 200
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == []
    await sdb.execute(
        insert(UserToken).values(
            UserToken(account_id=1, user_id="auth0|62a1ae88b6bba16c6dbc6870", name="xxx1")
            .create_defaults()
            .explode(),
        ),
    )
    await sdb.execute(
        insert(UserToken).values(
            UserToken(account_id=1, user_id="auth0|5e1f6e2e8bfa520ea5290741", name="xxx2")
            .create_defaults()
            .explode(),
        ),
    )
    await sdb.execute(
        insert(UserToken).values(
            UserToken(account_id=3, user_id="auth0|5e1f6e2e8bfa520ea5290741", name="xxx3")
            .create_defaults()
            .explode(),
        ),
    )
    response = await client.request(method="GET", path="/v1/tokens/1", headers=headers, json={})
    body = [ListedToken.from_dict(i) for i in json.loads((await response.read()).decode("utf-8"))]
    assert len(body) == 1
    assert body[0].id == 1
    assert body[0].name == "xxx1"
    assert isinstance(body[0].last_used, datetime)


@pytest.mark.parametrize("account, code", [(2, 200), (3, 404), (10, 404)])
async def test_list_tokens_nasty_input(client, headers, account, code):
    response = await client.request(
        method="GET", path="/v1/tokens/%d" % account, headers=headers, json={},
    )
    assert response.status == code
