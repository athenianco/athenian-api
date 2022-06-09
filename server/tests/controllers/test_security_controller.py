from datetime import datetime
import json
import random

import pytest
from sqlalchemy import insert, select

from athenian.api.models.state.models import UserToken
from athenian.api.models.web import CreatedToken, ListedToken
from tests.controllers.test_user_controller import vadim_email


@pytest.mark.flaky(reruns=5, reruns_delay=random.uniform(0.5, 2.5))
async def test_create_token_auth(client, headers, app):
    body = {
        "account": 1,
        "name": "xxx",
    }
    response = await client.request(
        method="POST", path="/v1/token/create", headers=headers, json=body,
    )
    response = CreatedToken.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert response.id == 1
    token = response.token
    assert token == "AQAAAAAAAAA="  # unencrypted
    app._auth0._default_user_id = None
    headers = headers.copy()
    headers["X-API-Key"] = token
    response = await client.request(
        method="GET", path="/v1/user", headers=headers, json={},
    )
    assert response.status == 200
    response = json.loads((await response.read()).decode("utf-8"))
    del response["updated"]
    assert response == {
        "id": "auth0|62a1ae88b6bba16c6dbc6870",
        "email": vadim_email,
        "name": "Vadim Markovtsev",
        "login": "vadim",
        "native_id": "62a1ae88b6bba16c6dbc6870",
        "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
        "accounts": {
            "1": {"is_admin": True,
                  "expired": False,
                  "has_ci": True,
                  "has_jira": True,
                  "has_deployments": True,
                  },
            "2": {"is_admin": False,
                  "expired": False,
                  "has_ci": False,
                  "has_jira": False,
                  "has_deployments": False,
                  },
        },
    }


@pytest.mark.parametrize("account, name, code", [(2, "yyy", 200),
                                                 (3, "zzz", 404),
                                                 (1, "", 400),
                                                 (1, "x" * 257, 400)])
async def test_create_token_nasty_input(client, headers, account, name, code):
    body = {
        "account": account,
        "name": name,
    }
    response = await client.request(
        method="POST", path="/v1/token/create", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert response.status == code, body


async def test_create_token_same_name(client, headers):
    body = {
        "account": 1,
        "name": "xxx",
    }
    response = await client.request(
        method="POST", path="/v1/token/create", headers=headers, json=body,
    )
    assert response.status == 200
    response = await client.request(
        method="POST", path="/v1/token/create", headers=headers, json=body,
    )
    assert response.status == 409


async def test_delete_token_smoke(client, headers, sdb):
    body = {
        "account": 1,
        "name": "xxx",
    }
    response = await client.request(
        method="POST", path="/v1/token/create", headers=headers, json=body,
    )
    token_id = json.loads((await response.read()).decode("utf-8"))["id"]
    response = await client.request(
        method="DELETE", path="/v1/token/%d" % token_id, headers=headers, json={},
    )
    assert response.status == 200
    assert json.loads((await response.read()).decode("utf-8")) == {}
    assert len(await sdb.fetch_all(select([UserToken.id]))) == 0


@pytest.mark.parametrize("token_id, code", [(20, 404), (1, 404)])
async def test_delete_token_nasty_input(client, headers, sdb, token_id, code):
    await sdb.execute(insert(UserToken).values(UserToken(
        account_id=3, user_id="auth0|5e1f6e2e8bfa520ea5290741", name="xxx",
    ).create_defaults().explode()))
    response = await client.request(
        method="DELETE", path="/v1/token/%d" % token_id, headers=headers, json={},
    )
    assert response.status == code


async def test_patch_token_smoke(client, headers, sdb):
    await sdb.execute(insert(UserToken).values(UserToken(
        account_id=1, user_id="auth0|62a1ae88b6bba16c6dbc6870", name="xxx",
    ).create_defaults().explode()))
    body = {
        "name": "yyy",
    }
    response = await client.request(
        method="PATCH", path="/v1/token/1", headers=headers, json=body,
    )
    assert response.status == 200
    assert json.loads((await response.read()).decode("utf-8")) == {}
    assert await sdb.fetch_val(select([UserToken.name])) == "yyy"


@pytest.mark.parametrize("token_id, name, code", [(1, "xxx1", 200),
                                                  (1, "", 400),
                                                  (1, "x" * 257, 400),
                                                  (1, "xxx2", 409),
                                                  (3, "yyyy", 404),
                                                  (10, "zzz", 404)])
async def test_patch_token_nasty_input(client, headers, sdb, token_id, name, code):
    await sdb.execute(insert(UserToken).values(UserToken(
        account_id=1, user_id="auth0|62a1ae88b6bba16c6dbc6870", name="xxx1",
    ).create_defaults().explode()))
    await sdb.execute(insert(UserToken).values(UserToken(
        account_id=1, user_id="auth0|62a1ae88b6bba16c6dbc6870", name="xxx2",
    ).create_defaults().explode()))
    await sdb.execute(insert(UserToken).values(UserToken(
        account_id=3, user_id="auth0|5e1f6e2e8bfa520ea5290741", name="xxx3",
    ).create_defaults().explode()))
    body = {
        "name": name,
    }
    response = await client.request(
        method="PATCH", path="/v1/token/%d" % token_id, headers=headers, json=body,
    )
    assert response.status == code


async def test_list_tokens_smoke(client, headers, sdb):
    response = await client.request(
        method="GET", path="/v1/tokens/1", headers=headers, json={},
    )
    assert response.status == 200
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == []
    await sdb.execute(insert(UserToken).values(UserToken(
        account_id=1, user_id="auth0|62a1ae88b6bba16c6dbc6870", name="xxx1",
    ).create_defaults().explode()))
    await sdb.execute(insert(UserToken).values(UserToken(
        account_id=1, user_id="auth0|5e1f6e2e8bfa520ea5290741", name="xxx2",
    ).create_defaults().explode()))
    await sdb.execute(insert(UserToken).values(UserToken(
        account_id=3, user_id="auth0|5e1f6e2e8bfa520ea5290741", name="xxx3",
    ).create_defaults().explode()))
    response = await client.request(
        method="GET", path="/v1/tokens/1", headers=headers, json={},
    )
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
