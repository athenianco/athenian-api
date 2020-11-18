from datetime import datetime, timezone
import json

from aiohttp import web
import dateutil.parser
import pytest
from sqlalchemy import insert, update

from athenian.api.async_utils import gather
from athenian.api.controllers.user_controller import get_user
from athenian.api.models.state.models import AccountFeature, Feature, God
from athenian.api.models.web import Account, ProductFeature
from athenian.api.request import AthenianWebRequest
from tests.conftest import disable_default_user


async def test_get_user_smoke(client, headers):
    response = await client.request(
        method="GET", path="/v1/user", headers=headers, json={},
    )
    assert response.status == 200
    body = (await response.read()).decode("utf-8")
    items = json.loads(body)
    updated = items["updated"]
    del items["updated"]
    assert items == {
        "id": "auth0|5e1f6dfb57bc640ea390557b",
        "email": "vadim@athenian.co",
        "login": "vadim",
        "name": "Vadim Markovtsev",
        "native_id": "5e1f6dfb57bc640ea390557b",
        "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
        "accounts": {"1": True, "2": False},
    }
    assert datetime.utcnow() >= dateutil.parser.parse(updated[:-1])


@pytest.mark.parametrize("value", (True, False))
async def test_is_default_user(client, headers, app, value):
    async def get_is_default_user(request: AthenianWebRequest) -> web.Response:
        return web.json_response({"is_default_user": request.is_default_user})

    for route in app.app.router.routes():
        if route.resource.canonical == "/v1/user" and route.method == "GET":
            wrapped = route.handler
            while hasattr(wrapped.__wrapped__, "__wrapped__"):
                wrapped = wrapped.__wrapped__
            wrapped.__wrapped__ = get_is_default_user
            for cell in wrapped.__closure__:
                if cell.cell_contents == get_user:
                    cell.cell_contents = get_is_default_user
    if not value:
        disable_default_user.__wrapped__(app)

    response = await client.request(
        method="GET", path="/v1/user", headers=headers, json={},
    )
    assert response.status == 200
    items = json.loads((await response.read()).decode("utf-8"))
    assert items == {"is_default_user": value}


async def test_get_default_user(client, headers, lazy_gkwillie):
    response = await client.request(
        method="GET", path="/v1/user", headers=headers, json={},
    )
    assert response.status == 200
    body = (await response.read()).decode("utf-8")
    items = json.loads(body)
    del items["updated"]
    assert items == {
        "id": "github|60340680",
        "login": "gkwillie",
        "name": "Groundskeeper Willie",
        "email": "<classified>",
        "native_id": "60340680",
        "picture": "https://avatars0.githubusercontent.com/u/60340680?v=4",
        "accounts": {},
    }


async def test_get_account_details_smoke(client, headers):
    response = await client.request(
        method="GET", path="/v1/account/1/details", headers=headers, json={},
    )
    body = Account.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert len(body.admins) == 1
    assert body.admins[0].name == "Vadim Markovtsev"
    assert body.admins[0].email == "<classified>"  # "vadim@athenian.co"
    assert len(body.regulars) == 1
    assert body.regulars[0].name == "Eiso Kant"
    assert body.regulars[0].email == "<classified>"  # "eiso@athenian.co"
    assert len(body.organizations) == 1
    assert body.organizations[0].login == "src-d"
    assert body.organizations[0].name == "source{d}"
    assert body.organizations[0].avatar_url == \
           "https://avatars3.githubusercontent.com/u/15128793?s=200&v=4"
    assert body.jira is not None
    assert body.jira.url == "https://athenianco.atlassian.net"
    assert body.jira.projects == ["CON", "CS", "DEV", "ENG", "GRW", "OPS", "PRO"]


@pytest.mark.parametrize("account, code", [[2, 200], [3, 403], [4, 404]])
async def test_get_account_details_nasty_input(client, headers, account, code):
    response = await client.request(
        method="GET", path="/v1/account/%d/details" % account, headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert response.status == code, body
    if code == 200:
        assert "jira" not in body


async def test_get_account_features_smoke(client, headers):
    response = await client.request(
        method="GET", path="/v1/account/1/features", headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    models = [ProductFeature.from_dict(p) for p in body]
    assert len(models) == 1
    assert models[0].name == "jira"
    assert models[0].parameters == {"a": "x", "c": "d"}


@pytest.mark.parametrize("account, code", [[3, 404], [4, 404]])
async def test_get_account_features_nasty_input(client, headers, account, code):
    response = await client.request(
        method="GET", path="/v1/account/%d/features" % account, headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert response.status == code, body


async def test_get_account_features_disabled(client, headers, sdb):
    await sdb.execute(update(Feature).values(
        {Feature.enabled: False, Feature.updated_at: datetime.now(timezone.utc)}))
    response = await client.request(
        method="GET", path="/v1/account/1/features", headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == []
    await sdb.execute(update(Feature).values(
        {Feature.enabled: True, Feature.updated_at: datetime.now(timezone.utc)}))
    await sdb.execute(update(AccountFeature).values(
        {AccountFeature.enabled: False, AccountFeature.updated_at: datetime.now(timezone.utc)}))
    response = await client.request(
        method="GET", path="/v1/account/1/features", headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == []


async def test_get_users_query_size_limit(xapp):
    users = await xapp._auth0.get_users(
        ["auth0|5e1f6dfb57bc640ea390557b"] * 200 + ["auth0|5e1f6e2e8bfa520ea5290741"] * 200)
    assert len(users) == 2
    assert users["auth0|5e1f6dfb57bc640ea390557b"].name == "Vadim Markovtsev"
    assert users["auth0|5e1f6dfb57bc640ea390557b"].email == "<classified>"  # "vadim@athenian.co"
    assert users["auth0|5e1f6e2e8bfa520ea5290741"].name == "Eiso Kant"
    assert users["auth0|5e1f6e2e8bfa520ea5290741"].email == "<classified>"  # "eiso@athenian.co"


async def test_get_users_rate_limit(xapp):
    users = await gather(*[xapp._auth0.get_user("auth0|5e1f6dfb57bc640ea390557b")
                           for _ in range(20)])
    for u in users:
        assert u.name == "Vadim Markovtsev"
        assert u.email == "<classified>"  # "vadim@athenian.co"


async def test_become_db(client, headers, sdb):
    await sdb.execute(insert(God).values(God(
        user_id="auth0|5e1f6dfb57bc640ea390557b",
    ).create_defaults().explode(with_primary_keys=True)))
    response = await client.request(
        method="GET", path="/v1/become?id=auth0|5e1f6e2e8bfa520ea5290741", headers=headers,
        json={},
    )
    body1 = json.loads((await response.read()).decode("utf-8"))
    response = await client.request(
        method="GET", path="/v1/user", headers=headers, json={},
    )
    body2 = json.loads((await response.read()).decode("utf-8"))
    assert body1 == body2
    del body1["updated"]
    assert body1 == {
        "id": "auth0|5e1f6e2e8bfa520ea5290741",
        "login": "eiso",
        "email": "<classified>",  # "eiso@athenian.co",
        "name": "Eiso Kant",
        "native_id": "5e1f6e2e8bfa520ea5290741",
        "picture": "https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
        "accounts": {"1": False, "3": True},
    }
    response = await client.request(
        method="GET", path="/v1/become", headers=headers, json={},
    )
    body3 = json.loads((await response.read()).decode("utf-8"))
    del body3["updated"]
    assert body3 == {
        "id": "auth0|5e1f6dfb57bc640ea390557b",
        "login": "vadim",
        "email": "<classified>",  # "vadim@athenian.co",
        "name": "Vadim Markovtsev",
        "native_id": "5e1f6dfb57bc640ea390557b",
        "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
        "accounts": {"1": True, "2": False},
    }


async def test_become_header(client, headers, sdb):
    await sdb.execute(insert(God).values(God(
        user_id="auth0|5e1f6dfb57bc640ea390557b",
    ).create_defaults().explode(with_primary_keys=True)))
    headers = headers.copy()
    headers["X-Identity"] = "auth0|5e1f6e2e8bfa520ea5290741"
    response = await client.request(
        method="GET", path="/v1/user", headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    del body["updated"]
    assert body == {
        "id": "auth0|5e1f6e2e8bfa520ea5290741",
        "login": "eiso",
        "email": "<classified>",  # "eiso@athenian.co",
        "name": "Eiso Kant",
        "native_id": "5e1f6e2e8bfa520ea5290741",
        "picture": "https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
        "accounts": {"1": False, "3": True},
    }


async def test_change_user_regular(client, headers):
    body = {
        "account": 1,
        "user": "auth0|5e1f6e2e8bfa520ea5290741",
        "status": "regular",
    }
    response = await client.request(
        method="PUT", path="/v1/account/user", headers=headers, json=body,
    )
    assert response.status == 200
    items = json.loads((await response.read()).decode("utf-8"))
    assert len(items["admins"]) == 1
    assert items["admins"][0]["id"] == "auth0|5e1f6dfb57bc640ea390557b"
    assert len(items["regulars"]) == 1
    assert items["regulars"][0]["id"] == "auth0|5e1f6e2e8bfa520ea5290741"


async def test_change_user_admin(client, headers):
    body = {
        "account": 1,
        "user": "auth0|5e1f6e2e8bfa520ea5290741",
        "status": "admin",
    }
    response = await client.request(
        method="PUT", path="/v1/account/user", headers=headers, json=body,
    )
    assert response.status == 200
    items = json.loads((await response.read()).decode("utf-8"))
    assert len(items["admins"]) == 2
    assert items["admins"][0]["id"] == "auth0|5e1f6dfb57bc640ea390557b"
    assert items["admins"][1]["id"] == "auth0|5e1f6e2e8bfa520ea5290741"


async def test_change_user_banish(client, headers):
    body = {
        "account": 1,
        "user": "auth0|5e1f6e2e8bfa520ea5290741",
        "status": "banished",
    }
    response = await client.request(
        method="PUT", path="/v1/account/user", headers=headers, json=body,
    )
    assert response.status == 200
    items = json.loads((await response.read()).decode("utf-8"))
    assert len(items["admins"]) == 1
    assert items["admins"][0]["id"] == "auth0|5e1f6dfb57bc640ea390557b"
    assert len(items["regulars"]) == 0


@pytest.mark.parametrize("account, user, status, code", [
    (1, "auth0|5e1f6dfb57bc640ea390557b", "regular", 403),
    (1, "auth0|5e1f6dfb57bc640ea390557b", "banished", 403),
    (2, "auth0|5e1f6dfb57bc640ea390557b", "regular", 403),
    (2, "auth0|5e1f6dfb57bc640ea390557b", "admin", 403),
    (2, "auth0|5e1f6dfb57bc640ea390557b", "banished", 403),
    (3, "auth0|5e1f6dfb57bc640ea390557b", "regular", 404),
])
async def test_change_user_errors(client, headers, account, user, status, code):
    body = {
        "account": account,
        "user": user,
        "status": status,
    }
    response = await client.request(
        method="PUT", path="/v1/account/user", headers=headers, json=body,
    )
    assert response.status == code
