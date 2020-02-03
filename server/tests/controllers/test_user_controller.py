import asyncio
from datetime import datetime
import json

import dateutil.parser


async def test_get_user(client, headers):
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
        "name": "Vadim Markovtsev",
        "native_id": "5e1f6dfb57bc640ea390557b",
        "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
        "accounts": {"1": True, "2": False},
    }
    assert datetime.utcnow() >= dateutil.parser.parse(updated[:-1])


async def test_get_account(client, headers):
    response = await client.request(
        method="GET", path="/v1/account/1", headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert len(body["admins"]) == 1
    assert body["admins"][0]["email"] == "vadim@athenian.co"
    assert len(body["regulars"]) == 1
    assert body["regulars"][0]["email"] == "eiso@athenian.co"


async def test_get_users_query_size_limit(xapp):
    users = await xapp._auth0.get_users(
        ["auth0|5e1f6dfb57bc640ea390557b"] * 200 + ["auth0|5e1f6e2e8bfa520ea5290741"] * 200)
    assert len(users) == 2
    assert users["auth0|5e1f6dfb57bc640ea390557b"].email == "vadim@athenian.co"
    assert users["auth0|5e1f6e2e8bfa520ea5290741"].email == "eiso@athenian.co"


async def test_get_users_rate_limit(xapp):
    users = await asyncio.gather(*[xapp._auth0.get_user("auth0|5e1f6dfb57bc640ea390557b")
                                   for _ in range(20)])
    for u in users:
        assert u.email == "vadim@athenian.co"


async def test_become(client, headers):
    response = await client.request(
        method="GET", path="/v1/become?id=auth0|5e1f6e2e8bfa520ea5290741", headers=headers, json={},
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
        "email": "eiso@athenian.co",
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
        "email": "vadim@athenian.co",
        "name": "Vadim Markovtsev",
        "native_id": "5e1f6dfb57bc640ea390557b",
        "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
        "accounts": {"1": True, "2": False},
    }
