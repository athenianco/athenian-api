from datetime import datetime
import json

import dateutil.parser


async def test_get_user(client):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="GET", path="/v1/user", headers=headers, json={},
    )
    body = (await response.read()).decode("utf-8")
    items = json.loads(body)
    updated = items["updated"]
    del items["updated"]
    assert items == {
        "id": "auth0|5e1f6dfb57bc640ea390557b",
        "email": "vadim@athenian.co",
        "name": "Vadim Markovtsev",
        "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
        "teams": {"1": True, "2": False},
    }
    assert datetime.utcnow() >= dateutil.parser.parse(updated)


async def test_get_team(client):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="GET", path="/v1/team/1", headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert len(body["admins"]) == 1
    assert body["admins"][0]["email"] == "vadim@athenian.co"
    assert len(body["regulars"]) == 1
    assert body["regulars"][0]["email"] == "eiso@athenian.co"
