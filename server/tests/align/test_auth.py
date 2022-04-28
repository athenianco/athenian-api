from datetime import datetime, timezone
import json

from sqlalchemy import update

from athenian.api.models.state.models import Account


async def test_auth_default(client, headers):
    body = {
        "query": "{__typename}",
    }
    response = await client.request(
        method="POST", path="/align/graphql", headers=headers, json=body)
    assert response.status == 200
    response = json.loads((await response.read()).decode("utf-8"))
    assert response == {"data": {"__typename": "Query"}}


async def test_auth_failure(client, headers):
    headers["Authorization"] = "Bearer invalid"
    body = {
        "query": "{__typename}",
    }
    response = await client.request(
        method="POST", path="/align/graphql", headers=headers, json=body)
    assert response.status == 200
    response = json.loads((await response.read()).decode("utf-8"))
    assert response == {
        "errors": [{
            "extensions": {
                "detail": "Invalid header: Error decoding token headers. "
                          "Use an RS256 signed JWT Access Token.",
                "type": "/errors/Unauthorized",
                "status": 401},
            "message": "Unauthorized"},
        ],
    }


async def test_auth_account_mismatch(client, headers):
    body = {
        "query": "query goals($account: Int!, $team: Int!)"
                 "{goals(accountId: $account, teamId: $team){id}}",
        "variables": {"account": 3, "team": 1},
    }
    response = await client.request(
        method="POST", path="/align/graphql", headers=headers, json=body)
    assert response.status == 200
    response = json.loads((await response.read()).decode("utf-8"))
    assert response == {
        "errors": [{
            "message": "Not Found",
            "locations": [{"line": 1, "column": 42}],
            "path": ["goals"],
            "extensions": {
                "status": 404,
                "type": "/errors/AccountNotFound",
                "detail": "Account 3 does not exist or user auth0|5e1f6dfb57bc640ea390557b is "
                          "not a member.",
            },
        }],
    }


async def test_auth_account_expired(client, headers, sdb):
    body = {
        "query": "query goals($account: Int!, $team: Int!)"
                 "{goals(accountId: $account, teamId: $team){id}}",
        "variables": {"account": 1, "team": 1},
    }
    await sdb.execute(update(Account).values({Account.expires_at: datetime.now(timezone.utc)}))
    response = await client.request(
        method="POST", path="/align/graphql", headers=headers, json=body)
    assert response.status == 200
    response = json.loads((await response.read()).decode("utf-8"))
    assert response == {
        "errors": [{
            "message": "Unauthorized",
            "locations": [{"line": 1, "column": 42}],
            "path": ["goals"],
            "extensions": {
                "status": 401,
                "type": "/errors/Unauthorized",
                "detail": "Account 1 has expired.",
            },
        }],
    }
