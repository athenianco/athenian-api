from datetime import datetime, timezone

from sqlalchemy import update

from athenian.api.models.state.models import Account
from tests.align.utils import align_graphql_request


async def test_auth_default(client, headers):
    body = {"query": "{__typename}"}
    response = await align_graphql_request(client, headers=headers, json=body)
    assert response == {"data": {"__typename": "Query"}}


async def test_auth_failure(client, headers):
    headers["Authorization"] = "Bearer invalid"
    body = {"query": "{__typename}"}
    response = await align_graphql_request(client, headers=headers, json=body)
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
    response = await align_graphql_request(client, headers=headers, json=body)
    assert response == {
        "errors": [{
            "message": "Not Found",
            "locations": [{"line": 1, "column": 42}],
            "path": ["goals"],
            "extensions": {
                "status": 404,
                "type": "/errors/AccountNotFound",
                "detail": "Account 3 does not exist or user auth0|62a1ae88b6bba16c6dbc6870 is "
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
    response = await align_graphql_request(client, headers=headers, json=body)
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
