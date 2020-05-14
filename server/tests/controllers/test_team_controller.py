import json

import pytest
from sqlalchemy import select

from athenian.api import ResponseError
from athenian.api.models.state.models import Team
from athenian.api.models.web.team_create_request import TeamCreateRequest


@pytest.mark.parametrize("account", [1, 2], ids=["as admin", "as non-admin"])
async def test_create_team(client, headers, sdb, account):
    body = TeamCreateRequest(account, "Engineering", ["github.com/se7entyse7en"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )

    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body

    assert len(await sdb.fetch_all(select([Team]))) == 1
    team = await sdb.fetch_one(select([Team]).where(Team.id == json.loads(body)["id"]))
    _test_same_team(team, {
        "id": 1,
        "members": ["github.com/se7entyse7en"],
        "members_checksum": -112991949876077516,
        "members_count": 1,
        "name": "Engineering",
        "owner": account,
    })


@pytest.mark.parametrize("account", [3, 4], ids=["not a member", "invalid account"])
async def test_create_team_wrong_account(client, headers, sdb, account):
    body = TeamCreateRequest(account, "Engineering", ["github.com/se7entyse7en"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )

    body = (await response.read()).decode("utf-8")
    assert response.status == 404, "Response body is : " + body
    parsed = json.loads((await response.read()).decode("utf-8"))
    assert parsed == {
        "type": "/errors/NotFoundError",
        "title": "Not Found",
        "status": 404,
        "detail": (f"Account {account} does not exist or user auth0|5e1f6dfb57bc640ea390557b "
                   "is not a member."),
    }

    assert len(await sdb.fetch_all(select([Team]))) == 0


async def test_create_team_wrong_member(client, headers, sdb):
    body = TeamCreateRequest(1, "Engineering",
                             ["github.com/se7entyse7en/foo",
                              "github.com/vmarkovtsev/bar",
                              "github.com/warenlg"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )

    body = (await response.read()).decode("utf-8")
    assert response.status == 400, "Response body is : " + body
    parsed = json.loads((await response.read()).decode("utf-8"))
    assert parsed == {
        "type": "/errors/BadRequest",
        "title": "Bad Request",
        "status": 400,
        "detail": "invalid members: github.com/se7entyse7en/foo, github.com/vmarkovtsev/bar",
    }

    assert len(await sdb.fetch_all(select([Team]))) == 0


async def test_create_team_same_members(client, headers, sdb):
    body = TeamCreateRequest(1, "Engineering 1",
                             ["github.com/se7entyse7en",
                              "github.com/vmarkovtsev"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )

    body = TeamCreateRequest(1, "Engineering 2",
                             ["github.com/vmarkovtsev",
                              "github.com/se7entyse7en"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )

    body = (await response.read()).decode("utf-8")
    assert response.status == 409, "Response body is : " + body
    parsed = json.loads(body)
    assert parsed == {
        "type": "/errors/DatabaseConflict",
        "title": "Conflict",
        "status": 409,
        "detail": ("this team already exists: UNIQUE constraint failed: "
                   "teams.owner, teams.members_checksum"),
    }

    teams = await sdb.fetch_all(select([Team]))
    assert len(teams) == 1
    _test_same_team(teams[0], {
        "id": 1,
        "members": ["github.com/se7entyse7en", "github.com/vmarkovtsev"],
        "members_checksum": 1112332547918387545,
        "members_count": 2,
        "name": "Engineering 1",
        "owner": 1,
    })


async def test_create_team_same_name(client, headers, sdb):
    body = TeamCreateRequest(1, "Engineering",
                             ["github.com/se7entyse7en"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )

    body = TeamCreateRequest(1, "Engineering",
                             ["github.com/vmarkovtsev"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )

    body = (await response.read()).decode("utf-8")
    assert response.status == 409, "Response body is : " + body
    parsed = json.loads(body)
    assert parsed == {
        "type": "/errors/DatabaseConflict",
        "title": "Conflict",
        "status": 409,
        "detail": ("this team already exists: UNIQUE constraint failed: "
                   "teams.owner, teams.name"),
    }

    teams = await sdb.fetch_all(select([Team]))
    assert len(teams) == 1
    _test_same_team(teams[0], {
        "id": 1,
        "members": ["github.com/se7entyse7en"],
        "members_checksum": -112991949876077516,
        "members_count": 1,
        "name": "Engineering",
        "owner": 1,
    })


def _test_same_team(actual, expected, no_timings=True):
    if not isinstance(actual, dict):
        actual = dict(actual)

    if no_timings:
        actual.pop("created_at", None)
        actual.pop("updated_at", None)
        expected.pop("created_at", None)
        expected.pop("updated_at", None)

    assert actual == expected
