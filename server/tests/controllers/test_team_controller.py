import json
from operator import attrgetter

from aiohttp.test_utils import TestClient
import pytest
from sqlalchemy import insert, select, update

from athenian.api.controllers.team_controller import get_root_team, get_team_from_db, \
    MultipleRootTeamsError, TeamNotFoundError
from athenian.api.db import Database
from athenian.api.models.state.models import AccountGitHubAccount, Team
from athenian.api.models.web import TeamUpdateRequest
from athenian.api.models.web.team import Team as TeamListItem
from athenian.api.models.web.team_create_request import TeamCreateRequest
from tests.testutils.db import model_insert_stmt
from tests.testutils.factory.state import TeamFactory


@pytest.mark.parametrize("account", [1, 2], ids=["as admin", "as non-admin"])
async def test_create_team_smoke(client, headers, sdb, account, disable_default_user):
    await sdb.execute(update(AccountGitHubAccount)
                      .where(AccountGitHubAccount.id == 6366825)
                      .values({AccountGitHubAccount.account_id: account}))
    body = TeamCreateRequest(account, "Engineering", ["github.com/se7entyse7en"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    assert len(await sdb.fetch_all(select([Team]))) == 1
    team = await sdb.fetch_one(select([Team]).where(Team.id == json.loads(rbody)["id"]))
    _test_same_team(team, {
        "id": 1,
        "members": [51],
        "name": "Engineering",
        "owner_id": account,
        "parent_id": None,
    })

    body["name"] = "Management"
    body["members"][0] = "github.com/vmarkovtsev"
    body["parent"] = 1
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    team = await sdb.fetch_one(select([Team]).where(Team.id == json.loads(rbody)["id"]))
    _test_same_team(team, {
        "id": 2,
        "members": [40020],
        "name": "Management",
        "owner_id": account,
        "parent_id": 1,
    })


async def test_create_team_bot(client, headers, sdb, disable_default_user):
    await sdb.execute(update(AccountGitHubAccount)
                      .where(AccountGitHubAccount.id == 6366825)
                      .values({AccountGitHubAccount.account_id: 1}))
    body = TeamCreateRequest(1, "Engineering", ["github.com/apps/dependabot"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    assert len(await sdb.fetch_all(select([Team]))) == 1
    team = await sdb.fetch_one(select([Team]).where(Team.id == json.loads(rbody)["id"]))
    _test_same_team(team, {
        "id": 1,
        "members": [17019778],
        "name": "Engineering",
        "owner_id": 1,
        "parent_id": None,
    })


@pytest.mark.parametrize("account", [3, 4], ids=["not a member", "invalid account"])
async def test_create_team_wrong_account(client, headers, sdb, account, disable_default_user):
    body = TeamCreateRequest(account, "Engineering", ["github.com/se7entyse7en"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )

    body = (await response.read()).decode("utf-8")
    assert response.status == 404, "Response body is : " + body
    parsed = json.loads((await response.read()).decode("utf-8"))
    assert parsed == {
        "type": "/errors/AccountNotFound",
        "title": "Not Found",
        "status": 404,
        "detail": (f"Account {account} does not exist or user auth0|5e1f6dfb57bc640ea390557b "
                   "is not a member."),
    }

    assert len(await sdb.fetch_all(select([Team]))) == 0


async def test_create_team_default_user(client, headers, sdb):
    body = TeamCreateRequest(1, "Engineering", ["github.com/se7entyse7en"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )

    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body
    assert len(await sdb.fetch_all(select([Team]))) == 0


async def test_create_team_wrong_member(client, headers, sdb, disable_default_user):
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
        "detail": "Invalid members of the team: "
                  "github.com/se7entyse7en/foo, github.com/vmarkovtsev/bar",
    }

    assert len(await sdb.fetch_all(select([Team]))) == 0


async def test_create_team_wrong_parent(client, headers, sdb, disable_default_user):
    body = TeamCreateRequest(1, "Engineering",
                             ["github.com/se7entyse7en",
                              "github.com/warenlg"],
                             1).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 400, "Response body is : " + rbody

    await sdb.execute(insert(Team).values(Team(
        owner_id=3, name="Test", members=[40020],
    ).create_defaults().explode()))
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 400, "Response body is : " + rbody


async def test_create_team_same_members(client, headers, sdb, disable_default_user):
    body = TeamCreateRequest(1, "Engineering 1",
                             ["github.com/se7entyse7en",
                              "github.com/vmarkovtsev"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )
    assert response.status == 200

    body = TeamCreateRequest(1, "Engineering 2",
                             ["github.com/vmarkovtsev",
                              "github.com/se7entyse7en"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )

    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body

    teams = await sdb.fetch_all(select([Team]))
    assert len(teams) == 2
    assert teams[0][Team.members.name] == teams[1][Team.members.name]
    assert {t[Team.name.name] for t in teams} == {"Engineering 1", "Engineering 2"}


async def test_create_team_same_name(client, headers, sdb, disable_default_user):
    body = TeamCreateRequest(1, "Engineering", ["github.com/se7entyse7en"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )
    assert response.status == 200

    body = TeamCreateRequest(1, "Engineering", ["github.com/vmarkovtsev"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/team/create", headers=headers, json=body,
    )

    body = (await response.read()).decode("utf-8")
    assert response.status == 409, "Response body is : " + body
    parsed = json.loads(body)
    detail = parsed["detail"]
    del parsed["detail"]
    assert "Team 'Engineering' already exists" in detail
    assert parsed == {
        "type": "/errors/DatabaseConflict",
        "title": "Conflict",
        "status": 409,
    }

    teams = await sdb.fetch_all(select([Team]))
    assert len(teams) == 1
    _test_same_team(teams[0], {
        "id": 1,
        "members": [51],
        "name": "Engineering",
        "owner_id": 1,
        "parent_id": None,
    })


@pytest.mark.parametrize(
    "initial_teams",
    [
        [],
        [
            {"owner_id": 1, "name": "Team 1", "members": [51]},
            {"owner_id": 1, "name": "Team 2", "members": [40020],
             "parent_id": 1},
        ],
    ],
    ids=["empty", "non-empty"],
)
@pytest.mark.parametrize("account", [1, 2], ids=["as admin", "as non-admin"])
async def test_list_teams_smoke(client, headers, initial_teams, sdb, account, vadim_id_mapping):
    await sdb.execute(insert(AccountGitHubAccount).values({
        AccountGitHubAccount.account_id: 2,
        AccountGitHubAccount.id: 1,
    }))
    contributors_details = {
        # No further details because didn't contribute to repos
        "github.com/se7entyse7en": {
            "login": "github.com/se7entyse7en",
            "email": "loumarvincaraig@gmail.com",
            "name": "Lou Marvin Caraig",
            "picture": "https://avatars.githubusercontent.com/u/5599208?s=600&u=46e13fb429c44109e0b125f133c4e694a6d2646e&v=4",  # noqa
        },
        "github.com/vmarkovtsev": {
            "login": "github.com/vmarkovtsev",
            "email": "gmarkhor@gmail.com",
            "name": "Vadim Markovtsev",
            "picture": "https://avatars1.githubusercontent.com/u/2793551?s=600&v=4",
            "jira_user": "Vadim Markovtsev",
        },
    }
    for t in initial_teams:
        await sdb.execute(insert(Team).values(Team(**t).create_defaults().explode()))

    response = await client.request(method="GET", path=f"/v1/teams/{account}", headers=headers)
    assert response.status == 200
    body = (await response.read()).decode("utf-8")
    teams = sorted([TeamListItem.from_dict(t) for t in json.loads(body)],
                   key=attrgetter("id"))

    for i, (actual, expected) in enumerate(zip(teams, initial_teams), 1):
        assert actual.id == i
        assert actual.name == expected["name"]
        for m in actual.members:
            assert m.to_dict() == contributors_details[m.login]


@pytest.mark.parametrize(
    "initial_teams",
    [
        [],
        [
            {"owner_id": 1, "name": "Team 1", "members": ["github.com/se7entyse7en"]},
            {"owner_id": 1, "name": "Team 2", "members": ["github.com/vmarkovtsev"]},
        ],
    ],
    ids=["empty", "non-empty"],
)
@pytest.mark.parametrize("account", [3, 4], ids=["not a member", "invalid account"])
async def test_list_teams_wrong_account(client, headers, sdb, account, initial_teams):
    for t in initial_teams:
        await sdb.execute(insert(Team).values(Team(**t).create_defaults().explode()))

    response = await client.request(method="GET", path=f"/v1/teams/{account}", headers=headers)

    body = (await response.read()).decode("utf-8")
    assert response.status == 404, "Response body is : " + body
    parsed = json.loads((await response.read()).decode("utf-8"))
    assert parsed == {
        "type": "/errors/AccountNotFound",
        "title": "Not Found",
        "status": 404,
        "detail": (f"Account {account} does not exist or user auth0|5e1f6dfb57bc640ea390557b "
                   "is not a member."),
    }


def _test_same_team(actual, expected, no_timings=True):
    if not isinstance(actual, dict):
        actual = dict(actual)

    if no_timings:
        actual.pop("created_at", None)
        actual.pop("updated_at", None)
        expected.pop("created_at", None)
        expected.pop("updated_at", None)

    assert actual == expected


async def test_resync_teams_smoke(client, headers, sdb, disable_default_user):
    response = await client.request(method="DELETE", path="/v1/teams/1", headers=headers)

    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    teams = {t["name"]: TeamListItem.from_dict(t) for t in json.loads(body)}
    actual_teams = await sdb.fetch_all(select([Team]).where(Team.owner_id == 1))
    assert len(teams) == len(actual_teams)

    assert teams.keys() == {
        "team", "engineering", "business", "operations", "product", "admin", "automation",
    }
    assert [m.login for m in teams["product"].members] == ["github.com/warenlg", "github.com/eiso"]


async def test_resync_teams_default_user(client, headers):
    response = await client.request(method="DELETE", path="/v1/teams/1", headers=headers)

    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


async def test_resync_teams_wrong_user(client, headers, disable_default_user):
    response = await client.request(method="DELETE", path="/v1/teams/3", headers=headers)

    body = (await response.read()).decode("utf-8")
    assert response.status == 404, "Response body is : " + body


async def test_resync_teams_regular_user(client, headers, disable_default_user):
    response = await client.request(method="DELETE", path="/v1/teams/2", headers=headers)

    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


class TestUpdateTeam:
    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_smoke(self, client, sdb, disable_default_user):
        for model in (
            TeamFactory(id=10, name="Parent"),
            TeamFactory(id=11, name="Test", members=[40020], parent_id=None),
        ):
            await sdb.execute(model_insert_stmt(model))
        body = TeamUpdateRequest("Dream", ["github.com/warenlg"], 10).to_dict()

        await self._request(client, 11, body, 200)
        team = await sdb.fetch_one(select([Team]).where(Team.id == 11))
        assert team[Team.name.name] == "Dream"
        assert team[Team.members.name] == [29]
        assert team[Team.parent_id.name] == 10

    async def test_default_user(self, client, sdb):
        await sdb.execute(insert(Team).values(Team(
            id=1, owner_id=1, name="Test", members=[40020],
        ).create_defaults().explode()))
        body = TeamUpdateRequest("Engineering", ["github.com/se7entyse7en"], None).to_dict()
        await self._request(client, 1, body, 403)

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    @pytest.mark.parametrize("owner, id, name, members, parent, status", [
        (1, 1, "Engineering", [], None, 400),
        (1, 1, "", ["github.com/se7entyse7en"], None, 400),
        (1, 1, "$" * 256, ["github.com/se7entyse7en"], None, 400),
        (1, 3, "Engineering", ["github.com/se7entyse7en"], None, 404),
        (2, 1, "Engineering", ["github.com/se7entyse7en"], None, 200),
        (3, 1, "Engineering", ["github.com/se7entyse7en"], None, 404),
        (1, 1, "Dream", ["github.com/se7entyse7en"], None, 409),
        (1, 1, "Engineering", ["github.com/eiso"], None, 200),
        (2, 1, "Dream", ["github.com/se7entyse7en"], None, 200),
        (2, 1, "Engineering", ["github.com/eiso"], None, 200),
        (2, 1, "Engineering", ["github.com/eiso"], 2, 400),
        (2, 1, "Engineering", ["github.com/eiso"], 1, 400),
    ])
    async def test_nasty_input(
        self, client, sdb, disable_default_user, owner, id, name, members, parent, status,
    ):
        await sdb.execute(update(AccountGitHubAccount)
                          .where(AccountGitHubAccount.id == 6366825)
                          .values({AccountGitHubAccount.account_id: owner}))
        for model in (
            TeamFactory(owner_id=owner, id=1, name="Engineering", members=[51]),
            TeamFactory(owner_id=1, id=2, name="Dream", members=[39936]),
        ):
            await sdb.execute(model_insert_stmt(model))

        body = TeamUpdateRequest(name, members, parent).to_dict()
        await self._request(client, id, body, status)

    async def test_parent_cycle(self, client, sdb, disable_default_user):
        for model in (
            TeamFactory(owner_id=1, id=1, name="Engineering", members=[51]),
            TeamFactory(owner_id=1, id=2, parent_id=1, name="Dream", members=[39936]),
        ):
            await sdb.execute(model_insert_stmt(model))
        body = TeamUpdateRequest("Engineering", ["github.com/se7entyse7en"], 2).to_dict()
        rbody = await self._request(client, 1, body, 400)
        assert "cycle" in rbody

    async def _request(
        self, client: TestClient, team_id: int, json: dict, assert_status: int,
    ) -> str:
        from tests.conftest import DEFAULT_HEADERS
        response = await client.request(
            method="PUT", path=f"/v1/team/{team_id}", headers=DEFAULT_HEADERS, json=json,
        )
        body = (await response.read()).decode("utf-8")
        assert response.status == assert_status, f"Response body is: {body}"
        return body


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_delete_team_smoke(client, headers, sdb, disable_default_user):
    await sdb.execute(insert(Team).values(Team(
        owner_id=1,
        name="Engineering",
        members=[51],
    ).create_defaults().explode()))
    await sdb.execute(insert(Team).values(Team(
        owner_id=1,
        name="Test",
        members=[40020],
        parent_id=1,
    ).create_defaults().explode()))
    response = await client.request(
        method="DELETE", path="/v1/team/1", headers=headers, json={},
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    teams = await sdb.fetch_all(select([Team]))
    assert len(teams) == 1
    assert teams[0][Team.name.name] == "Test"
    assert teams[0][Team.parent_id.name] is None


async def test_delete_team_default_user(client, headers, sdb):
    await sdb.execute(insert(Team).values(Team(
        owner_id=1,
        name="Engineering",
        members=[51],
    ).create_defaults().explode()))
    await sdb.execute(insert(Team).values(Team(
        owner_id=1,
        name="Test",
        members=[40020],
        parent_id=1,
    ).create_defaults().explode()))
    response = await client.request(
        method="DELETE", path="/v1/team/1", headers=headers, json={},
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize("owner, id, status", [
    (1, 2, 404),
    (2, 1, 200),
    (3, 1, 404),
])
async def test_delete_team_nasty_input(client, headers, sdb, disable_default_user,
                                       owner, id, status):
    await sdb.execute(insert(Team).values(Team(
        owner_id=owner,
        name="Engineering",
        members=[51],
    ).create_defaults().explode()))
    response = await client.request(
        method="DELETE", path="/v1/team/%d" % id, headers=headers, json={},
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == status, "Response body is : " + body


async def test_get_team_smoke(client, headers, sdb, vadim_id_mapping):
    await sdb.execute(insert(Team).values(Team(
        owner_id=1,
        name="Engineering",
        members=[40020, 39789],
    ).create_defaults().explode()))
    response = await client.request(
        method="GET", path="/v1/team/1", headers=headers, json={},
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = json.loads(body)
    assert body == {"id": 1, "name": "Engineering", "parent": None, "members": [
        {"login": "github.com/mcuadros",
         "name": "Máximo Cuadros",
         "email": "mcuadros@gmail.com",
         "picture": "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4"},
        {"login": "github.com/vmarkovtsev",
         "email": "gmarkhor@gmail.com",
         "name": "Vadim Markovtsev",
         "picture": "https://avatars1.githubusercontent.com/u/2793551?s=600&v=4",
         "jira_user": "Vadim Markovtsev",
         }]}


@pytest.mark.parametrize("owner, id, status", [
    (1, 2, 404),
    (2, 1, 422),
    (3, 1, 404),
])
async def test_get_team_nasty_input(client, headers, sdb, owner, id, status):
    await sdb.execute(insert(Team).values(Team(
        owner_id=owner,
        name="Engineering",
        members=[51, 39789],
    ).create_defaults().explode()))
    response = await client.request(
        method="GET", path="/v1/team/%d" % id, headers=headers, json={},
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == status, "Response body is : " + body


class TestGetRootTeam:
    async def test_existing(self, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(id=99, name="ROOT")))
        root_team = await get_root_team(1, sdb)
        assert root_team["id"] == 99
        assert root_team["name"] == "ROOT"

    async def test_not_existing(self, sdb: Database) -> None:
        with pytest.raises(TeamNotFoundError):
            await get_root_team(1, sdb)

    async def test_multiple_root_teams(self, sdb: Database) -> None:
        for model in (TeamFactory(), TeamFactory()):
            await sdb.execute(model_insert_stmt(model))
        with pytest.raises(MultipleRootTeamsError):
            await get_root_team(1, sdb)


class TestGetTeamFromDB:
    async def test_found(self, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(id=99, name="TEAM 99")))
        team = await get_team_from_db(1, 99, sdb)
        assert team["name"] == "TEAM 99"

    async def test_not_found(self, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(id=99, owner_id=2)))
        with pytest.raises(TeamNotFoundError):
            await get_team_from_db(1, 99, sdb)
