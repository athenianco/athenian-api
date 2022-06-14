from datetime import datetime, timezone
import json
from operator import attrgetter

from aiohttp import ClientResponse
from aiohttp.test_utils import TestClient
from freezegun import freeze_time
import pytest
from sqlalchemy import insert, select, update

from athenian.api.db import Database
from athenian.api.models.state.models import AccountGitHubAccount, Team
from athenian.api.models.web import TeamUpdateRequest
from athenian.api.models.web.team import Team as TeamListItem
from athenian.api.models.web.team_create_request import TeamCreateRequest
from tests.conftest import DEFAULT_HEADERS
from tests.testutils.db import (
    assert_existing_row,
    db_datetime_equals,
    model_insert_stmt,
    models_insert,
)
from tests.testutils.factory.state import TeamFactory


class TestCreateTeam:
    @pytest.mark.parametrize("account", [1, 2], ids=["as admin", "as non-admin"])
    async def test_smoke(self, client, headers, sdb, account, disable_default_user):
        await sdb.execute(
            update(AccountGitHubAccount)
            .where(AccountGitHubAccount.id == 6366825)
            .values({AccountGitHubAccount.account_id: account}),
        )
        root_team_id = await sdb.execute(
            model_insert_stmt(
                TeamFactory(owner_id=account, parent_id=None), with_primary_keys=False,
            ),
        )

        body = TeamCreateRequest(account, "Engineering", ["github.com/se7entyse7en"], 1).to_dict()
        response = await self._request(client, body, 200)
        assert len(await sdb.fetch_all(select(Team))) == 2

        eng_team_id = (await response.json())["id"]

        team = await sdb.fetch_one(select(Team).where(Team.id == eng_team_id))
        _test_same_team(
            team,
            {
                "id": eng_team_id,
                "members": [51],
                "name": "Engineering",
                "owner_id": account,
                "parent_id": root_team_id,
            },
        )

        body["name"] = "Management"
        body["members"][0] = "github.com/vmarkovtsev"
        body["parent"] = eng_team_id
        response = await self._request(client, body, 200)
        mngmt_team_id = (await response.json())["id"]
        team = await sdb.fetch_one(select(Team).where(Team.id == mngmt_team_id))
        _test_same_team(
            team,
            {
                "id": mngmt_team_id,
                "members": [40020],
                "name": "Management",
                "owner_id": account,
                "parent_id": eng_team_id,
            },
        )

    async def test_bot(self, client, headers, sdb, disable_default_user):
        await sdb.execute(
            update(AccountGitHubAccount)
            .where(AccountGitHubAccount.id == 6366825)
            .values({AccountGitHubAccount.account_id: 1}),
        )
        await sdb.execute(model_insert_stmt(TeamFactory(id=100, parent_id=None)))
        body = TeamCreateRequest(1, "Engineering", ["github.com/apps/dependabot"], 100).to_dict()
        response = await self._request(client, body, 200)
        assert len(await sdb.fetch_all(select(Team))) == 2
        eng_team_id = (await response.json())["id"]
        team = await sdb.fetch_one(select([Team]).where(Team.id == eng_team_id))
        _test_same_team(
            team,
            {
                "id": eng_team_id,
                "members": [17019778],
                "name": "Engineering",
                "owner_id": 1,
                "parent_id": 100,
            },
        )

    @pytest.mark.parametrize("account", [3, 4], ids=["not a member", "invalid account"])
    async def test_wrong_account(self, client, headers, sdb, account, disable_default_user):
        await sdb.execute(model_insert_stmt(TeamFactory(id=100, owner_id=3, parent_id=None)))
        body = TeamCreateRequest(account, "Engin", ["github.com/se7entyse7en"], 100).to_dict()
        response = await self._request(client, body, 404)
        parsed = await response.json()
        assert parsed == {
            "type": "/errors/AccountNotFound",
            "title": "Not Found",
            "status": 404,
            "detail": (
                f"Account {account} does not exist or user auth0|62a1ae88b6bba16c6dbc6870 "
                "is not a member."
            ),
        }

        assert len(await sdb.fetch_all(select(Team))) == 1

    async def test_default_user(self, client, headers, sdb):
        await sdb.execute(model_insert_stmt(TeamFactory(id=100, parent_id=None)))
        body = TeamCreateRequest(1, "Engineering", ["github.com/se7entyse7en"], 100).to_dict()
        await self._request(client, body, 403)
        assert len(await sdb.fetch_all(select(Team))) == 1

    async def test_wrong_member(self, client, headers, sdb, disable_default_user):
        await sdb.execute(model_insert_stmt(TeamFactory(id=100, parent_id=None)))
        body = TeamCreateRequest(
            1,
            "Engineering",
            ["github.com/se7entyse7en/foo", "github.com/vmarkovtsev/bar", "github.com/warenlg"],
            100,
        ).to_dict()
        response = await self._request(client, body, 400)
        parsed = await response.json()
        assert parsed == {
            "type": "/errors/BadRequest",
            "title": "Bad Request",
            "status": 400,
            "detail": (
                "Invalid members of the team: "
                "github.com/se7entyse7en/foo, github.com/vmarkovtsev/bar"
            ),
        }

        assert len(await sdb.fetch_all(select(Team))) == 1

    async def test_wrong_parent(self, client, headers, sdb, disable_default_user):
        body = TeamCreateRequest(
            1, "Engineering", ["github.com/se7entyse7en", "github.com/warenlg"], 1,
        ).to_dict()
        await self._request(client, body, 400)
        await sdb.execute(model_insert_stmt(TeamFactory(id=1, name="Test", owner_id=3)))
        await self._request(client, body, 400)

    async def test_same_members(self, client, headers, sdb, disable_default_user):
        await sdb.execute(model_insert_stmt(TeamFactory(id=100, name="Root", parent_id=None)))
        body = TeamCreateRequest(
            1, "Engineering 1", ["github.com/se7entyse7en", "github.com/vmarkovtsev"], 100,
        ).to_dict()
        await self._request(client, body, 200)

        body = TeamCreateRequest(
            1, "Engineering 2", ["github.com/vmarkovtsev", "github.com/se7entyse7en"], 100,
        ).to_dict()
        await self._request(client, body, 200)
        teams = await sdb.fetch_all(select(Team).order_by(Team.name))
        assert len(teams) == 3
        assert teams[0][Team.members.name] == teams[1][Team.members.name]
        assert [t[Team.name.name] for t in teams] == ["Engineering 1", "Engineering 2", "Root"]

    async def test_same_name(self, client, headers, sdb, disable_default_user):
        await sdb.execute(model_insert_stmt(TeamFactory(id=10, parent_id=None, name="Root")))
        body = TeamCreateRequest(1, "Engineering", ["github.com/se7entyse7en"], 10).to_dict()
        response = await self._request(client, body, 200)
        eng_team_id = (await response.json())["id"]

        body = TeamCreateRequest(1, "Engineering", ["github.com/vmarkovtsev"], 10).to_dict()
        response = await self._request(client, body, 409)
        parsed = await response.json()
        detail = parsed["detail"]
        del parsed["detail"]
        assert "Team 'Engineering' already exists" in detail
        assert parsed == {
            "type": "/errors/DatabaseConflict",
            "title": "Conflict",
            "status": 409,
        }

        teams = await sdb.fetch_all(select(Team).order_by(Team.name))
        assert len(teams) == 2
        _test_same_team(
            teams[0],
            {
                "id": eng_team_id,
                "members": [51],
                "name": "Engineering",
                "owner_id": 1,
                "parent_id": 10,
            },
        )

    async def test_no_parent(self, client, headers, sdb, disable_default_user):
        await sdb.execute(model_insert_stmt(TeamFactory(id=10)))
        body = TeamCreateRequest(1, "Engineering", ["github.com/se7entyse7en"], None).to_dict()
        await self._request(client, body, 200)
        t = await assert_existing_row(sdb, Team, name="Engineering")
        assert t[Team.parent_id.name] == 10

    async def _request(self, client: TestClient, json: dict, assert_status: int) -> ClientResponse:
        response = await client.request(
            method="POST", path="/v1/team/create", headers=DEFAULT_HEADERS, json=json,
        )
        assert response.status == assert_status
        return response


class TestListTeams:
    @pytest.mark.parametrize(
        "initial_teams",
        [
            [],
            [
                {"id": 1, "owner_id": 1, "name": "Root", "parent_id": None},
                {"id": 2, "owner_id": 1, "name": "Team 1", "members": [51], "parent_id": 1},
                {"id": 3, "owner_id": 1, "name": "Team 2", "members": [40020], "parent_id": 1},
            ],
        ],
        ids=["empty", "non-empty"],
    )
    @pytest.mark.parametrize("account", [1, 2])
    async def test_smoke(self, client, headers, initial_teams, sdb, account, vadim_id_mapping):
        await sdb.execute(
            insert(AccountGitHubAccount).values(
                {
                    AccountGitHubAccount.account_id: 2,
                    AccountGitHubAccount.id: 1,
                },
            ),
        )
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
            await sdb.execute(model_insert_stmt(TeamFactory(**t)))

        response = await client.request(method="GET", path=f"/v1/teams/{account}", headers=headers)
        assert response.status == 200
        body = (await response.read()).decode("utf-8")
        teams = sorted([TeamListItem.from_dict(t) for t in json.loads(body)], key=attrgetter("id"))

        if account == 2:
            expected_teams = []
        else:
            expected_teams = [t for t in initial_teams]

        assert len(teams) == len(expected_teams)

        for id_, (actual, expected) in enumerate(zip(teams, expected_teams), 1):
            assert actual.id == id_
            assert actual.parent == expected["parent_id"]
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
    async def test_wrong_account(self, client, headers, sdb, account, initial_teams):
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
            "detail": (
                f"Account {account} does not exist or user auth0|62a1ae88b6bba16c6dbc6870 "
                "is not a member."
            ),
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
    await sdb.execute(model_insert_stmt(TeamFactory(parent_id=None)))
    response = await client.request(method="DELETE", path="/v1/teams/1", headers=headers)
    body = await response.json()
    assert response.status == 200, "Response body is : " + body
    teams = {t["name"]: TeamListItem.from_dict(t) for t in body}
    actual_teams = await sdb.fetch_all(select([Team]).where(Team.owner_id == 1))
    assert len(teams) == len(actual_teams) - 1  # root team is not included in the response

    assert teams.keys() == {
        "team",
        "engineering",
        "business",
        "operations",
        "product",
        "admin",
        "automation",
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
    @freeze_time("2022-04-01")
    async def test_smoke(self, client, sdb, disable_default_user):
        created_at = datetime(2001, 12, 3, 3, 20, tzinfo=timezone.utc)
        for model in (
            TeamFactory(id=10, name="Parent"),
            TeamFactory(id=11, name="Test", members=[40020], parent_id=10, created_at=created_at),
        ):
            await sdb.execute(model_insert_stmt(model))
        body = TeamUpdateRequest("Dream", ["github.com/warenlg"], 10).to_dict()

        await self._request(client, 11, body, 200)
        team = await sdb.fetch_one(select([Team]).where(Team.id == 11))
        assert team[Team.name.name] == "Dream"
        assert team[Team.members.name] == [29]
        assert team[Team.parent_id.name] == 10
        assert db_datetime_equals(sdb, team[Team.created_at.name], created_at)
        assert db_datetime_equals(
            sdb, team[Team.updated_at.name], datetime(2022, 4, 1, tzinfo=timezone.utc),
        )

    async def test_default_user(self, client, sdb):
        await sdb.execute(
            insert(Team).values(
                Team(id=1, owner_id=1, name="Test", members=[40020]).create_defaults().explode(),
            ),
        )
        body = TeamUpdateRequest("Engineering", ["github.com/se7entyse7en"], None).to_dict()
        await self._request(client, 1, body, 403)

    @pytest.mark.parametrize(
        "owner, id, name, members, parent, status",
        [
            (1, 2, "Engineering", [], 1, 400),
            (1, 2, "", ["github.com/se7entyse7en"], 1, 400),
            (1, 2, "$" * 256, ["github.com/se7entyse7en"], 1, 400),
            (1, 4, "Engineering", ["github.com/se7entyse7en"], 1, 404),
            (2, 2, "Engineering", ["github.com/se7entyse7en"], 1, 200),
            (3, 2, "Engineering", ["github.com/se7entyse7en"], 1, 404),
            (1, 2, "Dream", ["github.com/se7entyse7en"], 1, 409),
            (1, 2, "Engineering", ["github.com/eiso"], 1, 200),
            (2, 2, "Dream", ["github.com/se7entyse7en"], 1, 200),
            (2, 2, "Engineering", ["github.com/eiso"], 1, 200),
            (2, 2, "Engineering", ["github.com/eiso"], 2, 400),
            (2, 1, "Root", ["github.com/eiso"], 2, 400),
        ],
    )
    async def test_nasty_input(
        self,
        client,
        sdb,
        disable_default_user,
        owner,
        id,
        name,
        members,
        parent,
        status,
    ):
        await sdb.execute(
            update(AccountGitHubAccount)
            .where(AccountGitHubAccount.id == 6366825)
            .values({AccountGitHubAccount.account_id: owner}),
        )
        for model in (
            TeamFactory(owner_id=owner, id=1, name="Root", members=[]),
            TeamFactory(owner_id=owner, id=2, name="Engineering", members=[51], parent_id=1),
            TeamFactory(owner_id=1, id=3, name="Dream", members=[39936], parent_id=1),
        ):
            await sdb.execute(model_insert_stmt(model))

        body = TeamUpdateRequest(name, members, parent).to_dict()
        await self._request(client, id, body, status)

    async def test_parent_cycle(self, client, sdb, disable_default_user):
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[51]),
            TeamFactory(id=2, parent_id=1),
            TeamFactory(id=3, parent_id=2),
        )
        body = TeamUpdateRequest("Engineering", ["github.com/se7entyse7en"], 3).to_dict()
        rbody = await self._request(client, 2, body, 400)
        assert "cycle" in rbody

    async def test_null_parent_as_root_parent(self, client, sdb, disable_default_user):
        await models_insert(
            sdb,
            TeamFactory(id=1, parent_id=None),
            TeamFactory(id=2, parent_id=1),
            TeamFactory(id=3, parent_id=2),
        )
        body = TeamUpdateRequest("Engineering", ["github.com/se7entyse7en"], None).to_dict()
        await self._request(client, 3, body)
        await assert_existing_row(sdb, Team, id=3, parent_id=1)

    async def test_parent_stays_null(self, client, sdb, disable_default_user):
        await sdb.execute(model_insert_stmt(TeamFactory(id=1)))
        body = TeamUpdateRequest("Engineering", ["github.com/se7entyse7en"], None).to_dict()
        await self._request(client, 1, body, 200)
        team = await sdb.fetch_one(select([Team]).where(Team.id == 1))
        assert team[Team.name.name] == "Engineering"

    async def test_set_root_parent_forbidden(self, client, sdb, disable_default_user):
        await sdb.execute(model_insert_stmt(TeamFactory(id=1)))
        await sdb.execute(model_insert_stmt(TeamFactory(id=2)))

        body = TeamUpdateRequest("Engineering", [], 2).to_dict()
        res = await self._request(client, 1, body, 400)
        res_data = json.loads(res)
        assert res_data["detail"] == "Cannot set parent for root team."

    async def test_change_parent(self, client, sdb, disable_default_user):
        for model in (
            TeamFactory(id=1, name="Root"),
            TeamFactory(id=2, parent_id=1, members=[39936]),
            TeamFactory(id=3, parent_id=1, members=[39936]),
        ):
            await sdb.execute(model_insert_stmt(model))

        body = TeamUpdateRequest("New Name", ["github.com/se7entyse7en"], 2).to_dict()
        await self._request(client, 3, body, 200)
        await assert_existing_row(sdb, Team, name="New Name", parent_id=2, id=3)

    async def _request(
        self,
        client: TestClient,
        team_id: int,
        json: dict,
        assert_status: int = 200,
    ) -> str:
        response = await client.request(
            method="PUT", path=f"/v1/team/{team_id}", headers=DEFAULT_HEADERS, json=json,
        )
        body = (await response.read()).decode("utf-8")
        assert response.status == assert_status, f"Response body is: {body}"
        return body


class TestDeleteTeam:
    async def test_smoke(self, client, headers, sdb, disable_default_user):
        for model in (TeamFactory(id=1, name="Root"), TeamFactory(id=2, parent_id=1, name="Test")):
            await sdb.execute(model_insert_stmt(model))

        await self._request(client, 2, 200)
        teams = await sdb.fetch_all(select(Team))
        assert len(teams) == 1
        assert teams[0][Team.name.name] == "Root"
        assert teams[0][Team.parent_id.name] is None

    async def test_default_user(self, client, headers, sdb):
        for model in (TeamFactory(id=1), TeamFactory(id=2, parent_id=1)):
            await sdb.execute(model_insert_stmt(model))

        await self._request(client, 2, 403)

    @pytest.mark.parametrize(
        "owner, id, status",
        [
            (1, 2, 404),
            (2, 1, 200),
            (3, 1, 404),
        ],
    )
    async def test_nasty_input(
        self,
        client,
        headers,
        sdb,
        disable_default_user,
        owner,
        id,
        status,
    ):
        await sdb.execute(model_insert_stmt(TeamFactory(id=9, parent_id=None, owner_id=owner)))
        await sdb.execute(model_insert_stmt(TeamFactory(id=1, parent_id=9, owner_id=owner)))
        await self._request(client, id, status)

    async def test_team_forbidden(
        self,
        client: TestClient,
        headers: dict,
        sdb: Database,
        disable_default_user: None,
    ) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(id=1, parent_id=None)))
        response = await self._request(client, 1, 400)
        assert response["detail"] == "Root team cannot be deleted."
        await assert_existing_row(sdb, Team, id=1)

    async def test_children_parent_is_updated(
        self,
        client: TestClient,
        headers: dict,
        sdb: Database,
        disable_default_user: None,
    ) -> None:
        for model in (
            TeamFactory(id=1, parent_id=None),
            TeamFactory(id=2, parent_id=1),
            TeamFactory(id=3, parent_id=2),
            TeamFactory(id=4, parent_id=2),
            TeamFactory(id=5, parent_id=1),
            TeamFactory(id=6, parent_id=5),
        ):
            await sdb.execute(model_insert_stmt(model))

        await self._request(client, 2, 200)
        rows = await sdb.fetch_all(select(Team.id, Team.parent_id).order_by(Team.id))

        assert rows[0] == (1, None)
        assert rows[1] == (3, 1)
        assert rows[2] == (4, 1)
        assert rows[3] == (5, 1)
        assert rows[4] == (6, 5)

    async def _request(self, client: TestClient, team_id: int, assert_status: int) -> dict:
        response = await client.request(
            method="DELETE", path=f"/v1/team/{team_id}", headers=DEFAULT_HEADERS, json={},
        )
        assert response.status == assert_status
        return await response.json()


async def test_get_team_smoke(client, headers, sdb, vadim_id_mapping):
    await sdb.execute(
        insert(Team).values(
            Team(
                owner_id=1,
                name="Engineering",
                members=[40020, 39789],
            )
            .create_defaults()
            .explode(),
        ),
    )
    response = await client.request(method="GET", path="/v1/team/1", headers=headers, json={})
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = json.loads(body)
    assert body == {
        "id": 1,
        "name": "Engineering",
        "parent": None,
        "members": [
            {
                "login": "github.com/mcuadros",
                "name": "Máximo Cuadros",
                "email": "mcuadros@gmail.com",
                "picture": "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4",
            },
            {
                "login": "github.com/vmarkovtsev",
                "email": "gmarkhor@gmail.com",
                "name": "Vadim Markovtsev",
                "picture": "https://avatars1.githubusercontent.com/u/2793551?s=600&v=4",
                "jira_user": "Vadim Markovtsev",
            },
        ],
    }


@pytest.mark.parametrize(
    "owner, id, status",
    [
        (1, 2, 404),
        (2, 1, 422),
        (3, 1, 404),
    ],
)
async def test_get_team_nasty_input(client, headers, sdb, owner, id, status):
    await sdb.execute(
        insert(Team).values(
            Team(
                owner_id=owner,
                name="Engineering",
                members=[51, 39789],
            )
            .create_defaults()
            .explode(),
        ),
    )
    response = await client.request(
        method="GET", path="/v1/team/%d" % id, headers=headers, json={},
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == status, "Response body is : " + body
