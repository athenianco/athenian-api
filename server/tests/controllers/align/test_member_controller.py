from typing import Optional

from athenian.api.db import Database
from tests.testutils.auth import force_request_auth
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.state import (
    MappedJIRAIdentityFactory,
    TeamFactory,
    UserAccountFactory,
)
from tests.testutils.requester import Requester


class BaseListTeamMembersTest(Requester):
    async def _request(
        self,
        team_id: int,
        assert_status: int = 200,
        recursive: Optional[bool] = None,
        account: Optional[int] = None,
        headers: Optional[dict] = None,
        user_id: Optional[str] = None,
    ) -> list[dict] | dict:
        path = f"/private/align/members/{team_id}"
        params = {}
        if recursive is not None:
            params["recursive"] = str(recursive).lower()
        if account is not None:
            params["account"] = str(account)
        headers = self.headers if headers is None else (self.headers | headers)

        with force_request_auth(user_id, headers) as headers:
            response = await self.client.request(
                method="GET", path=path, headers=headers, params=params,
            )

        assert response.status == assert_status
        return await response.json()


class TestListTeamMembersErrors(BaseListTeamMembersTest):
    async def test_invalid_auth_token(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[1, 2]))
        headers = {"Authorization": "Bearer invalid"}
        res = await self._request(1, 401, headers=headers)
        assert isinstance(res, dict)
        assert res["type"] == "/errors/Unauthorized"

    async def test_team_not_found(self) -> None:
        res = await self._request(999, 404)
        assert isinstance(res, dict)
        assert res["detail"] == "Team 999 not found or access denied"

    async def test_team_not_found_recursive(self) -> None:
        res = await self._request(999, 404, recursive=True)
        assert isinstance(res, dict)
        assert res["detail"] == "Team 999 not found or access denied"

    async def test_team_account_mismatch(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10, owner_id=3))
        res = await self._request(10, 404)
        assert isinstance(res, dict)
        assert res["detail"] == "Team 10 not found or access denied"

    async def test_team_correct_account_param_mismatch(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10))
        res = await self._request(10, 404, account=3)
        assert isinstance(res, dict)
        assert res["type"] == "/errors/AccountNotFound"

    async def test_team_account_mismatch_non_default_user(self, sdb: Database) -> None:
        await models_insert(
            sdb, UserAccountFactory(user_id="my-user", account_id=2), TeamFactory(id=10),
        )
        await self._request(10, 404, user_id="my-user")

    async def test_root_team_multiple_roots(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(), TeamFactory())
        res = await self._request(0, 500, account=1)
        assert isinstance(res, dict)
        assert res["title"] == "Multiple root teams"
        assert res["detail"] == "Account 1 has multiple root teams"

    async def test_root_team_no_team_exists(self, sdb: Database) -> None:
        res = await self._request(0, 404, account=1)
        assert isinstance(res, dict)
        assert res["type"] == "/errors/teams/TeamNotFound"
        assert res["detail"] == "Root team not found or access denied"

    async def test_root_team_missing_account(self, sdb: Database) -> None:
        res = await self._request(0, 400, account=None)
        assert isinstance(res, dict)
        assert res["type"] == "/errors/BadRequest"

    async def test_root_team_account_mismatch(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            UserAccountFactory(user_id="u0", account_id=2),
            TeamFactory(id=10, members=[40078]),
        )
        res = await self._request(0, 404, account=1, user_id="u0")
        assert isinstance(res, dict)
        assert res["type"] == "/errors/AccountNotFound"


class TestListTeamMembers(BaseListTeamMembersTest):
    async def test_explicit_non_root_team(self, sdb: Database) -> None:
        # members id are from the 6 MB metadata db fixture
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[]),
            TeamFactory(id=2, members=[40078, 39652900], parent_id=1),
        )

        res = await self._request(2)
        assert [m["login"] for m in res] == ["github.com/leantrace", "github.com/reujab"]
        assert [m["name"] for m in res] == ["Alexander Schamne", "Christopher Knight"]

        assert "email" in res[0]
        assert "picture" in res[0]

    async def test_explicit_root_team(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40078], parent_id=None))
        res = await self._request(1)
        assert [m["login"] for m in res] == ["github.com/reujab"]

    async def test_implicit_root_team_empty(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[]),
            TeamFactory(id=20, parent_id=10, members=[3]),
            TeamFactory(id=30, parent_id=10, members=[3, 4]),
        )
        res = await self._request(0, account=1)
        assert res == []

    async def test_implicit_root_team(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[40078]),
            TeamFactory(id=11, parent_id=10, members=[39652900]),
        )
        res = await self._request(0, account=1)
        assert [m["login"] for m in res] == ["github.com/reujab"]

    async def test_recursive(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[39652900]),
            TeamFactory(id=2, parent_id=1, members=[40078]),
        )

        res = await self._request(1, recursive=True)
        assert [m["login"] for m in res] == ["github.com/leantrace", "github.com/reujab"]

        res = await self._request(1, recursive=False)
        assert [m["login"] for m in res] == ["github.com/leantrace"]

        res = await self._request(1)
        assert [m["login"] for m in res] == ["github.com/leantrace"]

    async def test_result_ordering(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[100, 101, 102, 103]),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.UserFactory(node_id=100, html_url="https://c", name="Foo Bar"),
                md_factory.UserFactory(node_id=101, html_url="https://a", name=None),
                md_factory.UserFactory(node_id=102, html_url="https://b", name="zz-Top"),
                md_factory.UserFactory(node_id=103, html_url="https://d", name="aa"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            res = await self._request(1, recursive=True)

        assert [m["login"] for m in res] == ["d", "c", "b", "a"]
        assert [m.get("name") for m in res] == ["aa", "Foo Bar", "zz-Top", None]

    async def test_jira_user_field(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[100, 101]),
            MappedJIRAIdentityFactory(github_user_id=100, jira_user_id="200"),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.UserFactory(node_id=100),
                md_factory.UserFactory(node_id=101),
                md_factory.JIRAUserFactory(id="200", display_name="My JIRA name"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)
            res = await self._request(1)

        assert [m.get("jira_user") for m in res] == ["My JIRA name", None]
