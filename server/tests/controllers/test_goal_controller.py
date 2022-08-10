from aiohttp.test_utils import TestClient

from athenian.api.db import Database
from tests.conftest import DEFAULT_HEADERS
from tests.testutils.db import models_insert
from tests.testutils.factory.state import GoalFactory, GoalTemplateFactory


class TestGetGoalTemplate:
    async def test_not_found(self, client: TestClient) -> None:
        res = await self._request(client, 999, 404)
        assert res["type"] == "/errors/align/GoalTemplateNotFound"

    async def test_base(self, client: TestClient, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(id=200))

        await self._request(client, 200)

    async def test_account_mismatch(self, client: TestClient, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(id=101, account_id=3))
        await self._request(client, 101, 404)

    async def _request(self, client: TestClient, template: int, assert_status: int = 200) -> dict:
        path = f"/v1/goal_template/{template}"
        response = await client.request(method="GET", path=path, headers=DEFAULT_HEADERS)
        assert response.status == assert_status
        return await response.json()


class TestListGoalTemplates:
    async def test_base(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=201, template_id=1001),
            GoalTemplateFactory(id=1002, name="T1002"),
            GoalTemplateFactory(id=1001, name="T1001"),
        )
        res = await self._request(client, 1)
        assert [r["id"] for r in res] == [1001, 1002]
        assert [r["name"] for r in res] == ["T1001", "T1002"]

    async def test_wrong_account(self, client: TestClient) -> None:
        await self._request(client, 3, 404)

    async def _request(self, client: TestClient, account: int, assert_status: int = 200) -> dict:
        path = f"/v1/goal_templates/{account}"
        response = await client.request(method="GET", path=path, headers=DEFAULT_HEADERS)
        assert response.status == assert_status
        return await response.json()
