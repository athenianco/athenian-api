from aiohttp.test_utils import TestClient

from athenian.api.align.goals.templates import TEMPLATES_COLLECTION
from tests.conftest import DEFAULT_HEADERS


class TestGetGoalTemplate:
    async def test_from_predefined_templates(self, client: TestClient) -> None:
        res = await self._request(client, 5)
        assert res == {
            "id": 5,
            "name": TEMPLATES_COLLECTION[5]["name"],
            "metric": TEMPLATES_COLLECTION[5]["metric"],
        }

    async def test_not_found(self, client: TestClient) -> None:
        await self._request(client, 999, 404)

    async def _request(self, client: TestClient, template: int, assert_status: int = 200) -> dict:
        path = f"/v1/goal_template/{template}"
        response = await client.request(method="GET", path=path, headers=DEFAULT_HEADERS)
        assert response.status == assert_status
        return await response.json()


class TestListGoalTemplates:
    async def test_from_predefined_templates(self, client: TestClient) -> None:
        res = await self._request(client, 1)
        # result is ordered by ID
        expected = [template_def for _, template_def in sorted(TEMPLATES_COLLECTION.items())]

        assert [r["name"] for r in res] == [t["name"] for t in expected]
        assert [r["metric"] for r in res] == [t["metric"] for t in expected]

    async def test_wrong_account(self, client: TestClient) -> None:
        await self._request(client, 3, 404)

    async def _request(self, client: TestClient, account: int, assert_status: int = 200) -> dict:
        path = f"/v1/goal_templates/{account}"
        response = await client.request(method="GET", path=path, headers=DEFAULT_HEADERS)
        assert response.status == assert_status
        return await response.json()