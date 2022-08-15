from typing import Any

from aiohttp.test_utils import TestClient

from athenian.api.db import Database
from athenian.api.models.state.models import GoalTemplate
from athenian.api.models.web import (
    DeveloperMetricID,
    GoalTemplateCreateRequest,
    PullRequestMetricID,
)
from tests.conftest import DEFAULT_HEADERS
from tests.testutils.db import assert_existing_row, assert_missing_row, models_insert
from tests.testutils.factory.state import AccountFactory, GoalFactory, GoalTemplateFactory


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


class BaseCreateGoalTemplateTest:
    async def _request(self, client: TestClient, assert_status: int = 200, **kwargs: Any) -> dict:
        path = "/v1/goal_template/create"
        headers = DEFAULT_HEADERS
        response = await client.request(method="POST", path=path, headers=headers, **kwargs)
        assert response.status == assert_status
        return await response.json()


class TestCreateGoalTemplateErrors(BaseCreateGoalTemplateTest):
    async def test_empty_name(self, client: TestClient, sdb: Database) -> None:
        req = GoalTemplateCreateRequest(1, PullRequestMetricID.PR_CLOSED, "")
        await self._request(client, 400, json=req.to_dict())
        await assert_missing_row(sdb, GoalTemplate, account_id=1)

    async def test_account_mismatch(self, client: TestClient, sdb: Database) -> None:
        req = GoalTemplateCreateRequest(3, PullRequestMetricID.PR_CLOSED, "My template")
        await self._request(client, 404, json=req.to_dict())
        await assert_missing_row(sdb, GoalTemplate, account_id=3)

    async def test_duplicated_name(self, client: TestClient, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(name="T0"))
        req = GoalTemplateCreateRequest(1, PullRequestMetricID.PR_CLOSED, "T0")
        await self._request(client, 409, json=req.to_dict())

    async def test_invalid_metric(self, client: TestClient, sdb: Database) -> None:
        for invalid_metric in ("foo", "", DeveloperMetricID.REVIEW_REJECTIONS):
            req = GoalTemplateCreateRequest(1, invalid_metric, "T0")
            await self._request(client, 400, json=req.to_dict())
        await assert_missing_row(sdb, GoalTemplate, account_id=1)


class TestCreateGoalTemplate(BaseCreateGoalTemplateTest):
    async def test_base(self, client: TestClient, sdb: Database) -> None:
        req = GoalTemplateCreateRequest(1, PullRequestMetricID.PR_OPENED, "T0")
        res = await self._request(client, json=req.to_dict())
        template_id = res["id"]
        await assert_existing_row(sdb, GoalTemplate, account_id=1, id=template_id, name="T0")

    async def test_same_name_different_account(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb, AccountFactory(id=11), GoalTemplateFactory(name="T0", account_id=11),
        )
        req = GoalTemplateCreateRequest(1, PullRequestMetricID.PR_OPENED, "T0")
        res = await self._request(client, json=req.to_dict())
        template_id = res["id"]
        await assert_existing_row(sdb, GoalTemplate, account_id=1, id=template_id, name="T0")


class BaseDeleteGoalTemplateTest:
    async def _request(self, client: TestClient, template: int, assert_status: int = 200) -> dict:
        path = f"/v1/goal_template/{template}"
        response = await client.request(method="DELETE", path=path, headers=DEFAULT_HEADERS)
        assert response.status == assert_status
        return await response.json()


class TestDeleteGoalTemplateErrors(BaseDeleteGoalTemplateTest):
    async def test_not_found(self, client: TestClient) -> None:
        await self._request(client, 1121, 404)

    async def test_account_mismatch(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb, AccountFactory(id=10), GoalTemplateFactory(id=1121, account_id=10),
        )
        await self._request(client, 1121, 404)
        await assert_existing_row(sdb, GoalTemplate, account_id=10, id=1121)


class TestDeleteGoalTemplate(BaseDeleteGoalTemplateTest):
    async def test_delete(self, client: TestClient, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(id=1121))
        await self._request(client, 1121)
        await assert_missing_row(sdb, GoalTemplateFactory, id=1121)
