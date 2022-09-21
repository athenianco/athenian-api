from typing import Any

from athenian.api.db import Database
from athenian.api.models.state.models import GoalTemplate
from athenian.api.models.web import (
    DeveloperMetricID,
    GoalTemplateCreateRequest,
    GoalTemplateUpdateRequest,
    PullRequestMetricID,
)
from tests.testutils.db import DBCleaner, assert_existing_row, assert_missing_row, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.state import AccountFactory, GoalTemplateFactory
from tests.testutils.requester import Requester


class TestGetGoalTemplate(Requester):
    async def test_not_found(self) -> None:
        res = await self._request(999, 404)
        assert res["type"] == "/errors/align/GoalTemplateNotFound"

    async def test_base(self, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(id=200, name="T 1"))
        res = await self._request(200)
        assert res["id"] == 200
        assert res["name"] == "T 1"
        assert "repositories" not in res

    async def test_with_repositories(
        self,
        sdb: Database,
        mdb_rw: Database,
    ) -> None:
        await models_insert(
            sdb,
            GoalTemplateFactory(
                id=200, repositories=[[1000, None], [1001, "logical0"], [1001, "logical1"]],
            ),
        )
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_models = [
                md_factory.RepositoryFactory(node_id=1000, full_name="athenianco/repo-A"),
                md_factory.RepositoryFactory(node_id=1001, full_name="athenianco/repo-B"),
            ]
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            res = await self._request(200)
        assert res["repositories"] == [
            "github.com/athenianco/repo-A",
            "github.com/athenianco/repo-B/logical0",
            "github.com/athenianco/repo-B/logical1",
        ]

    async def test_account_mismatch(self, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(id=101, account_id=3))
        await self._request(101, 404)

    async def _request(self, template: int, assert_status: int = 200) -> dict:
        path = f"/v1/goal_template/{template}"
        response = await self.client.request(method="GET", path=path, headers=self.headers)
        assert response.status == assert_status
        return await response.json()


class TestListGoalTemplates(Requester):
    async def test_base(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalTemplateFactory(id=1002, name="T1002"),
            GoalTemplateFactory(id=1001, name="T1001"),
        )
        res = await self._request(1)
        assert [r["id"] for r in res] == [1001, 1002]
        assert [r["name"] for r in res] == ["T1001", "T1002"]
        assert not any("repositories" in r for r in res)

    async def test_repositories(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(
            sdb,
            GoalTemplateFactory(id=102, name="T0", repositories=None),
            GoalTemplateFactory(id=103, name="T1", repositories=[[200, ""]]),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_models = [
                md_factory.RepositoryFactory(node_id=200, full_name="athenianco/repo-A"),
            ]
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            res = await self._request(1)
        assert "repositories" not in res[0]
        assert res[1]["repositories"] == ["github.com/athenianco/repo-A"]

    async def test_wrong_account(self) -> None:
        await self._request(3, 404)

    async def _request(self, account: int, assert_status: int = 200) -> dict:
        path = f"/v1/goal_templates/{account}"
        response = await self.client.request(method="GET", path=path, headers=self.headers)
        assert response.status == assert_status
        return await response.json()


class BaseCreateGoalTemplateTest(Requester):
    async def _request(self, assert_status: int = 200, **kwargs: Any) -> dict:
        path = "/v1/goal_template/create"
        response = await self.client.request(
            method="POST", path=path, headers=self.headers, **kwargs,
        )
        assert response.status == assert_status
        return await response.json()


class TestCreateGoalTemplateErrors(BaseCreateGoalTemplateTest):
    async def test_empty_name(self, sdb: Database) -> None:
        req = GoalTemplateCreateRequest(
            account=1, metric=PullRequestMetricID.PR_CLOSED, name="?",
        ).to_dict()
        req["name"] = ""
        await self._request(400, json=req)
        await assert_missing_row(sdb, GoalTemplate, account_id=1)

    async def test_account_mismatch(self, sdb: Database) -> None:
        req = GoalTemplateCreateRequest(
            account=3, metric=PullRequestMetricID.PR_CLOSED, name="My template",
        )
        await self._request(404, json=req.to_dict())
        await assert_missing_row(sdb, GoalTemplate, account_id=3)

    async def test_duplicated_name(self, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(name="T0"))
        req = GoalTemplateCreateRequest(account=1, metric=PullRequestMetricID.PR_CLOSED, name="T0")
        await self._request(409, json=req.to_dict())

    async def test_invalid_metric(self, sdb: Database) -> None:
        for invalid_metric in ("foo", "", DeveloperMetricID.REVIEW_REJECTIONS):
            req = GoalTemplateCreateRequest(
                account=1, metric=PullRequestMetricID.PR_CLOSED, name="T0",
            ).to_dict()
            req["metric"] = invalid_metric
            await self._request(400, json=req)
        await assert_missing_row(sdb, GoalTemplate, account_id=1)

    async def test_invalid_repositories_format(self, sdb: Database) -> None:
        req = GoalTemplateCreateRequest(
            account=1, metric=PullRequestMetricID.PR_CLOSED, name="T0", repositories=[1],
        )
        await self._request(400, json=req.to_dict())

    async def test_unknown_repository(self, sdb: Database) -> None:
        req = GoalTemplateCreateRequest(
            account=1,
            metric=PullRequestMetricID.PR_CLOSED,
            name="T0",
            repositories=["github.com/org/repo"],
        )
        await self._request(400, json=req.to_dict())
        await assert_missing_row(sdb, GoalTemplate, name="T0")


class TestCreateGoalTemplate(BaseCreateGoalTemplateTest):
    async def test_base(self, sdb: Database) -> None:
        req = GoalTemplateCreateRequest(account=1, metric=PullRequestMetricID.PR_OPENED, name="T0")
        res = await self._request(json=req.to_dict())
        template_id = res["id"]
        row = await assert_existing_row(
            sdb, GoalTemplate, id=template_id, name="T0", metric=PullRequestMetricID.PR_OPENED,
        )
        assert row[GoalTemplate.repositories.name] is None

    async def test_with_repositories(self, sdb: Database, mdb_rw: Database) -> None:
        req = GoalTemplateCreateRequest(
            account=1,
            metric=PullRequestMetricID.PR_CLOSED,
            name="T0",
            repositories=[
                "github.com/org/a-repo",
                "github.com/org/b-repo/l",
                "github.com/org/b-repo/l2",
            ],
        )
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_models = [
                md_factory.RepositoryFactory(node_id=200, full_name="org/a-repo"),
                md_factory.RepositoryFactory(node_id=201, full_name="org/b-repo"),
            ]
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            res = await self._request(json=req.to_dict())
        template_id = res["id"]
        row = await assert_existing_row(sdb, GoalTemplate, id=template_id, account_id=1, name="T0")
        assert row[GoalTemplate.repositories.name] == [[200, None], [201, "l"], [201, "l2"]]

    async def test_same_name_different_account(self, sdb: Database) -> None:
        await models_insert(
            sdb, AccountFactory(id=11), GoalTemplateFactory(name="T0", account_id=11),
        )
        req = GoalTemplateCreateRequest(account=1, metric=PullRequestMetricID.PR_OPENED, name="T0")
        res = await self._request(json=req.to_dict())
        template_id = res["id"]
        await assert_existing_row(sdb, GoalTemplate, account_id=1, id=template_id, name="T0")


class BaseDeleteGoalTemplateTest(Requester):
    async def _request(self, template: int, assert_status: int = 200) -> dict:
        path = f"/v1/goal_template/{template}"
        response = await self.client.request(method="DELETE", path=path, headers=self.headers)
        assert response.status == assert_status
        return await response.json()


class TestDeleteGoalTemplateErrors(BaseDeleteGoalTemplateTest):
    async def test_not_found(self) -> None:
        await self._request(1121, 404)

    async def test_account_mismatch(self, sdb: Database) -> None:
        await models_insert(
            sdb, AccountFactory(id=10), GoalTemplateFactory(id=1121, account_id=10),
        )
        await self._request(1121, 404)
        await assert_existing_row(sdb, GoalTemplate, account_id=10, id=1121)


class TestDeleteGoalTemplate(BaseDeleteGoalTemplateTest):
    async def test_delete(self, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(id=1121))
        await self._request(1121)
        await assert_missing_row(sdb, GoalTemplate, id=1121)


class BaseUpdateGoalTemplateTest(Requester):
    async def _request(
        self,
        template: int,
        assert_status: int = 200,
        **kwargs: Any,
    ) -> dict:
        path = f"/v1/goal_template/{template}"
        response = await self.client.request(
            method="PUT", path=path, headers=self.headers, **kwargs,
        )
        assert response.status == assert_status
        return await response.json()


class TestUpdateGoalTemplateErrors(BaseUpdateGoalTemplateTest):
    async def test_not_found(self, sdb: Database) -> None:
        req = GoalTemplateUpdateRequest(name="new-name", metric=PullRequestMetricID.PR_DONE)
        await self._request(1121, 404, json=req.to_dict())
        await assert_missing_row(sdb, GoalTemplate, id=1121)

    async def test_account_mismatch(self, sdb: Database) -> None:
        await models_insert(
            sdb, AccountFactory(id=10), GoalTemplateFactory(id=1111, account_id=10, name="T0"),
        )
        req = GoalTemplateUpdateRequest(name="new-name", metric=PullRequestMetricID.PR_DONE)
        await self._request(1111, 404, json=req.to_dict())
        await assert_existing_row(sdb, GoalTemplate, id=1111, name="T0")

    async def test_empty_name(self, sdb: Database) -> None:
        await models_insert(sdb, AccountFactory(id=10), GoalTemplateFactory(id=111, name="T0"))
        req = GoalTemplateUpdateRequest(name="?", metric=PullRequestMetricID.PR_DONE).to_dict()
        req["name"] = ""
        await self._request(111, 400, json=req)
        await assert_existing_row(sdb, GoalTemplate, id=111, name="T0")

    async def test_null_name(self, sdb: Database) -> None:
        await models_insert(sdb, AccountFactory(id=10), GoalTemplateFactory(id=111, name="T0"))
        req = GoalTemplateUpdateRequest(name="?", metric=PullRequestMetricID.PR_DONE).to_dict()
        req["name"] = None
        await self._request(111, 400, json=req)
        await assert_existing_row(sdb, GoalTemplate, id=111, name="T0")

    async def test_invalid_repositories(self, sdb: Database) -> None:
        await models_insert(
            sdb, AccountFactory(id=10), GoalTemplateFactory(id=111, repositories=[[1, None]]),
        )
        req_body = GoalTemplateUpdateRequest(
            name="T", metric=PullRequestMetricID.PR_DONE,
        ).to_dict()
        req_body["repositories"] = 42
        await self._request(111, 400, json=req_body)

        req_body["repositories"] = ["github.com/not/existing"]
        await self._request(111, 400, json=req_body)

        row = await assert_existing_row(sdb, GoalTemplate, id=111)
        assert row[GoalTemplate.repositories.name] == [[1, None]]


class TestUpdateGoalTemplate(BaseUpdateGoalTemplateTest):
    async def test_update_name(self, sdb: Database) -> None:
        await models_insert(sdb, AccountFactory(id=10), GoalTemplateFactory(id=111, name="T0"))
        req = GoalTemplateUpdateRequest(name="T1", metric=PullRequestMetricID.PR_DONE)
        await self._request(111, json=req.to_dict())
        await assert_existing_row(sdb, GoalTemplate, id=111, name="T1")

    async def test_update_metric(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            AccountFactory(id=10),
            GoalTemplateFactory(id=111, repositories=[[10, None]]),
        )
        req = GoalTemplateUpdateRequest(name="T1", metric=PullRequestMetricID.PR_MERGED)
        await self._request(111, json=req.to_dict())
        row = await assert_existing_row(sdb, GoalTemplate, id=111)
        assert row[GoalTemplate.metric.name] == req.metric

    async def test_update_repositories(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(
            sdb,
            AccountFactory(id=10),
            GoalTemplateFactory(id=111, repositories=[[10, None]]),
        )
        req = GoalTemplateUpdateRequest(
            name="T1",
            metric=PullRequestMetricID.PR_DONE,
            repositories=["g.com/o/a", "g.com/o/b", "g.com/o/c/l1", "g.com/o/c/l2"],
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_models = [
                md_factory.RepositoryFactory(node_id=20, full_name="o/a"),
                md_factory.RepositoryFactory(node_id=21, full_name="o/b"),
                md_factory.RepositoryFactory(node_id=22, full_name="o/c"),
            ]
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            await self._request(111, json=req.to_dict())

        row = await assert_existing_row(sdb, GoalTemplate, id=111)
        assert row[GoalTemplate.repositories.name] == [
            [20, None],
            [21, None],
            [22, "l1"],
            [22, "l2"],
        ]
