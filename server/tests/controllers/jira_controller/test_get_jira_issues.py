from typing import Any

import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.models.state.models import AccountJiraInstallation
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_ACCOUNT_ID
from tests.testutils.factory.wizards import jira_issue_models
from tests.testutils.requester import Requester
from tests.testutils.time import dt


class BaseGetJIRAIssuesTests(Requester):
    path = "/v1/get/jira_issues"

    @classmethod
    def _body(self, **kwargs: Any) -> dict:
        kwargs.setdefault("account", DEFAULT_ACCOUNT_ID)
        return kwargs


class TestGetJIRAIssuesErrors(BaseGetJIRAIssuesTests):
    async def test_no_input_issue_keys(self) -> None:
        await self.post_json(json=self._body(issues=[]), assert_status=400)

    async def test_account_mismatch(self, sdb: Database) -> None:
        body = self._body(account=3, issues=["DEV-1012"])
        res = await self.post_json(json=body, assert_status=404)
        assert "Account 3 does not exist" in res["detail"]

    async def test_jira_not_installed(self, sdb: Database) -> None:
        await sdb.execute(
            sa.delete(AccountJiraInstallation).where(AccountJiraInstallation.account_id == 1),
        )
        body = self._body(issues=["DEV-1012"])
        res = await self.post_json(json=body, assert_status=422)
        assert "JIRA has not been installed" in res["detail"]


class TestGetJIRAIssues(BaseGetJIRAIssuesTests):
    async def test_none_found(self) -> None:
        body = self._body(issues=["DEV-1234"])
        res = await self.post_json(json=body)
        assert res == {"issues": []}

    async def test_smoke(self) -> None:
        body = self._body(issues=["DEV-1012"])
        res = await self.post_json(json=body)
        assert len(res["issues"]) == 1
        assert res["issues"][0]["id"] == "DEV-1012"

    async def test_request_order_is_preserved(self) -> None:
        body = self._body(issues=["DEV-90", "DEV-69", "DEV-729", "DEV-1012"])
        res = await self.post_json(json=body)
        assert [i["id"] for i in res["issues"]] == ["DEV-90", "DEV-69", "DEV-729", "DEV-1012"]

        # not found keys are ignored
        body = self._body(issues=["DEV-69", "DEV-1012", "DEV-YYY", "DEV-90", "DEV-729", "DEV-XXX"])
        res = await self.post_json(json=body)
        assert [i["id"] for i in res["issues"]] == ["DEV-69", "DEV-1012", "DEV-90", "DEV-729"]

    async def test_status(self, sdb: Database, mdb_rw: Database) -> None:
        issue_kwargs = {"project_id": "1"}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="P1"),
            md_factory.JIRAIssueTypeFactory(id="0", name="Type 0", project_id="1"),
            md_factory.JIRAIssueTypeFactory(id="1", name="Type 1", project_id="1"),
            *jira_issue_models("1", key="P1-1", type_id="0", **issue_kwargs),
            *jira_issue_models("2", key="P1-2", type_id="0", **issue_kwargs),
            *jira_issue_models("3", key="P1-3", type_id="1", **issue_kwargs),
            *jira_issue_models("4", key="P1-4", type_id="1", **issue_kwargs),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            body = self._body(issues=["P1-2", "P1-3", "P1-1"])
            res = await self.post_json(json=body)
        issues = res["issues"]

        assert [i["id"] for i in issues] == ["P1-2", "P1-3", "P1-1"]
        assert [i["type"] for i in issues] == ["Type 0", "Type 1", "Type 0"]
        assert all("pr" not in i for i in issues)

    async def test_lead_time_life_time(self, sdb: Database, mdb_rw: Database) -> None:
        issue_kwargs = {"project_id": "1"}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="P1"),
            md_factory.JIRAIssueTypeFactory(id="0", name="Type 0", project_id="1"),
            *jira_issue_models(
                "1",
                key="P1-1",
                type_id="0",
                created=dt(2016, 1, 1, 1),
                work_began=dt(2016, 1, 1, 2),
                **issue_kwargs,
                resolved=dt(2016, 1, 1, 4),
            ),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            body = self._body(issues=["P1-2", "P1-3", "P1-1"])
            res = await self.post_json(json=body)
        issues = res["issues"]

        assert [i["id"] for i in issues] == ["P1-1"]
        assert issues[0]["lead_time"] == f"{3600 * 2}s"
        assert issues[0]["life_time"] == f"{3600 * 3}s"
