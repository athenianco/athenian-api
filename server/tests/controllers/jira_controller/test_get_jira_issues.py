from operator import itemgetter
from typing import Any

import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.models.state.models import AccountJiraInstallation
from athenian.api.models.web import GetJIRAIssuesInclude
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_ACCOUNT_ID
from tests.testutils.factory.state import MappedJIRAIdentityFactory
from tests.testutils.factory.wizards import (
    insert_repo,
    jira_issue_models,
    pr_jira_issue_mappings,
    pr_models,
)
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
        body = self._body(issues=["DEV-164"])
        res = await self.post_json(json=body)
        assert len(res["issues"]) == 1
        issue = res["issues"][0]
        assert issue["id"] == "DEV-164"
        assert issue["type"] == "Story"
        assert issue["title"] == "Set the max number of avatars to show to 5"

        assert len(issue["prs"]) == 1
        assert issue["prs"][0]["title"] == "idxfile: optimise allocations in readObjectNames"
        assert issue["prs"][0]["number"] == 845

    async def test_request_order_is_preserved(self) -> None:
        body = self._body(issues=["DEV-90", "DEV-69", "DEV-729", "DEV-1012"])
        res = await self.post_json(json=body)
        assert [i["id"] for i in res["issues"]] == ["DEV-90", "DEV-69", "DEV-729", "DEV-1012"]

        # not found keys are ignored
        body = self._body(issues=["DEV-69", "DEV-1012", "DEV-YYY", "DEV-90", "DEV-729", "DEV-XXX"])
        res = await self.post_json(json=body)
        assert [i["id"] for i in res["issues"]] == ["DEV-69", "DEV-1012", "DEV-90", "DEV-729"]

    async def test_status(self, mdb_rw: Database) -> None:
        issue_kwargs: dict[str, Any] = {"project_id": "1"}
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

    async def test_lead_time_life_time(self, mdb_rw: Database) -> None:
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

    async def test_mapped_prs(self, sdb: Database, mdb_rw: Database) -> None:
        issue_kwargs: dict[str, Any] = {"project_id": "1", "type_id": "0"}
        pr_kwargs: dict[str, Any] = {"repository_full_name": "o/r"}
        mdb_models = [
            *pr_models(99, 1, 1, **pr_kwargs),
            *pr_models(99, 11, 11, **pr_kwargs),
            *pr_models(99, 2, 2, additions=10, title="PR 2 Title", **pr_kwargs),
            md_factory.JIRAProjectFactory(id="1", key="P1"),
            md_factory.JIRAIssueTypeFactory(id="0", name="Type 0", project_id="1"),
            *jira_issue_models("1", key="P1-1", **issue_kwargs),
            *jira_issue_models("2", key="P1-2", **issue_kwargs),
            *jira_issue_models("3", key="P1-3", **issue_kwargs),
            *pr_jira_issue_mappings((1, "1"), (11, "1"), (2, "2")),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=99, full_name="o/r")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)

            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            body = self._body(issues=["P1-2", "P1-1", "P1-3"])
            res = await self.post_json(json=body)

        issues = res["issues"]
        assert [i["id"] for i in issues] == ["P1-2", "P1-1", "P1-3"]

        issue_2 = issues[0]
        assert [pr["number"] for pr in issue_2["prs"]] == [2]
        assert issue_2["prs"][0]["size_added"] == 10
        assert issue_2["prs"][0]["title"] == "PR 2 Title"

        issue_1 = issues[1]
        assert sorted(pr["number"] for pr in issue_1["prs"]) == [1, 11]

        issue_3 = issues[2]
        assert "prs" not in issue_3


class TestGetJIRAIssuesInclude(BaseGetJIRAIssuesTests):
    async def test_include_no_issues(self) -> None:
        body = self._body(issues=["P1-33"], include=[GetJIRAIssuesInclude.JIRA_USERS.value])
        res = await self.post_json(json=body)
        assert res["issues"] == []
        assert res["include"] == {}

    async def test_users(self, sdb: Database, mdb_rw: Database) -> None:
        JIRA_USERS = GetJIRAIssuesInclude.JIRA_USERS.value
        issue_kwargs: dict[str, Any] = {"project_id": "1", "type_id": "0"}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="P1"),
            md_factory.JIRAIssueTypeFactory(id="0", project_id="1"),
            md_factory.JIRAUserFactory(id="u0", display_name="U 0", avatar_url="http://a.co/0"),
            md_factory.JIRAUserFactory(id="u1", display_name="U 1", avatar_url="http://a.co/1"),
            md_factory.UserFactory(node_id=333, login="gh333"),
            *jira_issue_models("1", key="P1-1", assignee_id="u0", **issue_kwargs),
            *jira_issue_models("2", key="P1-2", commenters_ids=["u1"], **issue_kwargs),
        ]

        await models_insert(sdb, MappedJIRAIdentityFactory(jira_user_id="u1", github_user_id=333))

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)
            body = self._body(issues=["P1-1", "P1-2"], include=[JIRA_USERS])
            res = await self.post_json(json=body)
        jira_users = sorted(res["include"]["jira_users"], key=itemgetter("name"))
        assert jira_users[0]["name"] == "U 0"
        assert jira_users[0]["type"] == "atlassian"
        assert jira_users[0]["avatar"] == "http://a.co/0"
        assert jira_users[1]["name"] == "U 1"
        assert jira_users[1]["type"] == "atlassian"
        assert jira_users[1]["avatar"] == "http://a.co/1"
        assert jira_users[1]["developer"] == "github.com/user-333"

    async def test_users_fixture(self) -> None:
        body = self._body(issues=["DEV-164"], include=[GetJIRAIssuesInclude.JIRA_USERS.value])
        res = await self.post_json(json=body)
        assert sorted(u["avatar"] for u in res["include"]["jira_users"]) == [
            "https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/"
            "initials/RS-0.png",
            "https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/initials/"
            "WL-5.png",
        ]
        assert [u["type"] for u in res["include"]["jira_users"]] == ["atlassian"] * 2
