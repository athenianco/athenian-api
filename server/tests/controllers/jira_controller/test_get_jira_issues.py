from operator import itemgetter
from typing import Any

from freezegun import freeze_time
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

    async def test_story_points(self, mdb_rw: Database) -> None:
        issue_kwargs: dict[str, Any] = {"project_id": "1", "type_id": "1"}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="P1"),
            md_factory.JIRAIssueTypeFactory(id="1", project_id="1"),
            *jira_issue_models("1", key="P-1", story_points=5, **issue_kwargs),
            *jira_issue_models("2", key="P-2", story_points=None, **issue_kwargs),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            body = self._body(issues=["P-1", "P-2"])
            res = await self.post_json(json=body)
        assert res["issues"][0]["story_points"] == 5
        assert "story_points" not in res["issues"][1]

    @freeze_time("2011-01-07")
    async def test_acknowledge_time(self, mdb_rw: Database, sdb: Database) -> None:
        issue_kw = {"project_id": "1", "type_id": "0"}
        pr_kwargs = {"repository_full_name": "o/r"}
        mdb_models = [
            *pr_models(99, 1, 1, **pr_kwargs, created_at=dt(2011, 1, 2)),
            *pr_models(99, 2, 2, **pr_kwargs, created_at=dt(2010, 12, 20)),
            md_factory.JIRAProjectFactory(id="1", key="P"),
            md_factory.StatusFactory(id="7", category_name="To Do"),
            md_factory.StatusFactory(id="8", category_name="In Progress"),
            md_factory.JIRAIssueTypeFactory(id="0", project_id="1"),
            *jira_issue_models(
                "1",
                key="P-1",
                status_id="8",
                created=dt(2011, 1, 1),
                work_began=dt(2011, 1, 3),
                **issue_kw,
            ),
            *jira_issue_models(
                "2",
                key="P-2",
                status_id="8",
                created=dt(2011, 1, 1),
                work_began=dt(2011, 1, 4),
                **issue_kw,
            ),
            *jira_issue_models(
                "3", key="P-3", status_id="7", created=dt(2011, 1, 3), work_began=None, **issue_kw,
            ),
            *jira_issue_models(
                "4",
                key="P-4",
                created=dt(2011, 1, 3),
                status_id="8",
                work_began=dt(2011, 1, 6),
                **issue_kw,
            ),
            *pr_jira_issue_mappings((1, "1"), (2, "2")),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=99, full_name="o/r")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            body = self._body(issues=["P-1", "P-2", "P-3", "P-4"])
            res = await self.post_json(json=body)
        issues = res["issues"]
        assert [i["id"] for i in issues] == ["P-1", "P-2", "P-3", "P-4"]
        assert issues[0]["acknowledge_time"] == f"{3600*24}s"  # pr began - issue created
        assert issues[1]["acknowledge_time"] == "0s"  # pr began - issue created would < 0, so 0
        assert issues[2]["acknowledge_time"] == f"{3600*24*4}s"  # now - issue created
        assert issues[3]["acknowledge_time"] == f"{3600*24*3}s"  # issue work began - issue created


class TestGetJIRAIssuesIncludeUsers(BaseGetJIRAIssuesTests):
    async def test_include_no_issues(self) -> None:
        JIRA_USERS = GetJIRAIssuesInclude.JIRA_USERS.value
        GITHUB_USERS = GetJIRAIssuesInclude.GITHUB_USERS.value
        body = self._body(issues=["P1-33"], include=[JIRA_USERS, GITHUB_USERS])
        res = await self.post_json(json=body)
        assert res["issues"] == []
        assert res["include"] == {}

    async def test_jira_users(self, sdb: Database, mdb_rw: Database) -> None:
        JIRA_USERS = GetJIRAIssuesInclude.JIRA_USERS.value
        issue_kwargs: dict[str, Any] = {"project_id": "1", "type_id": "0"}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="P1"),
            md_factory.JIRAIssueTypeFactory(id="0", project_id="1"),
            md_factory.JIRAUserFactory(id="u0", display_name="U 0", avatar_url="http://a.co/0"),
            md_factory.JIRAUserFactory(id="u1", display_name="U 1", avatar_url="http://a.co/1"),
            md_factory.UserFactory(node_id=333, login="gh33"),
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
        assert jira_users[1]["developer"] == "github.com/gh33"

    async def test_jira_users_fixture(self) -> None:
        body = self._body(issues=["DEV-164"], include=[GetJIRAIssuesInclude.JIRA_USERS.value])
        res = await self.post_json(json=body)
        assert sorted(u["avatar"] for u in res["include"]["jira_users"]) == [
            "https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/"
            "initials/RS-0.png",
            "https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/initials/"
            "WL-5.png",
        ]
        assert [u["type"] for u in res["include"]["jira_users"]] == ["atlassian"] * 2

    async def test_github_users(self, sdb: Database, mdb_rw: Database) -> None:
        GITHUB_USERS = GetJIRAIssuesInclude.GITHUB_USERS.value
        issue_kwargs: dict[str, Any] = {"project_id": "1", "type_id": "0"}
        pr_kwargs: dict[str, Any] = {"repository_full_name": "o/r"}
        mdb_models = [
            *pr_models(99, 1, 1, user_node_id=33, merged_by_id=44, **pr_kwargs),
            md_factory.JIRAProjectFactory(id="1", key="P1"),
            md_factory.JIRAIssueTypeFactory(id="0", project_id="1"),
            md_factory.UserFactory(node_id=33, login="gh33", avatar_url="https://a/3.jpg"),
            md_factory.UserFactory(node_id=44, login="gh44", avatar_url="https://a/4.jpg"),
            *jira_issue_models("1", key="P1-1", assignee_id="u0", **issue_kwargs),
            *pr_jira_issue_mappings((1, "1")),
        ]
        body = self._body(issues=["P1-1"], include=[GITHUB_USERS])
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=99, full_name="o/r")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)

            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            res = await self.post_json(json=body)

        assert res["include"]["github_users"] == {
            "github.com/gh33": {"avatar": "https://a/3.jpg"},
            "github.com/gh44": {"avatar": "https://a/4.jpg"},
        }


class TestGetJIRAIssuesIncludeComments(BaseGetJIRAIssuesTests):
    async def test_base(self, mdb_rw: Database) -> None:
        COMMENTS = GetJIRAIssuesInclude.COMMENTS.value
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            issue_kwargs = {"project_id": "1", "type_id": "0"}
            models = [
                md_factory.JIRAProjectFactory(id="1", key="P"),
                md_factory.JIRAIssueTypeFactory(id="0", project_id="1"),
                *jira_issue_models("1", key="P-1", comments_count=0, **issue_kwargs),
                *jira_issue_models("2", key="P-2", comments_count=2, **issue_kwargs),
                *jira_issue_models("3", key="P-3", comments_count=1, **issue_kwargs),
                md_factory.JIRACommentFactory(issue_id="2", author_display_name="A0", body="RB0"),
                md_factory.JIRACommentFactory(issue_id="2", author_display_name="A1", body="RB1"),
                md_factory.JIRACommentFactory(issue_id="3", author_display_name="A2", body="RB2"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            body = self._body(issues=["P-2", "P-3", "P-1"], include=[COMMENTS])
            res = await self.post_json(json=body)

        assert (p2 := res["issues"][0])["id"] == "P-2"
        assert p2["comments"] == 2
        assert sorted(c["rendered_body"] for c in p2["comment_list"]) == ["RB0", "RB1"]
        assert sorted(c["author"] for c in p2["comment_list"]) == ["A0", "A1"]

        assert (p3 := res["issues"][1])["id"] == "P-3"
        assert p3["comments"] == 1
        assert [c["rendered_body"] for c in p3["comment_list"]] == ["RB2"]
        assert [c["author"] for c in p3["comment_list"]] == ["A2"]

        assert (p1 := res["issues"][2])["id"] == "P-1"
        assert p1["comments"] == 0
        assert p1.get("comment_list", []) == []


class TestIncludeDescription(BaseGetJIRAIssuesTests):
    async def test_base(self, mdb_rw: Database) -> None:
        DESCRIPTION = GetJIRAIssuesInclude.DESCRIPTION.value
        issue_kwargs: dict[str, Any] = {"project_id": "1", "type_id": "0"}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="P1"),
            md_factory.JIRAIssueTypeFactory(id="0", project_id="1"),
            *jira_issue_models("1", key="P-1", description="D 0", **issue_kwargs),
            *jira_issue_models("2", key="P-2", description="D 1", **issue_kwargs),
            *jira_issue_models("3", key="P-3", **issue_kwargs),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)
            body = self._body(issues=["P-1", "P-2", "P-3"], include=[DESCRIPTION])
            res = await self.post_json(json=body)

            assert [i["id"] for i in res["issues"]] == ["P-1", "P-2", "P-3"]

            assert res["issues"][0]["rendered_description"] == "D 0"
            assert res["issues"][1]["rendered_description"] == "D 1"
            assert res["issues"][2].get("rendered_description") is None

            res = await self.post_json(json=self._body(issues=["P-1"], include=[]))
            assert [i["id"] for i in res["issues"]] == ["P-1"]
            assert res["issues"][0].get("rendered_description") is None
