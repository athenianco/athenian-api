from datetime import date
from operator import itemgetter
from typing import Any, Sequence

import pytest
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.models.state.models import AccountJiraInstallation
from athenian.api.models.web.search_prs import SearchPullRequestsRequest
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_ACCOUNT_ID
from tests.testutils.factory.state import TeamFactory
from tests.testutils.factory.wizards import insert_repo, pr_models
from tests.testutils.requester import Requester
from tests.testutils.time import dt


class BaseSearchPRsTest(Requester):
    async def _request(self, assert_status: int = 200, **kwargs: Any) -> dict:
        path = "/private/search/pull_requests"
        client = self.client

        response = await client.request(method="POST", path=path, headers=self.headers, **kwargs)
        assert response.status == assert_status
        return await response.json()

    def _body(
        self,
        account: int = DEFAULT_ACCOUNT_ID,
        date_from: date = date(2019, 11, 1),
        date_to: date = date(2019, 12, 1),
        **kwargs: Any,
    ) -> dict:
        req = SearchPullRequestsRequest(
            account=account, date_from=date_from, date_to=date_to, **kwargs,
        )
        body = req.to_dict()
        body["date_from"] = body["date_from"].isoformat()
        body["date_to"] = body["date_to"].isoformat()
        return body

    async def _fetch_pr_numbers(self, **kwargs) -> Sequence[int]:
        res = await self._request(**kwargs)
        return tuple(pr["number"] for pr in res["pull_requests"])


class TestSearchPRsError(BaseSearchPRsTest):
    async def test_invalid_repositories(self, sdb: Database) -> None:
        body = self._body(repositories=["github.com/wrong/repo"])
        await self._request(403, json=body)

    async def test_invalid_team_in_participants(self, sdb: Database) -> None:
        body = self._body(participants={"author": ["{42}"]})
        res = await self._request(404, json=body)
        assert res["title"] == "Team not found"

    @pytest.mark.xfail
    async def test_empty_team_in_participants(self, sdb: Database) -> None:
        # TODO: this fails with unhandled exception
        await models_insert(sdb, TeamFactory(id=42, members=[]))
        body = self._body(participants={"author": ["{42}"]})
        await self._request(400, json=body)

    async def test_jira_not_installed(self, sdb: Database, mdb_rw: Database) -> None:
        await sdb.execute(
            sa.delete(AccountJiraInstallation).where(AccountJiraInstallation.account_id == 1),
        )
        body = self._body(jira={"projects": ["DEV"]})
        res = await self._request(422, json=body)
        assert res["detail"] == "JIRA has not been installed to the metadata yet."

    async def test_invalid_date(self, sdb: Database) -> None:
        body = self._body()
        body["date_from"] = "foo"
        await self._request(400, json=body)

        body.pop("date_from")
        await self._request(400, json=body)


class TestSearchPRs(BaseSearchPRsTest):
    async def test_smoke(self, sdb: Database) -> None:
        body = self._body()
        res = await self._request(json=body)
        assert len(res["pull_requests"]) == 7

    async def test_repositories(self, sdb: Database, mdb_rw: Database) -> None:
        times = {"created_at": dt(2019, 11, 15), "updated_at": dt(2019, 11, 20)}
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo0 = md_factory.RepositoryFactory(node_id=99, full_name="org0/repo0")
            await insert_repo(repo0, mdb_cleaner, mdb_rw, sdb)
            repo1 = md_factory.RepositoryFactory(node_id=98, full_name="org0/repo1")
            await insert_repo(repo1, mdb_cleaner, mdb_rw, sdb)
            models = [
                *pr_models(99, 10, 1010, repository_full_name="org0/repo0", **times),
                *pr_models(98, 11, 1011, repository_full_name="org0/repo1", **times),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            body = self._body()
            assert {1010, 1011} & set(await self._fetch_pr_numbers(json=body)) == {1010, 1011}

            body = self._body(repositories=["github.com/org0/repo0", "github.com/org0/repo1"])
            assert {1010, 1011} & set(await self._fetch_pr_numbers(json=body)) == {1010, 1011}

            body = self._body(repositories=["github.com/org0/repo0"])
            assert {1010, 1011} & set(await self._fetch_pr_numbers(json=body)) == {1010}

            body = self._body(repositories=["github.com/org0/repo1"])
            assert {1010, 1011} & set(await self._fetch_pr_numbers(json=body)) == {1011}

    async def test_participants(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10, members=[40390]))
        body = self._body(participants={"author": ["{10}"]})
        res = await self._request(json=body)
        assert res["pull_requests"] == [
            {"number": 1247, "repository": "github.com/src-d/go-git"},
        ]

        await models_insert(sdb, TeamFactory(id=11, parent_id=10, members=[39874]))
        # now also PR authored subteam are selected
        body = self._body(participants={"author": ["{10}"]})
        res = await self._request(json=body)
        assert sorted(res["pull_requests"], key=itemgetter("number")) == [
            {"number": 1247, "repository": "github.com/src-d/go-git"},
            {"number": 1248, "repository": "github.com/src-d/go-git"},
        ]

    async def test_jira(self, sdb: Database, mdb_rw: Database) -> None:
        times = {"created_at": dt(2019, 11, 15), "updated_at": dt(2019, 11, 20)}
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=99, full_name="org0/repo0")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
            models = [
                *pr_models(99, 10, 110, repository_full_name="org0/repo0", **times),
                *pr_models(99, 11, 111, repository_full_name="org0/repo0", **times),
                md_factory.JIRAProjectFactory(id="200", key="DD"),
                md_factory.JIRAProjectFactory(id="201", key="EE"),
                md_factory.JIRAIssueFactory(id="20", priority_name="extreme", project_id="200"),
                md_factory.JIRAIssueFactory(id="21", priority_name="medium", project_id="200"),
                md_factory.JIRAIssueFactory(id="22", project_id="201"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=10, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=11, jira_id="21"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=11, jira_id="22"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            body = self._body()
            assert {110, 111} & set(await self._fetch_pr_numbers(json=body)) == {110, 111}

            body = self._body(jira={"priorities": ["extreme"]})
            assert {110, 111} & set(await self._fetch_pr_numbers(json=body)) == {110}

            body = self._body(jira={"priorities": ["medium"]})
            assert {110, 111} & set(await self._fetch_pr_numbers(json=body)) == {111}

            body = self._body(jira={"projects": ["EE"]})
            assert {111} & set(await self._fetch_pr_numbers(json=body)) == {111}
