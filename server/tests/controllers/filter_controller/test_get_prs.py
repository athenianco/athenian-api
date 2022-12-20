from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from athenian.api.db import Database
from athenian.api.models.web import (
    DeployedComponent,
    DeploymentNotification,
    PullRequestEvent,
    PullRequestSet,
    PullRequestStage,
)
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.wizards import insert_repo, pr_models
from tests.testutils.requester import Requester


class BaseGetPRsTest(Requester):
    async def _request(self, assert_status=200, **kwargs: Any) -> PullRequestSet:
        path = "/v1/get/pull_requests"
        headers = self.headers
        response = await self.client.request(method="POST", path=path, headers=headers, **kwargs)
        assert response.status == assert_status
        return await response.json()

    def _body(self, **kwargs: Any) -> dict:
        kwargs.setdefault("account", 1)
        return kwargs

    def _response_pr_numbers(self, response: dict) -> list[int]:
        return [pr["number"] for pr in response["data"]]


class TestGetPRs(BaseGetPRsTest):
    async def test_get_prs_smoke(self) -> None:
        body = self._body(
            prs=[{"repository": "github.com/src-d/go-git", "numbers": list(range(1000, 1100))}],
        )
        res = await self._request(json=body)
        model = PullRequestSet.from_dict(res)
        assert len(model.data) == 51
        assert len(model.include.users) == 40

    async def test_get_prs_deployments(self, precomputed_deployments, detect_deployments) -> None:
        body = self._body(
            prs=[{"repository": "github.com/src-d/go-git", "numbers": [1160, 1179, 1168]}],
            environments=["production"],
        )
        res = await self._request(json=body)
        prs = PullRequestSet.from_dict(res)

        for pr in prs.data:
            if pr.number in (1160, 1179):
                assert pr.stage_timings.deploy["production"] > timedelta(0)
                assert PullRequestEvent.DEPLOYED in pr.events_now
                assert PullRequestStage.DEPLOYED in pr.stages_now
                assert pr.deployments == ["Dummy deployment"]
            if pr.number == 1168:
                assert not pr.stage_timings.deploy
                assert PullRequestEvent.DEPLOYED not in pr.events_now
                assert PullRequestStage.DEPLOYED not in pr.stages_now
                assert not pr.deployments
        assert prs.include.deployments == {
            "Dummy deployment": DeploymentNotification(
                components=[
                    DeployedComponent(
                        repository="github.com/src-d/go-git",
                        reference="v4.13.1 (0d1a009cbb604db18be960db5f1525b99a55d727)",
                    ),
                ],
                environment="production",
                name="Dummy deployment",
                url=None,
                date_started=datetime(2019, 11, 1, 12, 0, tzinfo=timezone.utc),
                date_finished=datetime(2019, 11, 1, 12, 15, tzinfo=timezone.utc),
                conclusion="SUCCESS",
                labels=None,
            ),
        }

    async def test_prs_order(self, sdb: Database, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo0 = md_factory.RepositoryFactory(node_id=99, full_name="org0/r0")
            await insert_repo(repo0, mdb_cleaner, mdb_rw, sdb)
            repo1 = md_factory.RepositoryFactory(node_id=98, full_name="org0/r1")
            await insert_repo(repo1, mdb_cleaner, mdb_rw, sdb)
            models = [
                *pr_models(99, 10, 1, repository_full_name="org0/r0"),
                *pr_models(99, 11, 2, repository_full_name="org0/r0"),
                *pr_models(98, 12, 1, repository_full_name="org0/r1"),
                *pr_models(98, 13, 3, repository_full_name="org0/r1"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            body = self._body(
                prs=[
                    {"repository": "github.com/org0/r0", "numbers": [2]},
                    {"repository": "github.com/org0/r1", "numbers": [3, 1]},
                    {"repository": "github.com/org0/r0", "numbers": [1]},
                ],
            )

            res = await self._request(json=body)

            prs = [(pr["repository"], pr["number"]) for pr in res["data"]]
            assert prs == [
                ("github.com/org0/r0", 2),
                ("github.com/org0/r1", 3),
                ("github.com/org0/r1", 1),
                ("github.com/org0/r0", 1),
            ]


class TestGetPRsErrors(BaseGetPRsTest):
    @pytest.mark.parametrize(
        "account, repo, numbers, status",
        [
            (1, "bitbucket.org", [1, 2, 3], 400),
            (2, "github.com/src-d/go-git", [1, 2, 3], 422),
            (3, "github.com/src-d/go-git", [1, 2, 3], 404),
            (4, "github.com/src-d/go-git", [1, 2, 3], 404),
            (1, "github.com/whatever/else", [1, 2, 3], 403),
        ],
    )
    async def test_get_prs_nasty_input(self, account, repo, numbers, status):
        body = self._body(account=account, prs=[{"repository": repo, "numbers": numbers}])
        await self._request(status, json=body)
