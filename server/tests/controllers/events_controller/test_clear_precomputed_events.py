from datetime import datetime, timezone
from typing import Any

from freezegun import freeze_time
import pytest
import sqlalchemy as sa

from athenian.api.models.precomputed.models import (
    GitHubDeploymentFacts,
    GitHubDonePullRequestFacts,
    GitHubMergedPullRequestFacts,
    GitHubReleaseFacts,
)
from tests.testutils.db import models_insert
from tests.testutils.factory.precomputed import GitHubDonePullRequestFactsFactory
from tests.testutils.requester import Requester


class TestClearPrecomputedEvents(Requester):
    @pytest.mark.parametrize("devenv", [False, True])
    @freeze_time("2020-01-01")
    async def test_clear_releases_smoke(self, pdb, disable_default_user, app, devenv):
        await models_insert(
            pdb,
            GitHubDonePullRequestFactsFactory(
                acc_id=1,
                release_match="event",
                pr_node_id=111,
                repository_full_name="src-d/go-git",
                pr_created_at=datetime.now(timezone.utc),
                pr_done_at=datetime.now(timezone.utc),
            ),
            GitHubMergedPullRequestFacts(
                acc_id=1,
                release_match="event",
                pr_node_id=222,
                repository_full_name="src-d/go-git",
                merged_at=datetime.now(timezone.utc),
                checked_until=datetime.now(timezone.utc),
                labels={},
                activity_days=[],
            ).create_defaults(),
            GitHubReleaseFacts(
                acc_id=1,
                id=333,
                release_match="event",
                repository_full_name="src-d/go-git",
                published_at=datetime.now(timezone.utc),
                data=b"data",
            ).create_defaults(),
        )
        body = {"account": 1, "repositories": ["{1}"], "targets": ["release"]}
        app._devenv = devenv
        await self._request(json=body)
        for table, n in (
            (GitHubDonePullRequestFacts, 292),
            (GitHubMergedPullRequestFacts, 246),
            (GitHubReleaseFacts, 53),
        ):
            assert len(await pdb.fetch_all(sa.select(table))) == n, table

    @freeze_time("2020-01-01")
    async def test_clear_deployments(self, pdb, disable_default_user):
        await pdb.execute(
            sa.insert(GitHubDeploymentFacts).values(
                GitHubDeploymentFacts(
                    acc_id=1,
                    deployment_name="Dummy deployment",
                    release_matches="abracadabra",
                    format_version=1,
                    data=b"0",
                ).explode(with_primary_keys=True),
            ),
        )
        body = {"account": 1, "repositories": ["{1}"], "targets": ["deployment"]}
        await self._request(json=body)
        rows = await pdb.fetch_all(sa.select(GitHubDeploymentFacts))
        assert len(rows) == 1
        row = rows[0]
        assert row[GitHubDeploymentFacts.deployment_name.name] == "Dummy deployment"
        assert len(row[GitHubDeploymentFacts.data.name]) > 1

    @pytest.mark.parametrize(
        "status, body",
        [
            (200, {"account": 1, "repositories": ["github.com/src-d/go-git"], "targets": []}),
            (
                400,
                {"account": 1, "repositories": ["github.com/src-d/go-git"], "targets": ["wrong"]},
            ),
            (422, {"account": 2, "repositories": ["github.com/src-d/go-git"], "targets": []}),
            (404, {"account": 3, "repositories": ["github.com/src-d/go-git"], "targets": []}),
            (
                403,
                {
                    "account": 1,
                    "repositories": ["github.com/athenianco/athenian-api"],
                    "targets": [],
                },
            ),
        ],
    )
    async def test_nasty_input(self, body, status, disable_default_user):
        await self._request(status, json=body)

    async def _request(self, assert_status: int = 200, **kwargs: Any) -> None:
        path = "/v1/events/clear_cache"
        response = await self.client.request(
            method="POST", path=path, headers=self.headers, **kwargs,
        )
        assert response.status == assert_status
