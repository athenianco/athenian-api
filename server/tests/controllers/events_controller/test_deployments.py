from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest
import sqlalchemy as sa

from athenian.api.async_utils import gather
from athenian.api.controllers.events_controller.deployments import (
    resolve_deployed_component_references,
)
from athenian.api.db import Database
from athenian.api.models.persistentdata.models import (
    DeployedComponent,
    DeployedLabel,
    DeploymentNotification,
)
from athenian.api.models.state.models import AccountGitHubAccount
from tests.testutils.requester import Requester


class TestNotifyDeployments(Requester):
    @pytest.mark.parametrize(
        "ref, vhash, resolved",
        [
            ("4.2.0", "y9c5A0Df", True),
            ("v4.2.0", "y9c5A0Df", True),
            ("1d28459504251497e0ce6132a0fadd5eb44ffd22", "Cd34s0Jb", True),
            ("1d28459", "Cd34s0Jb", True),
            ("xxx", "u5VWde@k", False),
        ],
    )
    async def test_smoke(
        self,
        token,
        rdb,
        ref,
        vhash,
        resolved,
        disable_default_user,
    ):
        await _delete_deployments(rdb)
        body = [
            {
                "components": [{"repository": "github.com/src-d/go-git", "reference": ref}],
                "environment": "production",
                "date_started": "2021-01-12T00:00:00Z",
                "date_finished": "2021-01-12T01:00:00Z",
                "conclusion": "SUCCESS",
                "labels": {"one": 1, 2: "two"},
            },
        ]
        response = await self._request(token, json=body)

        rows = await rdb.fetch_all(sa.select(DeployedComponent))
        assert len(rows) == 1
        row = dict(rows[0])
        assert response == {
            "deployments": [
                {"name": row[DeployedComponent.deployment_name.name], "resolved": resolved},
            ],
        }
        created_at = row[DeployedComponent.created_at.name]
        assert isinstance(created_at, datetime)
        del row[DeployedComponent.created_at.name]
        if ref == "xxx":
            commit = None
        else:
            commit = 2755428
        name = f"prod-2021-01-12-{vhash}"
        assert row[DeployedComponent.resolved_at.name] or ref == "xxx"
        del row[DeployedComponent.resolved_at.name]
        assert row == {
            DeployedComponent.account_id.name: 1,
            DeployedComponent.deployment_name.name: name,
            DeployedComponent.repository_node_id.name: 40550,
            DeployedComponent.reference.name: ref,
            DeployedComponent.resolved_commit_node_id.name: commit,
        }
        rows = await rdb.fetch_all(sa.select(DeployedLabel))
        assert len(rows) == 2
        assert dict(rows[0]) == {
            DeployedLabel.account_id.name: 1,
            DeployedLabel.deployment_name.name: name,
            DeployedLabel.key.name: "one",
            DeployedLabel.value.name: 1,
        }
        assert dict(rows[1]) == {
            DeployedLabel.account_id.name: 1,
            DeployedLabel.deployment_name.name: name,
            DeployedLabel.key.name: "2",
            DeployedLabel.value.name: "two",
        }
        rows = await rdb.fetch_all(sa.select(DeploymentNotification))
        assert len(rows) == 1
        row = dict(rows[0])
        created_at = row[DeploymentNotification.created_at.name]
        assert isinstance(created_at, datetime)
        del row[DeploymentNotification.created_at.name]
        updated_at = row[DeploymentNotification.updated_at.name]
        assert isinstance(updated_at, datetime)
        del row[DeploymentNotification.updated_at.name]
        tzinfo = timezone.utc if rdb.url.dialect == "postgresql" else None
        assert row == {
            DeploymentNotification.account_id.name: 1,
            DeploymentNotification.name.name: name,
            DeploymentNotification.conclusion.name: "SUCCESS",
            DeploymentNotification.started_at.name: datetime(2021, 1, 12, 0, 0, tzinfo=tzinfo),
            DeploymentNotification.finished_at.name: datetime(2021, 1, 12, 1, 0, tzinfo=tzinfo),
            DeploymentNotification.url.name: None,
            DeploymentNotification.environment.name: "production",
        }

    async def test_duplicate(self, token, disable_default_user):
        body = [
            {
                "components": [{"repository": "github.com/src-d/go-git", "reference": "xxx"}],
                "environment": "staging",
                "date_started": "2021-01-12T00:00:00Z",
                "date_finished": "2021-01-12T00:01:00Z",
                "conclusion": "FAILURE",
                "labels": {"one": 1},
            },
        ]
        await self._request(token, json=body)
        await self._request(token, 409, json=body)

    @pytest.mark.parametrize(
        "body, code",
        [
            (
                {
                    "components": [{"repository": "github.com/src-d/go-git", "reference": "xxx"}],
                    "environment": "production",
                    "date_started": "2021-01-12T00:00:00Z",
                    "date_finished": "2021-01-12T01:00:00Z",
                },
                400,
            ),
            (
                {
                    "components": [{"repository": "github.com/src-d/go-git", "reference": "xxx"}],
                    "environment": "production",
                    "date_started": "2021-01-12T00:00:00Z",
                    "conclusion": "SUCCESS",
                },
                400,
            ),
            (
                {
                    "components": [{"repository": "github.com/src-d/go-git", "reference": "xxx"}],
                    "environment": "production",
                    "date_started": "2021-01-12T00:00:00Z",
                    "date_finished": "2021-01-12T01:00:00Z",
                    "conclusion": "WHATEVER",
                },
                400,
            ),
            (
                {
                    "components": [{"repository": "github.com/src-d/go-git", "reference": "xxx"}],
                    "environment": "production",
                },
                400,
            ),
            (
                {
                    "components": [],
                    "environment": "production",
                    "date_started": "2021-01-12T00:00:00Z",
                },
                400,
            ),
            (
                {
                    "components": [{"repository": "github.com/src-d/go-git", "reference": "xxx"}],
                    "environment": "production",
                    "date_started": "2021-01-12T00:00:00Z",
                    "date_finished": "2021-01-11T01:00:00Z",
                    "conclusion": "SUCCESS",
                },
                400,
            ),
            (
                {
                    "components": [
                        {"repository": "github.com/athenianco/athenian-api", "reference": "xxx"},
                    ],
                    "environment": "production",
                    "date_started": "2021-01-12T00:00:00Z",
                    "date_finished": "2021-01-12T01:00:00Z",
                    "conclusion": "SUCCESS",
                },
                403,
            ),
            (
                {
                    "components": [{"repository": "github.com", "reference": "xxx"}],
                    "environment": "production",
                    "date_started": "2021-01-12T00:00:00Z",
                    "date_finished": "2021-01-12T01:00:00Z",
                    "conclusion": "SUCCESS",
                },
                400,
            ),
        ],
    )
    async def test_nasty_input(self, token, body, code, disable_default_user):
        await self._request(token, code, json=[body])

    async def test_unauthorized(self) -> None:
        json = [
            {
                "components": [],
                "environment": "production",
                "date_started": "2021-01-12T00:00:00Z",
            },
        ]
        await self._request(None, 401, json=json)

    async def test_422(self, token, sdb, disable_default_user):
        await sdb.execute(sa.delete(AccountGitHubAccount))
        json = [
            {
                "components": [{"repository": "github.com/src-d/go-git", "reference": "xxx"}],
                "environment": "production",
                "date_started": "2021-01-12T00:00:00Z",
                "date_finished": "2021-01-12T00:01:00Z",
                "conclusion": "FAILURE",
            },
        ]
        await self._request(token, 422, json=json)

    async def test_default_user(self, token, sdb):
        json = [
            {
                "components": [{"repository": "github.com/src-d/go-git", "reference": "xxx"}],
                "environment": "production",
                "date_started": "2021-01-12T00:00:00Z",
                "date_finished": "2021-01-12T00:01:00Z",
                "conclusion": "FAILURE",
            },
        ]
        await self._request(token, 403, json=json)

    async def test_date_invalid_timezone_name(self, token, disable_default_user, rdb):
        await _delete_deployments(rdb)
        body = [
            {
                "components": [{"repository": "github.com/src-d/go-git", "reference": "4.2.0"}],
                "environment": "production",
                "date_started": "2021-01-12T10:00:00 MET",
                "date_finished": "2021-01-12T12:00:00 MET",
                "conclusion": "SUCCESS",
            },
        ]
        await self._request(token, 400, json=body)

    async def test_date_with_no_timezone_received(self, token, disable_default_user, rdb):
        await _delete_deployments(rdb)
        body = [
            {
                "components": [{"repository": "github.com/src-d/go-git", "reference": "4.2.0"}],
                "environment": "production",
                "date_started": "2021-01-12T10:00:00Z",
                "date_finished": "2021-01-12T12:00:00",
                "conclusion": "SUCCESS",
            },
        ]
        await self._request(token, 400, json=body)

    async def _request(
        self,
        token: Optional[str],
        assert_status: int = 200,
        **kwargs: Any,
    ) -> dict:
        headers = self.headers.copy()
        if token is not None:
            headers["X-API-Key"] = token
        path = "/v1/events/deployments"
        response = await self.client.request(method="POST", path=path, headers=headers, **kwargs)
        assert response.status == assert_status
        return await response.json()


@pytest.mark.parametrize("unresolved", [False, True])
async def test_resolve_deployed_component_references_smoke(sdb, mdb, rdb, unresolved):
    await _delete_deployments(rdb)

    async def execute_many(sql, values):
        if rdb.url.dialect == "sqlite":
            async with rdb.connection() as rdb_conn:
                async with rdb_conn.transaction():
                    await rdb_conn.execute_many(sql, values)
        else:
            await rdb.execute_many(sql, values)

    await execute_many(
        sa.insert(DeploymentNotification),
        [
            DeploymentNotification(
                account_id=1,
                name="dead",
                started_at=datetime.now(timezone.utc) - timedelta(hours=1),
                finished_at=datetime.now(timezone.utc),
                conclusion="SUCCESS",
                environment="production",
            )
            .create_defaults()
            .explode(with_primary_keys=True),
            DeploymentNotification(
                account_id=1,
                name="alive",
                started_at=datetime.now(timezone.utc) - timedelta(hours=1),
                finished_at=datetime.now(timezone.utc),
                conclusion="SUCCESS",
                environment="production",
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ],
    )
    commit = 2755428
    await execute_many(
        sa.insert(DeployedComponent),
        [
            DeployedComponent(
                account_id=1,
                deployment_name="dead",
                repository_node_id=40550,
                reference="bbb",
                created_at=datetime.now(timezone.utc)
                - (timedelta(hours=3) if unresolved else timedelta(days=14, minutes=1)),
            ).explode(with_primary_keys=True),
            DeployedComponent(
                account_id=1,
                deployment_name="alive",
                repository_node_id=40550,
                reference="ccc",
                resolved_commit_node_id=commit,
                created_at=datetime.now(timezone.utc) - timedelta(days=14, minutes=1),
            ).explode(with_primary_keys=True),
        ],
    )
    await execute_many(
        sa.insert(DeployedLabel),
        [
            DeployedLabel(account_id=1, deployment_name="dead", key="three", value="four")
            .create_defaults()
            .explode(with_primary_keys=True),
            DeployedLabel(account_id=1, deployment_name="alive", key="one", value="two")
            .create_defaults()
            .explode(with_primary_keys=True),
        ],
    )
    await resolve_deployed_component_references(sdb, mdb, rdb, None)
    rows = await rdb.fetch_all(
        sa.select(DeploymentNotification.name).order_by(DeploymentNotification.name),
    )
    assert len(rows) == 1 + unresolved
    assert rows[0][0] == "alive"
    rows = await rdb.fetch_all(
        sa.select(DeployedComponent.deployment_name).order_by(DeployedComponent.deployment_name),
    )
    assert len(rows) == 1 + unresolved
    assert rows[0][0] == "alive"
    rows = await rdb.fetch_all(
        sa.select(DeployedLabel.deployment_name).order_by(DeployedLabel.deployment_name),
    )
    assert len(rows) == 1 + unresolved
    assert rows[0][0] == "alive"


async def _delete_deployments(rdb: Database) -> None:
    await gather(
        rdb.execute(sa.delete(DeployedComponent)),
        rdb.execute(sa.delete(DeployedLabel)),
    )
    await rdb.execute(sa.delete(DeploymentNotification))
