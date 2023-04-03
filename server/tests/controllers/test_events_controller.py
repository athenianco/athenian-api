from datetime import datetime, timedelta, timezone
import json
from typing import Any, Optional

from freezegun import freeze_time
import pytest
import sqlalchemy as sa
from sqlalchemy import insert, select, update

from athenian.api.controllers.events_controller.deployments import (
    resolve_deployed_component_references,
)
from athenian.api.models.persistentdata.models import (
    DeployedComponent,
    DeployedLabel,
    DeploymentNotification,
    ReleaseNotification,
)
from athenian.api.models.precomputed.models import (
    GitHubDeploymentFacts,
    GitHubDonePullRequestFacts,
    GitHubMergedPullRequestFacts,
    GitHubReleaseFacts,
)
from athenian.api.models.state.models import AccountGitHubAccount, UserToken
from athenian.api.models.web import ReleaseNotificationStatus
from tests.testutils.db import models_insert
from tests.testutils.factory.precomputed import GitHubDonePullRequestFactsFactory
from tests.testutils.requester import Requester


@pytest.fixture(scope="function")
async def token(sdb):
    await sdb.execute(
        insert(UserToken).values(
            UserToken(account_id=1, user_id="auth0|62a1ae88b6bba16c6dbc6870", name="xxx")
            .create_defaults()
            .explode(),
        ),
    )
    return "AQAAAAAAAAA="  # unencrypted


async def test_notify_release_smoke(client, headers, sdb, rdb, token, disable_default_user):
    body = [
        {
            "commit": "8d20cc5",  # 8d20cc5916edf7cfa6a9c5ed069f0640dc823c12
            "repository": "github.com/src-d/go-git",
            "name": "xxx",
            "author": "github.com/yyy",
            "url": "https://google.com",
            "published_at": "2021-01-12T00:00:00Z",
        },
    ]
    headers = headers.copy()
    headers["X-API-Key"] = token
    response = await client.request(
        method="POST", path="/v1/events/releases", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    statuses = json.loads(text)
    assert isinstance(statuses, list)
    assert len(statuses) == 1
    assert statuses[0] == ReleaseNotificationStatus.ACCEPTED_RESOLVED
    rows = await rdb.fetch_all(select(ReleaseNotification))
    assert len(rows) == 1
    columns = dict(rows[0])
    assert columns[ReleaseNotification.created_at.name]
    assert columns[ReleaseNotification.updated_at.name]
    del columns[ReleaseNotification.created_at.name]
    del columns[ReleaseNotification.updated_at.name]
    published_at = datetime(2021, 1, 12, 0, 0)
    if rdb.url.dialect != "sqlite":
        published_at = published_at.replace(tzinfo=timezone.utc)
    assert columns[ReleaseNotification.resolved_at.name]
    del columns[ReleaseNotification.resolved_at.name]
    assert columns == {
        ReleaseNotification.account_id.name: 1,
        ReleaseNotification.repository_node_id.name: 40550,
        ReleaseNotification.commit_hash_prefix.name: "8d20cc5916edf7cfa6a9c5ed069f0640dc823c12",
        ReleaseNotification.resolved_commit_hash.name: "8d20cc5916edf7cfa6a9c5ed069f0640dc823c12",
        ReleaseNotification.resolved_commit_node_id.name: 2756775,  # noqa
        ReleaseNotification.name.name: "xxx",
        ReleaseNotification.author_node_id.name: None,
        ReleaseNotification.url.name: "https://google.com",
        ReleaseNotification.published_at.name: published_at,
        ReleaseNotification.cloned.name: False,
    }

    # check updates and when published_at and author are empty
    for col in ("published_at", "url"):
        del body[0][col]
    body[0]["author"] = "github.com/vmarkovtsev"
    body[0]["commit"] = "8d20cc5916edf7cfa6a9c5ed069f0640dc823c12"
    response = await client.request(
        method="POST", path="/v1/events/releases", headers=headers, json=body,
    )
    assert response.status == 200
    rows = await rdb.fetch_all(select(ReleaseNotification))
    assert len(rows) == 1
    yesterday = datetime.now() - timedelta(days=1)
    if rdb.url.dialect != "sqlite":
        yesterday = published_at.replace(tzinfo=timezone.utc)
    assert rows[0][ReleaseNotification.published_at.name] > yesterday
    assert rows[0][ReleaseNotification.author_node_id.name] == 40020
    assert (
        rows[0][ReleaseNotification.url.name]
        == "https://github.com/src-d/go-git/commit/8d20cc5916edf7cfa6a9c5ed069f0640dc823c12"
    )

    del body[0]["author"]
    response = await client.request(
        method="POST", path="/v1/events/releases", headers=headers, json=body,
    )
    assert response.status == 200
    rows = await rdb.fetch_all(select(ReleaseNotification))
    assert len(rows) == 1
    assert rows[0][ReleaseNotification.author_node_id.name] is None

    await sdb.execute(
        update(UserToken).values(
            {
                UserToken.user_id: "auth0|5e1f6e2e8bfa520ea5290741",
                UserToken.updated_at: datetime.now(timezone.utc),
            },
        ),
    )
    response = await client.request(
        method="POST", path="/v1/events/releases", headers=headers, json=body,
    )
    assert response.status == 200
    rows = await rdb.fetch_all(select(ReleaseNotification))
    assert len(rows) == 1
    assert rows[0][ReleaseNotification.author_node_id.name] == 39936


async def test_notify_release_url(client, headers, sdb, rdb, token, disable_default_user):
    body = [
        {
            "commit": "8d20cc5",  # 8d20cc5916edf7cfa6a9c5ed069f0640dc823c12
            "repository": "github.com/src-d/go-git",
            "name": "xxx",
            "author": "github.com/yyy",
            "published_at": "2021-01-12T00:00:00Z",
        },
    ]
    headers = headers.copy()
    headers["X-API-Key"] = token
    response = await client.request(
        method="POST", path="/v1/events/releases", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    rows = await rdb.fetch_all(select(ReleaseNotification))
    assert len(rows) == 1
    assert (
        rows[0][ReleaseNotification.url.name]
        == "https://github.com/src-d/go-git/commit/8d20cc5916edf7cfa6a9c5ed069f0640dc823c12"
    )


@pytest.mark.parametrize(
    "status, body, statuses",
    [
        # duplicates
        (
            200,
            [
                {
                    "commit": "abcdef0",
                    "repository": "github.com/src-d/go-git",
                },
                {
                    "commit": "abcdef0",
                    "repository": "github.com/src-d/go-git",
                },
            ],
            [
                ReleaseNotificationStatus.ACCEPTED_PENDING,
                ReleaseNotificationStatus.IGNORED_DUPLICATE,
            ],
        ),
        # duplicates
        (
            200,
            [
                {
                    "name": "xxx",
                    "commit": "abcdef0",
                    "repository": "github.com/src-d/go-git",
                },
                {
                    "name": "xxx",
                    "commit": "abcdef1",
                    "repository": "github.com/src-d/go-git",
                },
            ],
            [
                ReleaseNotificationStatus.ACCEPTED_PENDING,
                ReleaseNotificationStatus.IGNORED_DUPLICATE,
            ],
        ),
        # not duplicates
        (
            200,
            [
                {
                    "commit": "abcdef0",
                    "repository": "github.com/src-d/go-git",
                },
                {
                    "name": "xxx",
                    "commit": "abcdef0",
                    "repository": "github.com/src-d/go-git",
                },
            ],
            [
                ReleaseNotificationStatus.ACCEPTED_PENDING,
                ReleaseNotificationStatus.ACCEPTED_PENDING,
            ],
        ),
        # wrong hash
        (
            400,
            [
                {
                    "commit": "abcdef01",
                    "repository": "github.com/src-d/go-git",
                },
            ],
            [],
        ),
        # denied repo
        (
            403,
            [
                {
                    "commit": "abcdef0",
                    "repository": "github.com/athenianco/metadata",
                },
            ],
            [],
        ),
        # bad date
        (
            400,
            [
                {
                    "commit": "abcdef0",
                    "repository": "github.com/src-d/go-git",
                    "published_at": "date",
                },
            ],
            [],
        ),
    ],
)
async def test_notify_release_nasty_input(
    client,
    headers,
    token,
    rdb,
    body,
    status,
    statuses,
    disable_default_user,
):
    headers = headers.copy()
    headers["X-API-Key"] = token
    response = await client.request(
        method="POST", path="/v1/events/releases", headers=headers, json=body,
    )
    assert response.status == status
    if status == 200:
        assert json.loads((await response.read()).decode("utf-8")) == statuses
        rows = await rdb.fetch_all(select(ReleaseNotification))
        assert len(rows) == 1 + (
            statuses[0] == statuses[1] == ReleaseNotificationStatus.ACCEPTED_PENDING
        )
        assert rows[0][ReleaseNotification.commit_hash_prefix.name] == body[0]["commit"]
        assert rows[0][ReleaseNotification.resolved_commit_hash.name] is None
        assert rows[0][ReleaseNotification.resolved_commit_node_id.name] is None


async def test_notify_release_422(client, headers, sdb, disable_default_user):
    body = [
        {
            "commit": "8d20cc5",  # 8d20cc5916edf7cfa6a9c5ed069f0640dc823c12
            "repository": "github.com/src-d/go-git",
        },
    ]
    await sdb.execute(
        insert(UserToken).values(
            UserToken(account_id=2, user_id="auth0|62a1ae88b6bba16c6dbc6870", name="xxx")
            .create_defaults()
            .explode(),
        ),
    )
    headers = headers.copy()
    headers["X-API-Key"] = "AQAAAAAAAAA="  # unencrypted
    response = await client.request(
        method="POST", path="/v1/events/releases", headers=headers, json=body,
    )
    assert response.status == 422


async def test_notify_release_default_user(client, headers, sdb):
    body = [
        {
            "commit": "8d20cc5",  # 8d20cc5916edf7cfa6a9c5ed069f0640dc823c12
            "repository": "github.com/src-d/go-git",
        },
    ]
    await sdb.execute(
        insert(UserToken).values(
            UserToken(account_id=2, user_id="auth0|62a1ae88b6bba16c6dbc6870", name="xxx")
            .create_defaults()
            .explode(),
        ),
    )
    headers = headers.copy()
    headers["X-API-Key"] = "AQAAAAAAAAA="  # unencrypted
    response = await client.request(
        method="POST", path="/v1/events/releases", headers=headers, json=body,
    )
    assert response.status == 403


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
            assert len(await pdb.fetch_all(select(table))) == n, table

    @freeze_time("2020-01-01")
    async def test_clear_deployments(self, pdb, disable_default_user):
        await pdb.execute(
            insert(GitHubDeploymentFacts).values(
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
        rows = await pdb.fetch_all(select(GitHubDeploymentFacts))
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
        await rdb.execute(sa.delete(DeploymentNotification))
        await rdb.execute(sa.delete(DeployedComponent))
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

        rows = await rdb.fetch_all(select(DeployedComponent))
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
        rows = await rdb.fetch_all(select(DeployedLabel))
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
        rows = await rdb.fetch_all(select(DeploymentNotification))
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
        await rdb.execute(sa.delete(DeploymentNotification))
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
        await rdb.execute(sa.delete(DeploymentNotification))
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
    await rdb.execute(sa.delete(DeployedComponent))
    await rdb.execute(sa.delete(DeploymentNotification))

    async def execute_many(sql, values):
        if rdb.url.dialect == "sqlite":
            async with rdb.connection() as rdb_conn:
                async with rdb_conn.transaction():
                    await rdb_conn.execute_many(sql, values)
        else:
            await rdb.execute_many(sql, values)

    await execute_many(
        insert(DeploymentNotification),
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
        insert(DeployedComponent),
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
        insert(DeployedLabel),
        [
            DeployedLabel(
                account_id=1,
                deployment_name="dead",
                key="three",
                value="four",
            )
            .create_defaults()
            .explode(with_primary_keys=True),
            DeployedLabel(
                account_id=1,
                deployment_name="alive",
                key="one",
                value="two",
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ],
    )
    await resolve_deployed_component_references(sdb, mdb, rdb, None)
    rows = await rdb.fetch_all(
        select(DeploymentNotification.name).order_by(DeploymentNotification.name),
    )
    assert len(rows) == 1 + unresolved
    assert rows[0][0] == "alive"
    rows = await rdb.fetch_all(
        select(DeployedComponent.deployment_name).order_by(DeployedComponent.deployment_name),
    )
    assert len(rows) == 1 + unresolved
    assert rows[0][0] == "alive"
    rows = await rdb.fetch_all(
        select(DeployedLabel.deployment_name).order_by(DeployedLabel.deployment_name),
    )
    assert len(rows) == 1 + unresolved
    assert rows[0][0] == "alive"
