from datetime import datetime, timedelta, timezone

from freezegun import freeze_time
import pytest
from sqlalchemy import delete, insert, select, update

from athenian.api.controllers.events_controller import resolve_deployed_component_references
from athenian.api.models.persistentdata.models import DeployedComponent, DeployedLabel, \
    DeploymentNotification, \
    ReleaseNotification
from athenian.api.models.state.models import AccountGitHubAccount, UserToken
from athenian.precomputer.db.models import GitHubDonePullRequestFacts, \
    GitHubMergedPullRequestFacts, GitHubReleaseFacts


@pytest.fixture(scope="function")
async def token(sdb):
    await sdb.execute(insert(UserToken).values(UserToken(
        account_id=1, user_id="auth0|5e1f6dfb57bc640ea390557b", name="xxx",
    ).create_defaults().explode()))
    return "AQAAAAAAAAA="  # unencrypted


async def test_notify_release_smoke(client, headers, sdb, rdb, token):
    body = [{
        "commit": "8d20cc5",  # 8d20cc5916edf7cfa6a9c5ed069f0640dc823c12
        "repository": "github.com/src-d/go-git",
        "name": "xxx",
        "author": "github.com/yyy",
        "url": "https://google.com",
        "published_at": "2021-01-12T00:00:00Z",
    }]
    headers = headers.copy()
    headers["X-API-Key"] = token
    response = await client.request(
        method="POST", path="/v1/events/releases", headers=headers, json=body,
    )
    assert response.status == 200
    assert (await response.read()).decode("utf-8") == ""
    rows = await rdb.fetch_all(select([ReleaseNotification]))
    assert len(rows) == 1
    columns = dict(rows[0])
    assert columns[ReleaseNotification.created_at.key]
    assert columns[ReleaseNotification.updated_at.key]
    del columns[ReleaseNotification.created_at.key]
    del columns[ReleaseNotification.updated_at.key]
    published_at = datetime(2021, 1, 12, 0, 0)
    if rdb.url.dialect != "sqlite":
        published_at = published_at.replace(tzinfo=timezone.utc)
    assert columns == {
        ReleaseNotification.account_id.key: 1,
        ReleaseNotification.repository_node_id.key: "MDEwOlJlcG9zaXRvcnk0NDczOTA0NA==",
        ReleaseNotification.commit_hash_prefix.key: "8d20cc5916edf7cfa6a9c5ed069f0640dc823c12",
        ReleaseNotification.resolved_commit_hash.key: "8d20cc5916edf7cfa6a9c5ed069f0640dc823c12",
        ReleaseNotification.resolved_commit_node_id.key: "MDY6Q29tbWl0NDQ3MzkwNDQ6OGQyMGNjNTkxNmVkZjdjZmE2YTljNWVkMDY5ZjA2NDBkYzgyM2MxMg==",  # noqa
        ReleaseNotification.name.key: "xxx",
        ReleaseNotification.author_node_id.key: None,
        ReleaseNotification.url.key: "https://google.com",
        ReleaseNotification.published_at.key: published_at,
        ReleaseNotification.cloned.key: False,
    }

    # check updates and when published_at and author are empty
    del body[0]["published_at"]
    body[0]["author"] = "github.com/vmarkovtsev"
    body[0]["commit"] = "8d20cc5916edf7cfa6a9c5ed069f0640dc823c12"
    response = await client.request(
        method="POST", path="/v1/events/releases", headers=headers, json=body,
    )
    assert response.status == 200
    rows = await rdb.fetch_all(select([ReleaseNotification]))
    assert len(rows) == 1
    yesterday = datetime.now() - timedelta(days=1)
    if rdb.url.dialect != "sqlite":
        yesterday = published_at.replace(tzinfo=timezone.utc)
    assert rows[0][ReleaseNotification.published_at.key] > yesterday
    assert rows[0][ReleaseNotification.author_node_id.key] == "MDQ6VXNlcjI3OTM1NTE="

    del body[0]["author"]
    response = await client.request(
        method="POST", path="/v1/events/releases", headers=headers, json=body,
    )
    assert response.status == 200
    rows = await rdb.fetch_all(select([ReleaseNotification]))
    assert len(rows) == 1
    assert rows[0][ReleaseNotification.author_node_id.key] is None

    await sdb.execute(update(UserToken).values({
        UserToken.user_id: "auth0|5e1f6e2e8bfa520ea5290741",
        UserToken.updated_at: datetime.now(timezone.utc),
    }))
    response = await client.request(
        method="POST", path="/v1/events/releases", headers=headers, json=body,
    )
    assert response.status == 200
    rows = await rdb.fetch_all(select([ReleaseNotification]))
    assert len(rows) == 1
    assert rows[0][ReleaseNotification.author_node_id.key] == "MDQ6VXNlcjEyNDc2MDg="


@pytest.mark.parametrize("status, body", [
    # duplicates
    (400, [
        {
            "commit": "abcdef0",
            "repository": "github.com/src-d/go-git",
        },
        {
            "commit": "abcdef0",
            "repository": "github.com/src-d/go-git",
        },
    ]),
    # wrong hash
    (400, [
        {
            "commit": "abcdef01",
            "repository": "github.com/src-d/go-git",
        },
    ]),
    # denied repo
    (403, [
        {
            "commit": "abcdef0",
            "repository": "github.com/athenianco/metadata",
        },
    ]),
    # wrong hash, but we register it
    (200, [
        {
            "commit": "abcdef0",
            "repository": "github.com/src-d/go-git",
        },
    ]),
    # bad date
    (400, [
        {
            "commit": "abcdef0",
            "repository": "github.com/src-d/go-git",
            "published_at": "date",
        },
    ]),
])
async def test_notify_release_nasty_input(client, headers, token, rdb, body, status):
    headers = headers.copy()
    headers["X-API-Key"] = token
    response = await client.request(
        method="POST", path="/v1/events/releases", headers=headers, json=body,
    )
    assert response.status == status
    if status == 200:
        rows = await rdb.fetch_all(select([ReleaseNotification]))
        assert len(rows) == 1
        assert rows[0][ReleaseNotification.commit_hash_prefix.key] == body[0]["commit"]
        assert rows[0][ReleaseNotification.resolved_commit_hash.key] is None
        assert rows[0][ReleaseNotification.resolved_commit_node_id.key] is None


async def test_notify_release_422(client, headers, sdb):
    body = [{
        "commit": "8d20cc5",  # 8d20cc5916edf7cfa6a9c5ed069f0640dc823c12
        "repository": "github.com/src-d/go-git",
    }]
    await sdb.execute(insert(UserToken).values(UserToken(
        account_id=2, user_id="auth0|5e1f6dfb57bc640ea390557b", name="xxx",
    ).create_defaults().explode()))
    headers = headers.copy()
    headers["X-API-Key"] = "AQAAAAAAAAA="  # unencrypted
    response = await client.request(
        method="POST", path="/v1/events/releases", headers=headers, json=body,
    )
    assert response.status == 422


@freeze_time("2020-01-01")
async def test_clear_precomputed_events_smoke(client, headers, pdb):
    await pdb.execute(insert(GitHubDonePullRequestFacts).values(GitHubDonePullRequestFacts(
        acc_id=1,
        release_match="event",
        pr_node_id="xxx",
        repository_full_name="src-d/go-git",
        pr_created_at=datetime.now(timezone.utc),
        pr_done_at=datetime.now(timezone.utc),
        number=1,
        reviewers={},
        commenters={},
        commit_authors={},
        commit_committers={},
        labels={},
        activity_days=[],
        data=b"data",
    ).create_defaults().explode(with_primary_keys=True)))
    await pdb.execute(insert(GitHubMergedPullRequestFacts).values(GitHubMergedPullRequestFacts(
        acc_id=1,
        release_match="event",
        pr_node_id="yyy",
        repository_full_name="src-d/go-git",
        merged_at=datetime.now(timezone.utc),
        checked_until=datetime.now(timezone.utc),
        labels={},
        activity_days=[],
    ).create_defaults().explode(with_primary_keys=True)))
    await pdb.execute(insert(GitHubReleaseFacts).values(GitHubReleaseFacts(
        acc_id=1,
        id="zzz",
        release_match="event",
        repository_full_name="src-d/go-git",
        published_at=datetime.now(timezone.utc),
        data=b"data",
    ).create_defaults().explode(with_primary_keys=True)))
    body = {
        "account": 1,
        "repositories": ["{1}"],
        "targets": ["release"],
    }
    response = await client.request(
        method="POST", path="/v1/events/clear_cache", headers=headers, json=body,
    )
    assert response.status == 200
    for table, n in ((GitHubDonePullRequestFacts, 292),
                     (GitHubMergedPullRequestFacts, 245),
                     (GitHubReleaseFacts, 53)):
        assert len(await pdb.fetch_all(select([table]))) == n, table


@pytest.mark.parametrize("status, body", [
    (200, {"account": 1, "repositories": ["github.com/src-d/go-git"], "targets": []}),
    (400, {"account": 1, "repositories": ["github.com/src-d/go-git"], "targets": ["wrong"]}),
    (422, {"account": 2, "repositories": ["github.com/src-d/go-git"], "targets": []}),
    (404, {"account": 3, "repositories": ["github.com/src-d/go-git"], "targets": []}),
    (403, {"account": 1, "repositories": ["github.com/athenianco/athenian-api"], "targets": []}),
])
async def test_clear_precomputed_events_nasty_input(client, headers, body, status):
    response = await client.request(
        method="POST", path="/v1/events/clear_cache", headers=headers, json=body,
    )
    assert response.status == status


@pytest.mark.parametrize("ref, vhash", [
    ("4.2.0", "y9c5A0Df"),
    ("v4.2.0", "y9c5A0Df"),
    ("1d28459504251497e0ce6132a0fadd5eb44ffd22", "Cd34s0Jb"),
    ("1d28459", "Cd34s0Jb"),
    ("xxx", "u5VWde@k"),
])
async def test_notify_deployment_smoke(client, headers, token, rdb, ref, vhash):
    body = [{
        "components": [{
            "repository": "github.com/src-d/go-git",
            "reference": ref,
        }],
        "environment": "production",
        "date_started": "2021-01-12T00:00:00Z",
        "date_finished": "2021-01-12T01:00:00Z",
        "conclusion": "SUCCESS",
        "labels": {
            "one": 1,
            2: "two",
        },
    }]
    headers = headers.copy()
    headers["X-API-Key"] = token
    response = await client.request(
        method="POST", path="/v1/events/deployments", headers=headers, json=body,
    )
    assert response.status == 200
    assert (await response.read()).decode("utf-8") == ""
    rows = await rdb.fetch_all(select([DeployedComponent]))
    assert len(rows) == 1
    row = dict(rows[0])
    created_at = row[DeployedComponent.created_at.key]
    assert isinstance(created_at, datetime)
    del row[DeployedComponent.created_at.key]
    if ref == "xxx":
        commit = None
    else:
        commit = "MDY6Q29tbWl0NDQ3MzkwNDQ6MWQyODQ1OTUwNDI1MTQ5N2UwY2U2MTMyYTBmYWRkNWViNDRmZmQyMg=="
    name = f"prod-2021-01-12-{vhash}"
    assert row == {
        "account_id": 1,
        "deployment_name": name,
        "repository_node_id": "MDEwOlJlcG9zaXRvcnk0NDczOTA0NA==",
        "reference": ref,
        "resolved_commit_node_id": commit,
    }
    rows = await rdb.fetch_all(select([DeployedLabel]))
    assert len(rows) == 2
    assert dict(rows[0]) == {
        "account_id": 1,
        "deployment_name": name,
        "key": "one",
        "value": 1,
    }
    assert dict(rows[1]) == {
        "account_id": 1,
        "deployment_name": name,
        "key": "2",
        "value": "two",
    }
    rows = await rdb.fetch_all(select([DeploymentNotification]))
    assert len(rows) == 1
    row = dict(rows[0])
    created_at = row[DeploymentNotification.created_at.key]
    assert isinstance(created_at, datetime)
    del row[DeploymentNotification.created_at.key]
    updated_at = row[DeploymentNotification.updated_at.key]
    assert isinstance(updated_at, datetime)
    del row[DeploymentNotification.updated_at.key]
    tzinfo = timezone.utc if rdb.url.dialect == "postgresql" else None
    assert row == {
        "account_id": 1,
        "name": name,
        "conclusion": "SUCCESS",
        "started_at": datetime(2021, 1, 12, 0, 0, tzinfo=tzinfo),
        "finished_at": datetime(2021, 1, 12, 1, 0, tzinfo=tzinfo),
        "url": None,
        "environment": "production",
    }


async def test_notify_deployment_duplicate(client, headers, token):
    body = [{
        "components": [{
            "repository": "github.com/src-d/go-git",
            "reference": "xxx",
        }],
        "environment": "staging",
        "date_started": "2021-01-12T00:00:00Z",
        "date_finished": "2021-01-12T00:01:00Z",
        "conclusion": "FAILURE",
        "labels": {
            "one": 1,
        },
    }]
    headers = headers.copy()
    headers["X-API-Key"] = token
    response = await client.request(
        method="POST", path="/v1/events/deployments", headers=headers, json=body,
    )
    assert response.status == 200
    response = await client.request(
        method="POST", path="/v1/events/deployments", headers=headers, json=body,
    )
    assert response.status == 409


@pytest.mark.parametrize("body, code", [
    ({
        "components": [{
            "repository": "github.com/src-d/go-git",
            "reference": "xxx",
        }],
        "environment": "production",
        "date_started": "2021-01-12T00:00:00Z",
        "date_finished": "2021-01-12T01:00:00Z",
    }, 400),
    ({
        "components": [{
            "repository": "github.com/src-d/go-git",
            "reference": "xxx",
        }],
        "environment": "production",
        "date_started": "2021-01-12T00:00:00Z",
        "conclusion": "SUCCESS",
    }, 400),
    ({
        "components": [{
            "repository": "github.com/src-d/go-git",
            "reference": "xxx",
        }],
        "environment": "production",
        "date_started": "2021-01-12T00:00:00Z",
        "date_finished": "2021-01-12T01:00:00Z",
        "conclusion": "WHATEVER",
    }, 400),
    ({
        "components": [{
            "repository": "github.com/src-d/go-git",
            "reference": "xxx",
        }],
        "environment": "production",
    }, 400),
    ({
        "components": [],
        "environment": "production",
        "date_started": "2021-01-12T00:00:00Z",
    }, 400),
    ({
        "components": [{
            "repository": "github.com/src-d/go-git",
            "reference": "xxx",
        }],
        "environment": "production",
        "date_started": "2021-01-12T00:00:00Z",
        "date_finished": "2021-01-11T01:00:00Z",
        "conclusion": "SUCCESS",
    }, 400),
    ({
        "components": [{
            "repository": "github.com/athenianco/athenian-api",
            "reference": "xxx",
        }],
        "environment": "production",
        "date_started": "2021-01-12T00:00:00Z",
        "date_finished": "2021-01-12T01:00:00Z",
        "conclusion": "SUCCESS",
    }, 403),
    ({
        "components": [{
            "repository": "github.com",
            "reference": "xxx",
        }],
        "environment": "production",
        "date_started": "2021-01-12T00:00:00Z",
        "date_finished": "2021-01-12T01:00:00Z",
        "conclusion": "SUCCESS",
    }, 400),
])
async def test_notify_deployment_nasty_input(client, headers, token, body, code):
    headers = headers.copy()
    headers["X-API-Key"] = token
    response = await client.request(
        method="POST", path="/v1/events/deployments", headers=headers, json=[body],
    )
    assert response.status == code


async def test_notify_deployment_unauthorized(client, headers):
    response = await client.request(
        method="POST", path="/v1/events/deployments", headers=headers, json=[{
            "components": [],
            "environment": "production",
            "date_started": "2021-01-12T00:00:00Z",
        }],
    )
    assert response.status == 401


async def test_notify_deployment_422(client, headers, token, sdb):
    await sdb.execute(delete(AccountGitHubAccount))
    headers = headers.copy()
    headers["X-API-Key"] = token
    response = await client.request(
        method="POST", path="/v1/events/deployments", headers=headers, json=[{
            "components": [{
                "repository": "github.com/src-d/go-git",
                "reference": "xxx",
            }],
            "environment": "production",
            "date_started": "2021-01-12T00:00:00Z",
            "date_finished": "2021-01-12T00:01:00Z",
            "conclusion": "FAILURE",
        }],
    )
    assert response.status == 422


async def test_resolve_deployed_component_references_smoke(sdb, mdb, rdb):
    async def execute_many(sql, values):
        if rdb.url.dialect == "sqlite":
            async with rdb.connection() as rdb_conn:
                async with rdb_conn.transaction():
                    await rdb_conn.execute_many(sql, values)
        else:
            await rdb.execute_many(sql, values)

    await execute_many(insert(DeploymentNotification), [
        DeploymentNotification(
            account_id=1,
            name="dead",
            started_at=datetime.now(timezone.utc) - timedelta(hours=1),
            finished_at=datetime.now(timezone.utc),
            conclusion="SUCCESS",
            environment="production",
        ).create_defaults().explode(with_primary_keys=True),
        DeploymentNotification(
            account_id=1,
            name="alive",
            started_at=datetime.now(timezone.utc) - timedelta(hours=1),
            finished_at=datetime.now(timezone.utc),
            conclusion="SUCCESS",
            environment="production",
        ).create_defaults().explode(with_primary_keys=True),
    ])
    commit = "MDY6Q29tbWl0NDQ3MzkwNDQ6MWQyODQ1OTUwNDI1MTQ5N2UwY2U2MTMyYTBmYWRkNWViNDRmZmQyMg=="
    await execute_many(insert(DeployedComponent), [
        DeployedComponent(
            account_id=1,
            deployment_name="dead",
            repository_node_id="MDEwOlJlcG9zaXRvcnk0NDczOTA0NA==",
            reference="bbb",
            created_at=datetime.now(timezone.utc) - timedelta(days=2),
        ).explode(with_primary_keys=True),
        DeployedComponent(
            account_id=1,
            deployment_name="alive",
            repository_node_id="MDEwOlJlcG9zaXRvcnk0NDczOTA0NA==",
            reference="ccc",
            resolved_commit_node_id=commit,
            created_at=datetime.now(timezone.utc) - timedelta(days=2),
        ).explode(with_primary_keys=True),
    ])
    await execute_many(insert(DeployedLabel), [
        DeployedLabel(
            account_id=1,
            deployment_name="dead",
            key="three",
            value="four",
        ).create_defaults().explode(with_primary_keys=True),
        DeployedLabel(
            account_id=1,
            deployment_name="alive",
            key="one",
            value="two",
        ).create_defaults().explode(with_primary_keys=True),
    ])
    await resolve_deployed_component_references(sdb, mdb, rdb, None)
    rows = await rdb.fetch_all(select([DeploymentNotification.name]))
    assert len(rows) == 1
    assert rows[0][0] == "alive"
    rows = await rdb.fetch_all(select([DeployedComponent.deployment_name]))
    assert len(rows) == 1
    assert rows[0][0] == "alive"
    rows = await rdb.fetch_all(select([DeployedLabel.deployment_name]))
    assert len(rows) == 1
    assert rows[0][0] == "alive"
