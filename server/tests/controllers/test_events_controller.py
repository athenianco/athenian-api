from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import insert, select

from athenian.api.models.persistentdata.models import ReleaseNotification
from athenian.api.models.state.models import UserToken


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
        ReleaseNotification.author.key: "yyy",
        ReleaseNotification.url.key: "https://google.com",
        ReleaseNotification.published_at.key: published_at,
        ReleaseNotification.cloned.key: False,
    }

    # check updates and when published_at and author are empty
    del body[0]["published_at"]
    del body[0]["author"]
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
    assert rows[0][ReleaseNotification.author.key] == "vadim"


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
