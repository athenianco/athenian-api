from datetime import datetime, timedelta, timezone
import json

import pytest
import sqlalchemy as sa

from athenian.api.models.persistentdata.models import ReleaseNotification
from athenian.api.models.state.models import UserToken
from athenian.api.models.web import ReleaseNotificationStatus


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
    rows = await rdb.fetch_all(sa.select(ReleaseNotification))
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
    rows = await rdb.fetch_all(sa.select(ReleaseNotification))
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
    rows = await rdb.fetch_all(sa.select(ReleaseNotification))
    assert len(rows) == 1
    assert rows[0][ReleaseNotification.author_node_id.name] is None

    await sdb.execute(
        sa.update(UserToken).values(
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
    rows = await rdb.fetch_all(sa.select(ReleaseNotification))
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
    rows = await rdb.fetch_all(sa.select(ReleaseNotification))
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
                {"commit": "abcdef0", "repository": "github.com/src-d/go-git"},
                {"commit": "abcdef0", "repository": "github.com/src-d/go-git"},
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
                {"name": "xxx", "commit": "abcdef0", "repository": "github.com/src-d/go-git"},
                {"name": "xxx", "commit": "abcdef1", "repository": "github.com/src-d/go-git"},
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
                {"commit": "abcdef0", "repository": "github.com/src-d/go-git"},
                {"name": "xxx", "commit": "abcdef0", "repository": "github.com/src-d/go-git"},
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
                {"commit": "abcdef01", "repository": "github.com/src-d/go-git"},
            ],
            [],
        ),
        # denied repo
        (
            403,
            [
                {"commit": "abcdef0", "repository": "github.com/athenianco/metadata"},
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
        rows = await rdb.fetch_all(sa.select(ReleaseNotification))
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
        sa.insert(UserToken).values(
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
        sa.insert(UserToken).values(
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
