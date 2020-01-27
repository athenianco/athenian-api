from datetime import datetime, timedelta
import json
from random import randint

import pytest
from sqlalchemy import and_, insert, select, update

from athenian.api.controllers import invitation_controller
from athenian.api.models.state.models import Invitation


async def test_gen_invitation_new(client, app, headers):
    response = await client.request(
        method="GET", path="/v1/invite/generate/1", headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    prefix = invitation_controller.url_prefix
    assert body["url"].startswith(prefix)
    x = body["url"][len(prefix):]
    iid, salt = invitation_controller.decode_slug(x)
    inv = await app.sdb.fetch_one(
        select([Invitation])
        .where(and_(Invitation.id == iid, Invitation.salt == salt)))
    assert inv is not None
    assert inv[Invitation.is_active.key]
    assert inv[Invitation.accepted.key] == 0
    assert inv[Invitation.account_id.key] == 1
    assert inv[Invitation.created_by.key] == "auth0|5e1f6dfb57bc640ea390557b"
    assert inv[Invitation.created_at.key] > datetime.utcnow() - timedelta(minutes=1)


async def test_gen_invitation_no_admin(client, headers):
    response = await client.request(
        method="GET", path="/v1/invite/generate/2", headers=headers, json={},
    )
    assert response.status == 403


async def test_gen_invitation_no_member(client, headers):
    response = await client.request(
        method="GET", path="/v1/invite/generate/3", headers=headers, json={},
    )
    assert response.status == 404


async def test_gen_invitation_existing(client, eiso, headers):
    response = await client.request(
        method="GET", path="/v1/invite/generate/3", headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    prefix = invitation_controller.url_prefix
    assert body["url"].startswith(prefix)
    x = body["url"][len(prefix):]
    iid, salt = invitation_controller.decode_slug(x)
    assert iid == 1
    assert salt == 777


async def test_accept_invitation(client, headers):
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(1, 777),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    del body["user"]["updated"]
    assert body == {
        "account": 3,
        "user": {
            "id": "auth0|5e1f6dfb57bc640ea390557b",
            "name": "Vadim Markovtsev",
            "native_id": "5e1f6dfb57bc640ea390557b",
            "email": "vadim@athenian.co",
            "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
            "accounts": {"1": True, "2": False, "3": False},
        },
    }


async def test_accept_invitation_noop(client, eiso, headers):
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(1, 777),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    del body["user"]["updated"]
    assert body == {
        "account": 3,
        "user": {
            "id": "auth0|5e1f6e2e8bfa520ea5290741",
            "name": "Eiso Kant",
            "native_id": "5e1f6e2e8bfa520ea5290741",
            "email": "eiso@athenian.co",
            "picture": "https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
            "accounts": {"1": False, "3": True},
        },
    }


@pytest.mark.parametrize("trash", ["0", "0" * 8, "a" * 8])
async def test_accept_invitation_trash(client, trash, headers):
    body = {
        "url": invitation_controller.url_prefix + "0" * 8,
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == 400


async def test_accept_invitation_inactive(client, app, headers):
    await app.sdb.execute(
        update(Invitation).where(Invitation.id == 1).values({Invitation.is_active.key: False}))
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(1, 777),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == 403


async def test_accept_invitation_admin(client, app, headers):
    iid = await app.sdb.execute(
        insert(Invitation).values(
            Invitation(salt=888, account_id=invitation_controller.admin_backdoor)
            .create_defaults().explode()))
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(iid, 888),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    del body["user"]["updated"]
    assert body == {
        "account": 4,
        "user": {
            "id": "auth0|5e1f6dfb57bc640ea390557b",
            "name": "Vadim Markovtsev",
            "native_id": "5e1f6dfb57bc640ea390557b",
            "email": "vadim@athenian.co",
            "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png", # noqa
            "accounts": {"1": True, "2": False, "4": True},
        },
    }


async def test_check_invitation(client, headers):
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(1, 777),
    }
    response = await client.request(
        method="POST", path="/v1/invite/check", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == {"valid": True, "active": True, "type": "regular"}


async def test_check_invitation_not_exists(client, headers):
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(1, 888),
    }
    response = await client.request(
        method="POST", path="/v1/invite/check", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == {"valid": False}


async def test_check_invitation_admin(client, app, headers):
    iid = await app.sdb.execute(
        insert(Invitation).values(
            Invitation(salt=888, account_id=invitation_controller.admin_backdoor)
            .create_defaults().explode()))
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(iid, 888),
    }
    response = await client.request(
        method="POST", path="/v1/invite/check", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == {"valid": True, "active": True, "type": "admin"}


async def test_check_invitation_inactive(client, app, headers):
    await app.sdb.execute(
        update(Invitation).where(Invitation.id == 1).values({Invitation.is_active.key: False}))
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(1, 777),
    }
    response = await client.request(
        method="POST", path="/v1/invite/check", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == {"valid": True, "active": False, "type": "regular"}


async def test_check_invitation_malformed(client, headers):
    body = {
        "url": "https://athenian.co",
    }
    response = await client.request(
        method="POST", path="/v1/invite/check", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == {"valid": False}


def test_encode_decode():
    for _ in range(1000):
        iid = randint(0, invitation_controller.admin_backdoor)
        salt = randint(0, (1 << 16) - 1)
        try:
            iid_back, salt_back = invitation_controller.decode_slug(
                invitation_controller.encode_slug(iid, salt))
        except Exception as e:
            print(iid, salt)
            raise e from None
        assert iid_back == iid
        assert salt_back == salt


async def test_progress_200(client, headers, app, cache):
    app._cache = cache
    true_body = {
        "owner": "676724",
        "repositories": 19,
        "tables": [
            {"fetched": 30, "name": "Commit", "total": 50},
        ],
    }
    for _ in range(2):
        response = await client.request(
            method="GET", path="/v1/invite/progress/1", headers=headers, json={},
        )
        assert response.status == 200
        body = json.loads((await response.read()).decode("utf-8"))
        sd = body["started_date"]
        assert sd
        del body["started_date"]
        assert body == true_body


@pytest.mark.parametrize("account, code", [(2, 422), (3, 404)])
async def test_progress_errors(client, headers, account, code):
    response = await client.request(
        method="GET", path="/v1/invite/progress/%d" % account, headers=headers, json={},
    )
    assert response.status == code
