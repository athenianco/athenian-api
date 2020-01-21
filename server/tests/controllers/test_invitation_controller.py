from datetime import datetime, timedelta
import json

import pytest
from sqlalchemy import and_, select

from athenian.api.controllers import invitation_controller
from athenian.api.models.state.models import Invitation


@pytest.fixture(scope="function")
def ikey(request) -> None:
    invitation_controller.ikey = "vadim"

    def nullify():
        invitation_controller.ikey = None

    request.addfinalizer(nullify)


async def test_gen_invitation_new(client, app, ikey):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="GET", path="/v1/invite/generate/1", headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    prefix = invitation_controller.prefix
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


async def test_gen_invitation_no_admin(client):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="GET", path="/v1/invite/generate/2", headers=headers, json={},
    )
    assert response.status == 403


async def test_gen_invitation_no_member(client):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="GET", path="/v1/invite/generate/3", headers=headers, json={},
    )
    assert response.status == 404


async def test_gen_invitation_existing(client, eiso, ikey):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="GET", path="/v1/invite/generate/3", headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    prefix = invitation_controller.prefix
    assert body["url"].startswith(prefix)
    x = body["url"][len(prefix):]
    iid, salt = invitation_controller.decode_slug(x)
    assert iid == 1
    assert salt == 777


async def test_accept_invitation(client, ikey):
    body = {
        "origin": invitation_controller.prefix + invitation_controller.encode_slug(1, 777),
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    del body["updated"]
    assert body == {
        "id": "auth0|5e1f6dfb57bc640ea390557b",
        "name": "Vadim Markovtsev",
        "email": "vadim@athenian.co",
        "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
        "accounts": {"1": True, "2": False, "3": False},
    }


async def test_accept_invitation_noop(client, eiso, ikey):
    body = {
        "origin": invitation_controller.prefix + invitation_controller.encode_slug(1, 777),
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    del body["updated"]
    assert body == {
        "id": "auth0|5e1f6e2e8bfa520ea5290741",
        "name": "Eiso Kant",
        "email": "eiso@athenian.co",
        "picture": "https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
        "accounts": {"1": False, "3": True},
    }


@pytest.mark.parametrize("trash", ["0", "0" * 8, "a" * 8])
async def test_accept_invitation_trash(client, ikey, trash):
    body = {
        "origin": invitation_controller.prefix + "0" * 8,
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == 400
