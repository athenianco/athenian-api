import json
import random

import pytest
from sqlalchemy import insert, select

from athenian.api.controllers.share_controller import _encode_share_id
from athenian.api.models.state.models import Share


@pytest.mark.flaky(reruns=5, reruns_delay=random.uniform(0.5, 2.5))
async def test_share_cycle_smoke(client, headers, disable_default_user, sdb):
    secret = {"key": 777}
    response = await client.request(method="POST", path="/v1/share", headers=headers, json=secret)
    ref = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 200, ref
    rows = await sdb.fetch_all(select([Share]))
    assert len(rows) == 1
    assert not rows[0][Share.divine.name]
    assert ref == "7gsyzgrzminj8"
    response = await client.request(method="GET", path=f"/v1/share/{ref}", headers=headers)
    data = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 200, data
    assert data["author"] == "Vadim Markovtsev"
    assert data["created"]
    assert data["data"] == secret


async def test_share_god(client, headers, disable_default_user, sdb, god):
    secret = {"key": 777}
    response = await client.request(method="POST", path="/v1/share", headers=headers, json=secret)
    ref = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 200, ref
    rows = await sdb.fetch_all(select([Share]))
    assert len(rows) == 1
    assert rows[0][Share.divine.name]


async def test_share_default_user(client, headers):
    secret = {"key": 777}
    response = await client.request(method="POST", path="/v1/share", headers=headers, json=secret)
    ref = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 403, ref


async def test_share_null(client, headers, disable_default_user):
    response = await client.request(method="POST", path="/v1/share", headers=headers, json=None)
    ref = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 400, ref


async def test_share_wrong_user(client, headers, sdb, app):
    await sdb.execute(
        insert(Share).values(
            Share(
                id=10,
                created_by="incognito",
                data={"key": 777},
            )
            .create_defaults()
            .explode(),
        ),
    )
    ref = _encode_share_id(10, app.app["auth"])
    response = await client.request(method="GET", path=f"/v1/share/{ref}", headers=headers)
    data = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 404, data


@pytest.mark.parametrize(
    "ref, code",
    [
        ("7gsyzgrzminj8", 404),
        ("7gsyzgrzminj9", 400),
        ("xxx", 400),
        ("", 405),
    ],
)
async def test_share_nasty_input(client, headers, ref, code):
    response = await client.request(method="GET", path=f"/v1/share/{ref}", headers=headers)
    data = json.loads((await response.read()).decode("utf-8"))
    assert response.status == code, data
