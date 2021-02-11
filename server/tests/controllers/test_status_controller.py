import json

import pytest

from athenian.api import metadata
from athenian.api.models.metadata import __min_version__


async def test_versions(client, headers):
    response = await client.request(
        method="GET", path="/v1/versions", headers=headers, json={},
    )
    assert response.status == 200
    body = json.loads((await response.read()).decode("utf-8"))
    assert body["api"] == metadata.__version__
    assert body["metadata"] == str(__min_version__)


async def test_status(client, headers, client_cache, app):
    for _ in range(2):
        response = await client.request(
            method="GET", path="/status", headers=headers, json={},
        )
        assert response.status == 200
        body = (await response.read()).decode("utf-8")
        assert "requests_total" in body


@pytest.mark.parametrize("limit", ["", "?limit=30"])
async def test_memory(client, headers, limit):
    response = await client.request(
        method="GET", path="/memory" + limit, headers=headers, json={},
    )
    assert response.status == 200
    body = (await response.read()).decode("utf-8")
    assert "total size" in body


async def test_memory_400(client, headers):
    response = await client.request(
        method="GET", path="/memory?limit=x", headers=headers, json={},
    )
    assert response.status == 400


async def test_objgraph(client, headers):
    response = await client.request(
        method="GET", path="/objgraph?type=uvloop.Loop", headers=headers, json={},
    )
    assert response.status == 200
    body = (await response.read()).decode("utf-8")
    assert body.startswith("digraph ObjectGraph")


@pytest.mark.parametrize("query", ["depth=10",
                                   "type=uvloop.Loop&depth=50",
                                   "type=uvloop.Loop&depth=0",
                                   "type=uvloop.Loop&depth=xxx"])
async def test_objgraph_400(client, headers, query):
    response = await client.request(
        method="GET", path="/objgraph?" + query, headers=headers, json={},
    )
    assert response.status == 400
