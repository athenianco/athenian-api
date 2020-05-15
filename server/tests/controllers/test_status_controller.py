import json

from athenian.api import metadata


async def test_versions(client, headers):
    response = await client.request(
        method="GET", path="/v1/versions", headers=headers, json={},
    )
    assert response.status == 200
    body = json.loads((await response.read()).decode("utf-8"))
    assert body["api"] == metadata.__version__


async def test_status(client, headers):
    response = await client.request(
        method="GET", path="/status", headers=headers, json={},
    )
    assert response.status == 200
    body = (await response.read()).decode("utf-8")
    assert "requests_total" in body
