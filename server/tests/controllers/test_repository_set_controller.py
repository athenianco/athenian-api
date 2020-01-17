import json

import pytest
from sqlalchemy import select

from athenian.api.models.state.models import RepositorySet
from athenian.api.models.web.repository_set_create_request import RepositorySetCreateRequest


async def test_delete_repository_set(client, app):
    body = {}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="DELETE", path="/v1/reposet/1", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    rs = await app.sdb.fetch_one(select([RepositorySet]).where(RepositorySet.id == 1))
    assert rs is None


async def test_delete_repository_set_404(client, app):
    body = {}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="DELETE", path="/v1/reposet/10", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 404, "Response body is : " + body


@pytest.mark.parametrize("reposet", [2, 3])
async def test_delete_repository_set_bad_team(client, reposet):
    body = {}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="DELETE", path="/v1/reposet/%d" % reposet, headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


@pytest.mark.parametrize("reposet", [1, 2])
async def test_get_repository_set(client, reposet):
    body = {}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="GET", path="/v1/reposet/%d" % reposet, headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = json.loads(body)
    assert len(body) == 2
    assert "github.com/athenianco/athenian-api" in body


async def test_get_repository_set_404(client):
    body = {}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="GET", path="/v1/reposet/10", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 404, "Response body is : " + body


async def test_get_repository_set_bad_team(client):
    body = {}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="GET", path="/v1/reposet/3", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


async def test_set_repository_set(client):
    body = ["github.com/vmarkovtsev/hercules"]
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="PUT", path="/v1/reposet/1", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    assert body == '["github.com/vmarkovtsev/hercules"]'


async def test_set_repository_set_404(client):
    body = ["github.com/vmarkovtsev/hercules"]
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="PUT", path="/v1/reposet/10", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 404, "Response body is : " + body


@pytest.mark.parametrize("reposet", [2, 3])
async def test_set_repository_set_bad_team(client, reposet):
    body = ["github.com/vmarkovtsev/hercules"]
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="PUT", path="/v1/reposet/%d" % reposet, headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


async def test_create_repository_set(client):
    body = RepositorySetCreateRequest(1, ["github.com/vmarkovtsev/hercules"]).to_dict()
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="POST", path="/v1/reposet/create", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = json.loads(body)
    assert body["id"] >= 4


@pytest.mark.parametrize("team", [2, 3, 10])
async def test_create_repository_set_bad_team(client, team):
    body = RepositorySetCreateRequest(team, ["github.com/vmarkovtsev/hercules"]).to_dict()
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="POST", path="/v1/reposet/create", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


@pytest.mark.parametrize("team", [1, 2])
async def test_list_repository_sets(client, team):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="GET", path="/v1/reposets/%d" % team, headers=headers, json={},
    )
    body = (await response.read()).decode("utf-8")
    items = json.loads(body)
    assert len(items) > 0
    assert items[0]["id"] == team
    assert items[0]["items_count"] == 2
    assert items[0]["created"] != ""
    assert items[0]["updated"] != ""


@pytest.mark.parametrize("team", [3, 10])
async def test_list_repository_sets_bad_team(client, team):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="GET", path="/v1/reposets/%d" % team, headers=headers, json={},
    )
    assert response.status == 403
