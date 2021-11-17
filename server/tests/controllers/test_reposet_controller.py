import json

import pytest
from sqlalchemy import select

from athenian.api.controllers.reposet import load_account_reposets
from athenian.api.models.state.models import RepositorySet
from athenian.api.models.web import RepositorySetCreateRequest, RepositorySetWithName
from athenian.api.response import ResponseError


async def test_delete_repository_set(client, app, headers, disable_default_user, sdb):
    body = {}
    response = await client.request(
        method="DELETE", path="/v1/reposet/1", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    rs = await sdb.fetch_one(select([RepositorySet]).where(RepositorySet.id == 1))
    assert rs is None


async def test_delete_repository_set_404(client, app, headers, disable_default_user):
    body = {}
    response = await client.request(
        method="DELETE", path="/v1/reposet/10", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 404, "Response body is : " + body


async def test_delete_repository_set_default_user(client, app, headers):
    body = {}
    response = await client.request(
        method="DELETE", path="/v1/reposet/1", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


@pytest.mark.parametrize("reposet", [2, 3])
async def test_delete_repository_set_bad_account(client, reposet, headers, disable_default_user):
    body = {}
    response = await client.request(
        method="DELETE", path="/v1/reposet/%d" % reposet, headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == (403 if reposet == 2 else 404), "Response body is : " + body


@pytest.mark.parametrize("reposet,checked", [(1, "github.com/src-d/go-git"),
                                             (2, "github.com/src-d/hercules")])
async def test_get_repository_set_smoke(client, reposet, headers, checked):
    response = await client.request(
        method="GET", path="/v1/reposet/%d" % reposet, headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = RepositorySetWithName.from_dict(json.loads(body))
    assert len(body.items) == 2
    assert checked in body.items
    assert body.name == "all"
    assert body.precomputed == (reposet == 1)


async def test_get_repository_set_logical(client, headers, logical_settings_db):
    response = await client.request(
        method="GET", path="/v1/reposet/1", headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = RepositorySetWithName.from_dict(json.loads(body))
    assert body.items == [
        "github.com/src-d/gitbase", "github.com/src-d/go-git", "github.com/src-d/go-git/alpha",
    ]
    assert body.name == "all"


async def test_get_repository_set_404(client, headers):
    response = await client.request(
        method="GET", path="/v1/reposet/10", headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 404, "Response body is : " + body


async def test_get_repository_set_bad_account(client, headers):
    response = await client.request(
        method="GET", path="/v1/reposet/3", headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 404, "Response body is : " + body


@pytest.mark.parametrize("name, items, new_name, new_items", [
    ("xxx", ["github.com/src-d/hercules"], "xxx", ["github.com/src-d/hercules"]),
    (None, ["github.com/src-d/hercules"], "all", ["github.com/src-d/hercules"]),
    ("xxx", None, "xxx", ["github.com/src-d/gitbase", "github.com/src-d/go-git"]),
])
async def test_set_repository_set_smoke(
        client, headers, disable_default_user, name, items, new_name, new_items):
    body = {}
    if name is not None:
        body["name"] = name
    if not (precomputed := items is None):
        body["items"] = items
    response = await client.request(
        method="PUT", path="/v1/reposet/1", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    assert json.loads(body) == {"name": new_name, "items": new_items, "precomputed": precomputed}


@pytest.mark.parametrize("name, items", [
    ("", ["github.com/src-d/gitbase"]),
    (None, []),
])
async def test_set_repository_set_400(client, headers, disable_default_user, name, items):
    body = {}
    if name is not None:
        body["name"] = name
    if items is not None:
        body["items"] = items
    response = await client.request(
        method="PUT", path="/v1/reposet/1", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 400, "Response body is : " + body


async def test_set_repository_set_default_user(client, headers):
    body = {"name": "xxx", "items": ["github.com/src-d/hercules"]}
    response = await client.request(
        method="PUT", path="/v1/reposet/1", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


async def test_set_repository_set_404(client, headers, disable_default_user):
    body = {"name": "xxx", "items": ["github.com/src-d/hercules"]}
    response = await client.request(
        method="PUT", path="/v1/reposet/10", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 404, "Response body is : " + body


async def test_set_repository_set_same(client, headers, disable_default_user):
    body = {"name": "xxx", "items": ["github.com/src-d/go-git", "github.com/src-d/gitbase"]}
    response = await client.request(
        method="PUT", path="/v1/reposet/1", headers=headers, json=body,
    )
    assert response.status == 200, "Response body is : " + body


async def test_set_repository_set_409(client, headers, disable_default_user):
    body = RepositorySetCreateRequest(
        1, name="xxx", items=["github.com/src-d/go-git"]).to_dict()
    await client.request(
        method="POST", path="/v1/reposet/create", headers=headers, json=body,
    )
    body = RepositorySetWithName(name="yyy", items=["github.com/src-d/go-git"]).to_dict()
    response = await client.request(
        method="PUT", path="/v1/reposet/1", headers=headers, json=body,
    )
    assert response.status == 409
    body["name"] = "xxx"
    body["items"] = ["github.com/src-d/go-git", "github.com/src-d/gitbase"]
    response = await client.request(
        method="PUT", path="/v1/reposet/1", headers=headers, json=body,
    )
    assert response.status == 409


@pytest.mark.parametrize("reposet", [2, 3])
async def test_set_repository_set_bad_account(client, reposet, headers, disable_default_user):
    body = {"name": "xxx", "items": ["github.com/src-d/hercules"]}
    response = await client.request(
        method="PUT", path="/v1/reposet/%d" % reposet, headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == (403 if reposet == 2 else 404), "Response body is : " + body


async def test_set_repository_set_access_denied(client, headers, disable_default_user):
    body = {"name": "xxx", "items": ["github.com/athenianco/athenian-api"]}
    response = await client.request(
        method="PUT", path="/v1/reposet/1", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


async def test_create_repository_set_smoke(client, headers, disable_default_user):
    body = RepositorySetCreateRequest(
        1, name="xxx", items=["github.com/src-d/hercules"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/reposet/create", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = json.loads(body)
    assert body["id"] >= 4


async def test_create_repository_set_default_user(client, headers):
    body = RepositorySetCreateRequest(
        1, name="xxx", items=["github.com/src-d/hercules"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/reposet/create", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


async def test_create_repository_set_409(client, headers, disable_default_user):
    body = RepositorySetCreateRequest(
        1, name="xxx", items=["github.com/src-d/go-git", "github.com/src-d/gitbase"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/reposet/create", headers=headers, json=body,
    )
    assert response.status == 409
    body = RepositorySetCreateRequest(
        1, name="all", items=["github.com/src-d/go-git"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/reposet/create", headers=headers, json=body,
    )
    assert response.status == 409


@pytest.mark.parametrize("account", [2, 3, 10])
async def test_create_repository_set_bad_account(client, account, headers, disable_default_user):
    body = RepositorySetCreateRequest(
        account, name="xxx", items=["github.com/src-d/hercules"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/reposet/create", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == (404 if account != 2 else 403), "Response body is : " + body


async def test_create_repository_set_access_denied(client, headers, disable_default_user):
    body = RepositorySetCreateRequest(
        1, name="xxx", items=["github.com/athenianco/athenian-api"]).to_dict()
    response = await client.request(
        method="POST", path="/v1/reposet/create", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


@pytest.mark.parametrize("account", [1, 2])
async def test_list_repository_sets(client, account, headers):
    response = await client.request(
        method="GET", path="/v1/reposets/%d" % account, headers=headers, json={},
    )
    body = (await response.read()).decode("utf-8")
    items = json.loads(body)
    assert len(items) > 0
    assert items[0]["id"] == account
    assert items[0]["name"] == "all"
    assert items[0]["items_count"] == 2
    assert items[0]["created"] != ""
    assert items[0]["updated"] != ""


@pytest.mark.parametrize("account", [3, 10])
async def test_list_repository_sets_bad_account(client, account, headers):
    response = await client.request(
        method="GET", path="/v1/reposets/%d" % account, headers=headers, json={},
    )
    assert response.status == 404


async def test_list_repository_sets_installation(client, sdb, headers):
    await sdb.execute(RepositorySet.__table__.delete())
    response = await client.request(
        method="GET", path="/v1/reposets/1", headers=headers, json={},
    )
    body = (await response.read()).decode("utf-8")
    items = json.loads(body)
    assert len(items) == 1
    assert items[0]["id"] == 4
    assert items[0]["name"] == "all"
    assert items[0]["items_count"] == 2
    assert items[0]["created"] != ""
    assert items[0]["updated"] != ""


@pytest.mark.parametrize("user_id", ["777", "676724"])
async def test_load_account_reposets_bad_installation(sdb, mdb, user_id):
    async def login():
        return user_id

    with pytest.raises(ResponseError):
        await load_account_reposets(
            4, login, [RepositorySet.items], sdb, mdb, None, None)
