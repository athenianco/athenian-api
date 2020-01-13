from sqlalchemy import select

from athenian.api.models.state.models import RepositorySet


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


async def test_get_repository_set(client):
    body = {}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="GET", path="/v1/reposet/1", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    assert body == '["github.com/src-d/go-git", "github.com/athenianco/athenian-api"]'


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


async def test_create_repository_set(client):
    body = ["github.com/vmarkovtsev/hercules"]
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="POST", path="/v1/reposet/create", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    assert body == '{"id": 2}'
