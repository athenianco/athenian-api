import json


async def test_filter_repositories_no_repos(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == ["github.com/src-d/go-git"]
    body["date_from"] = body["date_to"]
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == []


async def test_filter_repositories(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
        "in": ["github.com/src-d/go-git"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == ["github.com/src-d/go-git"]
    body["in"] = ["github.com/athenianco/athenian-api"]
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == []


async def test_filter_repositories_bad_account(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 3,
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    assert response.status == 403
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 2,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    assert response.status == 403
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    assert response.status == 200


async def test_filter_contributors_no_repos(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    contribs = json.loads((await response.read()).decode("utf-8"))
    assert len(contribs) == 166
    assert len(set(contribs)) == len(contribs)
    assert all(c.startswith("github.com/") for c in contribs)
    assert "github.com/mcuadros" in contribs
    body["date_from"] = body["date_to"]
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    contribs = json.loads((await response.read()).decode("utf-8"))
    assert contribs == []


async def test_filter_contributors(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
        "in": ["github.com/src-d/go-git"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    contribs = json.loads((await response.read()).decode("utf-8"))
    assert len(contribs) == 166
    assert len(set(contribs)) == len(contribs)
    assert all(c.startswith("github.com/") for c in contribs)
    assert "github.com/mcuadros" in contribs
    body["in"] = ["github.com/athenianco/athenian-api"]
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    contribs = json.loads((await response.read()).decode("utf-8"))
    assert contribs == []


async def test_filter_contributors_bad_account(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 3,
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    assert response.status == 403
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 2,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    assert response.status == 403
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    assert response.status == 200
