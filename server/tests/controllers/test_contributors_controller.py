import json

import pytest

from athenian.api.models.web import Contributor


@pytest.mark.parametrize("cached", [False, True], ids=["no cache", "with cache"])
async def test_get_contributors_as_admin(client, cached, headers, app, cache):
    await _test_get_contributors(client, cached, headers, app, cache)


@pytest.mark.parametrize("cached", [False, True], ids=["no cache", "with cache"])
async def test_get_contributors_as_non_admin(client, cached, headers, app, cache, eiso):
    await _test_get_contributors(client, cached, headers, app, cache)


async def _test_get_contributors(client, cached, headers, app, cache):
    if cached:
        app._cache = cache

    response = await client.request(
        method="GET", path="/v1/contributors/1", headers=headers,
    )

    assert response.status == 200

    contribs = [Contributor.from_dict(c) for c in json.loads(
        (await response.read()).decode("utf-8"))]

    assert len(contribs) == 206
    assert len(set(c.login for c in contribs)) == len(contribs)
    assert all(c.login.startswith("github.com/") for c in contribs)

    contribs = {c.login: c for c in contribs}
    assert "github.com/mcuadros" in contribs
    assert "github.com/author_login" not in contribs
    assert "github.com/committer_login" not in contribs
    assert contribs["github.com/mcuadros"].picture
    assert contribs["github.com/mcuadros"].name == "Máximo Cuadros"


async def test_get_contributors_no_installation(client, headers):
    response = await client.request(
        method="GET", path="/v1/contributors/2", headers=headers,
    )

    assert response.status == 422

    parsed = json.loads((await response.read()).decode("utf-8"))

    assert parsed == {
        "type": "/errors/NoSourceDataError",
        "title": "Unprocessable Entity",
        "status": 422,
        "detail": "The account installation has not finished yet.",
    }


@pytest.mark.parametrize("account", [3, 4],
                         ids=["user is not a member of account",
                              "account doesn't exist"])
async def test_get_contributors_not_found(client, account, headers):
    response = await client.request(
        method="GET", path="/v1/contributors/3", headers=headers,
    )

    assert response.status == 404

    parsed = json.loads((await response.read()).decode("utf-8"))
    err_detail = parsed.pop("detail")

    assert err_detail.startswith("Account None does not exist or user")
    assert parsed == {
        "status": 404,
        "title": "Not Found",
        "type": "/errors/NotFoundError",
    }
