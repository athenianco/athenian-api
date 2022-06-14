import json

import pytest
from sqlalchemy import delete

from athenian.api.cache import CACHE_VAR_NAME
from athenian.api.models.state.models import RepositorySet
from athenian.api.models.web import Contributor


@pytest.mark.parametrize("cached", [False, True], ids=["no cache", "with cache"])
async def test_get_contributors_as_admin(client, cached, headers, app, client_cache):
    await _test_get_contributors(client, cached, headers, app, client_cache)


@pytest.mark.parametrize("cached", [False, True], ids=["no cache", "with cache"])
async def test_get_contributors_as_non_admin(client, cached, headers, app, client_cache, eiso):
    await _test_get_contributors(client, cached, headers, app, client_cache)


async def _test_get_contributors(client, cached, headers, app, client_cache):
    if not cached:
        app.app[CACHE_VAR_NAME] = None

    response = await client.request(method="GET", path="/v1/get/contributors/1", headers=headers)

    assert response.status == 200

    contribs = [
        Contributor.from_dict(c) for c in json.loads((await response.read()).decode("utf-8"))
    ]

    assert len(contribs) == 212
    assert len(set(c.login for c in contribs)) == len(contribs)
    assert all(c.login.startswith("github.com/") for c in contribs)

    contribs = {c.login: c for c in contribs}
    assert len(contribs) == 212
    assert "github.com/mcuadros" in contribs
    assert "github.com/author_login" not in contribs
    assert "github.com/committer_login" not in contribs
    assert contribs["github.com/mcuadros"].picture
    assert contribs["github.com/mcuadros"].name == "Máximo Cuadros"
    assert "github.com/warenlg" in contribs


async def test_get_contributors_no_installation(client, headers, sdb):
    await sdb.execute(delete(RepositorySet))
    response = await client.request(method="GET", path="/v1/get/contributors/2", headers=headers)

    assert response.status == 422

    parsed = json.loads((await response.read()).decode("utf-8"))

    assert parsed == {
        "type": "/errors/NoSourceDataError",
        "title": "Unprocessable Entity",
        "status": 422,
        "detail": "The installation of account 2 has not finished yet.",
    }


@pytest.mark.parametrize(
    "account", [3, 4], ids=["user is not a member of account", "account doesn't exist"]
)
async def test_get_contributors_not_found(client, account, headers):
    response = await client.request(method="GET", path="/v1/get/contributors/3", headers=headers)

    assert response.status == 404

    parsed = json.loads((await response.read()).decode("utf-8"))
    err_detail = parsed.pop("detail")

    assert err_detail.startswith("Account None does not exist or user")
    assert parsed == {
        "status": 404,
        "title": "Not Found",
        "type": "/errors/NotFoundError",
    }
