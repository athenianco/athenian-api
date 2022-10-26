from athenian.api.models.state.models import Team
from athenian.api.models.web import ResetTarget
from tests.testutils.db import models_insert_auto_pk
from tests.testutils.factory.state import TeamFactory


async def test_reset_account_everything(client, headers, god):
    body = {
        "targets": list(ResetTarget),
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/private/reset", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text


async def test_reset_account_repos(client, headers, god, sdb):
    await models_insert_auto_pk(sdb, TeamFactory(name=Team.ROOT))
    body = {
        "targets": list(ResetTarget),
        "repositories": ["github.com/src-d/go-git"],
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/private/reset", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text


async def test_reset_account_bad_repos(client, headers, god):
    body = {
        "targets": list(ResetTarget),
        "repositories": ["github.com/athenianco/athenian-api"],
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/private/reset", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 403, text


async def test_reset_account_no_god(client, headers):
    body = {
        "targets": [],
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/private/reset", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 403, text


async def test_reset_account_bad_account(client, headers):
    body = {
        "targets": [],
        "account": 10,
    }
    response = await client.request(
        method="POST", path="/private/reset", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 404, text
