import contextlib
from unittest import mock

from athenian.api.models.state.models import Team
from athenian.api.models.web import ResetTarget
from tests.testutils.db import models_insert_auto_pk
from tests.testutils.factory.state import TeamFactory


async def test_reset_account_everything(client, headers, god, sdb):
    body = {"targets": list(ResetTarget), "account": 1}
    await models_insert_auto_pk(sdb, TeamFactory(name=Team.ROOT))
    with _spy_resetters():
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
    with _spy_resetters():
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
    with _spy_resetters():
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
    with _spy_resetters():
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
    with _spy_resetters():
        response = await client.request(
            method="POST", path="/private/reset", headers=headers, json=body,
        )
    text = (await response.read()).decode("utf-8")
    assert response.status == 404, text


@contextlib.contextmanager
def _spy_resetters(assert_no_errors: bool = True):
    from athenian.api.controllers.backoffice_controller import _resetters

    wrapped_resetters = {}
    errored_resetters = {}
    for k, v in _resetters.items():

        async def wrapper(*args, v=v, k=k, **kwargs):
            try:
                await v(*args, **kwargs)
            except Exception as e:
                errored_resetters[k] = e
                raise

        wrapped_resetters[k] = mock.Mock(side_effect=wrapper)

    with mock.patch.dict(_resetters, wrapped_resetters):
        yield (wrapped_resetters, errored_resetters)

    if assert_no_errors:
        assert not errored_resetters
