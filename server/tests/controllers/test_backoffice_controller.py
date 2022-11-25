import contextlib
from unittest import mock

import pytest
from sqlalchemy import select

from athenian.api.models.state.models import Team, UserAccount
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
    assert response.status == 401, text


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
        yield wrapped_resetters, errored_resetters

    if assert_no_errors:
        assert not errored_resetters


@pytest.mark.parametrize("acc_key", ["new_account_regular", "new_account_admin"])
async def test_user_move_smoke(client, headers, acc_key, god, sdb):
    body = {
        "user": "auth0|5e1f6e2e8bfa520ea5290741",
        "old_account": 1,
        acc_key: 2,
    }
    response = await client.request(
        method="POST", path="/private/user/move", headers=headers, json=body,
    )
    assert response.status == 200, (await response.read()).decode()
    rows = await sdb.fetch_all(
        select(UserAccount.account_id, UserAccount.is_admin)
        .where(UserAccount.user_id == "auth0|5e1f6e2e8bfa520ea5290741")
        .order_by(UserAccount.account_id),
    )
    assert len(rows) == 2, rows
    assert rows[0][0] == 2
    assert rows[0][1] == (acc_key == "new_account_admin")
    assert rows[1][0] == 3


async def test_user_move_no_god(client, headers, sdb):
    body = {
        "user": "auth0|5e1f6e2e8bfa520ea5290741",
        "old_account": 1,
        "new_account_regular": 2,
    }
    response = await client.request(
        method="POST", path="/private/user/move", headers=headers, json=body,
    )
    assert response.status == 403, (await response.read()).decode()
    rows = await sdb.fetch_all(
        select(UserAccount.account_id)
        .where(UserAccount.user_id == "auth0|5e1f6e2e8bfa520ea5290741")
        .order_by(UserAccount.account_id),
    )
    assert len(rows) == 2, rows
    assert rows[0][0] == 1
    assert rows[1][0] == 3


@pytest.mark.parametrize(
    "user,old_account,new_account,status",
    [
        ("???", 1, {"new_account_regular": 2}, 404),
        ("auth0|5e1f6e2e8bfa520ea5290741", 2, {"new_account_regular": 3}, 404),
        ("auth0|5e1f6e2e8bfa520ea5290741", 1, {"new_account_regular": 1}, 400),
        ("auth0|5e1f6e2e8bfa520ea5290741", 1, {"new_account_regular": 4}, 409),
        (
            "auth0|5e1f6e2e8bfa520ea5290741",
            1,
            {"new_account_regular": 2, "new_account_admin": 2},
            400,
        ),
        ("auth0|5e1f6e2e8bfa520ea5290741", 1, {}, 400),
        ("auth0|5e1f6e2e8bfa520ea5290741", 1, {"new_account_regular": 3}, 409),
    ],
)
async def test_user_move_nasty_input(client, headers, user, old_account, new_account, status, god):
    body = {
        "user": user,
        "old_account": old_account,
        **new_account,
    }
    response = await client.request(
        method="POST", path="/private/user/move", headers=headers, json=body,
    )
    assert response.status == status, (await response.read()).decode()
