from datetime import date, datetime, timezone
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestClient
from dateutil.parser import parse as parse_datetime
from freezegun import freeze_time
import pytest
from sqlalchemy import delete, insert, select, update

from athenian.api.async_utils import gather
from athenian.api.controllers.user_controller import get_user
from athenian.api.db import Database
from athenian.api.internal.datasources import AccountDatasources
from athenian.api.models.state.models import (
    Account as DBAccount,
    AccountFeature,
    BanishedUserAccount,
    Feature,
    FeatureComponent,
    Invitation,
    UserAccount,
    UserToken,
)
from athenian.api.models.web import Account, ProductFeature
from athenian.api.request import AthenianWebRequest
from athenian.api.serialization import deserialize_datetime
from tests.conftest import disable_default_user
from tests.testutils.db import assert_existing_row, assert_missing_row, models_insert
from tests.testutils.factory.state import DEFAULT_ACCOUNT_ID, UserAccountFactory, UserTokenFactory
from tests.testutils.requester import Requester

vadim_email = "vadim@athenian.co"
eiso_email = "eiso@athenian.co"


async def test_get_user_smoke(client, headers, app):
    response = await client.request(method="GET", path="/v1/user", headers=headers, json={})
    assert response.status == 200
    items = await response.json()
    updated = items["updated"]
    del items["updated"]
    assert items == {
        "id": "auth0|62a1ae88b6bba16c6dbc6870",
        "email": vadim_email,
        "login": "vadim",
        "name": "Vadim Markovtsev",
        "native_id": "62a1ae88b6bba16c6dbc6870",
        "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
        "accounts": {
            "1": {
                "is_admin": True,
                "expired": False,
                "has_ci": True,
                "has_jira": True,
                "has_deployments": True,
            },
            "2": {
                "is_admin": False,
                "expired": False,
                "has_ci": False,
                "has_jira": False,
                "has_deployments": False,
            },
        },
    }
    assert datetime.utcnow() >= parse_datetime(updated[:-1])


# the handler is a mock with custom response so validation must be skipped
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize("value", (True, False))
async def test_is_default_user(client: TestClient, headers: dict, app, value) -> None:
    async def get_is_default_user(request: AthenianWebRequest) -> web.Response:
        return web.json_response({"is_default_user": request.is_default_user})

    for route in app.app.router.routes():
        if route.resource.canonical == "/v1/user" and route.method == "GET":
            wrapped = route.handler
            while hasattr(wrapped.__wrapped__, "__wrapped__"):
                wrapped = wrapped.__wrapped__
            wrapped.__wrapped__ = get_is_default_user
            for cell in wrapped.__closure__:
                if cell.cell_contents == get_user:
                    cell.cell_contents = get_is_default_user
    if not value:
        disable_default_user.__wrapped__(app)

    response = await client.request(method="GET", path="/v1/user", headers=headers, json={})
    items = await response.json()
    assert response.status == 200, items
    assert items == {"is_default_user": value}


async def test_get_default_user(client, headers, lazy_gkwillie):
    response = await client.request(method="GET", path="/v1/user", headers=headers, json={})
    items = await response.json()
    assert response.status == 200, items
    del items["updated"]
    assert items == {
        "id": "github|60340680",
        "login": "gkwillie",
        "name": "Groundskeeper Willie",
        "native_id": "60340680",
        "picture": "https://avatars0.githubusercontent.com/u/60340680?v=4",
        "accounts": {},
    }


async def test_get_user_sso_join(client, headers, app, sdb):
    await sdb.execute(delete(UserAccount))
    app.app["auth"]._default_user.account = 1
    response = await client.request(method="GET", path="/v1/user", headers=headers, json={})
    items = await response.json()
    assert response.status == 200, items
    updated = items["updated"]
    del items["updated"]
    assert items == {
        "id": "auth0|62a1ae88b6bba16c6dbc6870",
        "email": vadim_email,
        "login": "vadim",
        "name": "Vadim Markovtsev",
        "native_id": "62a1ae88b6bba16c6dbc6870",
        "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
        "accounts": {
            "1": {
                "is_admin": True,
                "expired": False,
                "has_ci": True,
                "has_jira": True,
                "has_deployments": True,
            },
        },
    }
    assert datetime.utcnow() >= parse_datetime(updated[:-1])


class TestGetAccountDetails(Requester):
    path = "/v1/account/{account}/details"

    @pytest.mark.flaky(reruns=10, reruns_delay=1)
    async def test_smoke(self):
        res = await self.get_json(200, path_kwargs={"account": 1})
        body = Account.from_dict(res)
        assert len(body.admins) == 1
        assert body.admins[0].name == "Vadim Markovtsev"
        assert body.admins[0].email == vadim_email
        assert len(body.regulars) == 1
        assert body.regulars[0].name == "Eiso Kant"
        assert body.regulars[0].email == eiso_email
        assert len(body.organizations) == 1
        assert body.organizations[0].login == "src-d"
        assert body.organizations[0].name == "source{d}"
        assert (
            body.organizations[0].avatar_url
            == "https://avatars3.githubusercontent.com/u/15128793?s=200&v=4"
        )
        assert body.jira is not None
        assert body.jira.url == "https://athenianco.atlassian.net"
        assert body.jira.projects == ["CON", "CS", "DEV", "ENG", "GRW", "OPS", "PRO"]
        assert sorted(body.datasources) == sorted(
            [AccountDatasources.GITHUB, AccountDatasources.JIRA],
        )

    # TODO(vmarkovtsev): remove app_validate_responses(False) after we don't test the pentest
    @pytest.mark.app_validate_responses(False)
    @pytest.mark.parametrize("account, code", [[2, 200], [3, 403], [4, 404]])
    async def test_nasty_input(self, account, code):
        res = await self.get_json(code, path_kwargs={"account": account})
        if code == 200:
            assert "jira" not in res

    async def test_account_datasources(self, sdb: Database) -> None:
        res = await self.get_json(path_kwargs={"account": 1})
        assert sorted(res["datasources"]) == sorted(
            [AccountDatasources.GITHUB, AccountDatasources.JIRA],
        )

        res = await self.get_json(path_kwargs={"account": 2})
        assert res["datasources"] == [AccountDatasources.GITHUB]


@pytest.mark.app_validate_responses(False)
async def test_get_account_features_smoke(client, headers):
    response = await client.request(
        method="GET", path="/v1/account/1/features", headers=headers, json={},
    )
    assert response.status == 200
    body = await response.json()
    models = [ProductFeature.from_dict(p) for p in body]
    assert len(models) == 3
    assert models[0].name == DBAccount.expires_at.name
    assert parse_datetime(models[0].parameters) > datetime.now(timezone.utc)
    assert models[1].name == "jira"
    assert models[1].parameters == {"a": "x", "c": "d"}
    assert models[2].name == "bare_value"
    assert models[2].parameters == 28


@pytest.mark.parametrize("account, code", [[3, 404], [4, 404]])
async def test_get_account_features_nasty_input(client, headers, account, code):
    response = await client.request(
        method="GET", path="/v1/account/%d/features" % account, headers=headers, json={},
    )
    body = await response.json()
    assert response.status == code, body


async def test_get_account_features_disabled(client, headers, sdb):
    await sdb.execute(
        update(Feature).values(
            {Feature.enabled: False, Feature.updated_at: datetime.now(timezone.utc)},
        ),
    )
    response = await client.request(
        method="GET", path="/v1/account/1/features", headers=headers, json={},
    )
    body = await response.json()
    assert len(body) == 1
    assert body[0]["name"] == DBAccount.expires_at.name
    await sdb.execute(
        update(Feature).values(
            {Feature.enabled: True, Feature.updated_at: datetime.now(timezone.utc)},
        ),
    )
    await sdb.execute(
        update(AccountFeature).values(
            {AccountFeature.enabled: False, AccountFeature.updated_at: datetime.now(timezone.utc)},
        ),
    )
    response = await client.request(
        method="GET", path="/v1/account/1/features", headers=headers, json={},
    )
    body = await response.json()
    assert len(body) == 1
    assert body[0]["name"] == DBAccount.expires_at.name


@freeze_time("2021-01-01")
async def test_set_account_features_smoke(client, headers, god, sdb):
    body = [
        {"name": "expires_at", "parameters": "2020-01-01"},
        {"name": "jira", "parameters": {"enabled": True, "parameters": {"a": "test"}}},
        {"name": "bare_value", "parameters": {"enabled": False}},
    ]
    response = await client.request(
        method="POST", path="/private/account/1/features", headers=headers, json=body,
    )
    assert response.status == 200, await response.json()
    body = await response.json()
    assert isinstance(body, list)
    assert len(body) == 2
    models = [ProductFeature.from_dict(f) for f in body]
    model = models[0]
    expires_at = deserialize_datetime(model.parameters)
    assert expires_at.date() == date(2020, 1, 1)
    model = models[1]
    expires_at = await sdb.fetch_val(select([DBAccount.expires_at]).where(DBAccount.id == 1))
    assert expires_at.date() == date(2020, 1, 1)
    assert model.parameters == {"a": "test", "c": "d"}


# TODO: response validation fails because
# ProductFeatures.properties.parameters.oneOf has both type number and type integer
@pytest.mark.app_validate_responses(False)
async def test_set_account_expires_far_future(client, headers, god, sdb) -> None:
    body = [{"name": "expires_at", "parameters": "2112-05-12T16:00:00Z"}]
    response = await client.request(
        method="POST", path="/private/account/1/features", headers=headers, json=body,
    )

    assert response.status == 200
    res_body = await response.json()
    res_expires_feat = next(feat for feat in res_body if feat["name"] == "expires_at")
    assert res_expires_feat["parameters"] == "2112-05-12T16:00:00Z"

    expires_at = await sdb.fetch_val(select([DBAccount.expires_at]).where(DBAccount.id == 1))
    if sdb.url.dialect == "sqlite":
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    assert expires_at == datetime(2112, 5, 12, 16, tzinfo=timezone.utc)


@pytest.mark.app_validate_responses(False)
async def test_set_account_expires_far_past(client, headers, god, sdb) -> None:
    orig_expires_at = await sdb.fetch_val(select([DBAccount.expires_at]).where(DBAccount.id == 1))
    body = [{"name": "expires_at", "parameters": "1984-05-01T06:00:00Z"}]
    response = await client.request(
        method="POST", path="/private/account/1/features", headers=headers, json=body,
    )
    assert response.status == 400
    # expires_at hasn't changed
    expires_at = await sdb.fetch_val(select([DBAccount.expires_at]).where(DBAccount.id == 1))
    assert expires_at == orig_expires_at


async def test_set_account_features_nongod(client, headers, sdb):
    body = [
        {"name": "expires_at", "parameters": "2020-01-01"},
    ]
    response = await client.request(
        method="POST", path="/private/account/1/features", headers=headers, json=body,
    )
    assert response.status == 403
    expires_at = await sdb.fetch_val(select([DBAccount.expires_at]).where(DBAccount.id == 1))
    assert expires_at.date() == date(2030, 1, 1)


async def test_set_account_features_nasty(client, headers, god):
    body = [
        {"name": "xxx", "parameters": "2020-01-01"},
    ]
    response = await client.request(
        method="POST", path="/private/account/1/features", headers=headers, json=body,
    )
    assert response.status == 400


@pytest.mark.parametrize(
    "body",
    [
        {
            "name": "bare_value",
            "parameters": {"enabled": True, "parameters": {"test": "2020-01-01"}},
        },
        {"name": "jira", "parameters": {"enabled": True, "parameters": '"test"'}},
    ],
)
async def test_set_account_features_type_mismatch(client, headers, god, sdb, body):
    body = [body]
    response = await client.request(
        method="POST", path="/private/account/1/features", headers=headers, json=body,
    )
    text = await response.json()
    assert "format mismatch" in str(text), text
    assert response.status == 400, text


@pytest.mark.flaky(reruns=10, reruns_delay=1)
async def test_get_users_query_size_limit(xapp):
    users = await xapp._auth0.get_users(
        ["auth0|62a1ae88b6bba16c6dbc6870"] * 200 + ["auth0|5e1f6e2e8bfa520ea5290741"] * 200,
    )
    assert len(users) == 2
    assert users["auth0|62a1ae88b6bba16c6dbc6870"].name == "Vadim Markovtsev"
    assert users["auth0|62a1ae88b6bba16c6dbc6870"].email == vadim_email
    assert users["auth0|5e1f6e2e8bfa520ea5290741"].name == "Eiso Kant"
    assert users["auth0|5e1f6e2e8bfa520ea5290741"].email == eiso_email


@pytest.mark.flaky(reruns=3, reruns_delay=60)
async def test_get_users_rate_limit(xapp):
    users = await gather(
        *[xapp._auth0.get_user("auth0|62a1ae88b6bba16c6dbc6870") for _ in range(20)],
    )
    for u in users:
        assert u is not None
        assert u.name == "Vadim Markovtsev"
        assert u.email == vadim_email


@pytest.mark.flaky(reruns=5, reruns_delay=1)
async def test_become_db(client, headers, sdb, god):
    response = await client.request(
        method="GET",
        path="/private/become?id=auth0|5e1f6e2e8bfa520ea5290741",
        headers=headers,
        json={},
    )
    body1 = await response.json()
    assert response.status == 200, body1
    response = await client.request(method="GET", path="/v1/user", headers=headers, json={})
    body2 = await response.json()
    assert response.status == 200, body2
    assert body2["impersonated_by"] == "auth0|62a1ae88b6bba16c6dbc6870"
    del body2["impersonated_by"]
    assert body1 == body2
    del body1["updated"]
    assert body1 == {
        "id": "auth0|5e1f6e2e8bfa520ea5290741",
        "login": "eiso",
        "email": eiso_email,
        "name": "Eiso Kant",
        "native_id": "5e1f6e2e8bfa520ea5290741",
        "picture": "https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
        "accounts": {
            "1": {
                "is_admin": False,
                "expired": False,
                "has_ci": True,
                "has_jira": True,
                "has_deployments": True,
            },
            "3": {
                "is_admin": True,
                "expired": False,
                "has_ci": False,
                "has_jira": False,
                "has_deployments": False,
            },
        },
    }
    response = await client.request(method="GET", path="/private/become", headers=headers, json={})
    body3 = await response.json()
    del body3["updated"]
    assert body3 == {
        "id": "auth0|62a1ae88b6bba16c6dbc6870",
        "login": "vadim",
        "email": vadim_email,
        "name": "Vadim Markovtsev",
        "native_id": "62a1ae88b6bba16c6dbc6870",
        "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
        "accounts": {
            "1": {
                "is_admin": True,
                "expired": False,
                "has_ci": True,
                "has_jira": True,
                "has_deployments": True,
            },
            "2": {
                "is_admin": False,
                "expired": False,
                "has_ci": False,
                "has_jira": False,
                "has_deployments": False,
            },
        },
    }


async def test_become_header(client, headers, sdb, god):
    headers = headers.copy()
    headers["X-Identity"] = "auth0|5e1f6e2e8bfa520ea5290741"
    response = await client.request(method="GET", path="/v1/user", headers=headers, json={})
    body = await response.json()
    del body["updated"]
    assert body == {
        "id": "auth0|5e1f6e2e8bfa520ea5290741",
        "login": "eiso",
        "email": eiso_email,
        "name": "Eiso Kant",
        "native_id": "5e1f6e2e8bfa520ea5290741",
        "picture": "https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
        "accounts": {
            "1": {
                "is_admin": False,
                "expired": False,
                "has_ci": True,
                "has_jira": True,
                "has_deployments": True,
            },
            "3": {
                "is_admin": True,
                "expired": False,
                "has_ci": False,
                "has_jira": False,
                "has_deployments": False,
            },
        },
        "impersonated_by": "auth0|62a1ae88b6bba16c6dbc6870",
    }


class TestChangeUser(Requester):
    async def test_regular(self) -> None:
        body = {"account": 1, "user": "auth0|5e1f6e2e8bfa520ea5290741", "status": "regular"}
        items = await self._request(json=body)
        assert len(items["admins"]) == 1
        assert items["admins"][0]["id"] == "auth0|62a1ae88b6bba16c6dbc6870"
        assert len(items["regulars"]) == 1
        assert items["regulars"][0]["id"] == "auth0|5e1f6e2e8bfa520ea5290741"

    async def test_invalid_status(self) -> None:
        body = {"account": 1, "user": "auth0|5e1f6e2e8bfa520ea5290741", "status": "attacker"}
        response_body = await self._request(400, json=body)
        assert "attacker" in response_body["detail"]

    async def test_admin(self) -> None:
        body = {"account": 1, "user": "auth0|5e1f6e2e8bfa520ea5290741", "status": "admin"}
        items = await self._request(json=body)
        assert len(items["admins"]) == 2
        assert items["admins"][0]["id"] == "auth0|62a1ae88b6bba16c6dbc6870"
        assert items["admins"][1]["id"] == "auth0|5e1f6e2e8bfa520ea5290741"

    @pytest.mark.parametrize("membership_check", [False, True])
    async def test_banish(self, sdb, membership_check) -> None:
        response = await self.client.request(
            method="GET", path="/v1/invite/generate/1", headers=self.headers, json={},
        )
        link1 = await response.json()
        await assert_existing_row(sdb, Invitation, is_active=True, account_id=1)
        if not membership_check:
            check_fid = await sdb.fetch_val(
                select(Feature.id).where(
                    Feature.name == Feature.USER_ORG_MEMBERSHIP_CHECK,
                    Feature.component == FeatureComponent.server,
                ),
            )
            await sdb.execute(
                insert(AccountFeature).values(
                    AccountFeature(
                        account_id=1,
                        feature_id=check_fid,
                        enabled=False,
                    )
                    .create_defaults()
                    .explode(with_primary_keys=True),
                ),
            )
        body = {
            "account": 1,
            "user": "auth0|5e1f6e2e8bfa520ea5290741",
            "status": "banished",
        }
        items = await self._request(json=body)
        assert len(items["admins"]) == 1
        assert items["admins"][0]["id"] == "auth0|62a1ae88b6bba16c6dbc6870"
        assert len(items["regulars"]) == 0
        assert "auth0|5e1f6e2e8bfa520ea5290741" == await sdb.fetch_val(
            select([BanishedUserAccount.user_id]),
        )
        if not membership_check:
            await assert_missing_row(sdb, Invitation, is_active=True, account_id=1)
            response = await self.client.request(
                method="GET", path="/v1/invite/generate/1", headers=self.headers, json={},
            )
            link2 = await response.json()
            assert link1 != link2

        await assert_existing_row(sdb, Invitation, is_active=True, account_id=1)

    @pytest.mark.parametrize(
        "account, user, status, code",
        [
            (1, "auth0|62a1ae88b6bba16c6dbc6870", "regular", 403),
            (1, "auth0|62a1ae88b6bba16c6dbc6870", "banished", 403),
            (2, "auth0|62a1ae88b6bba16c6dbc6870", "regular", 403),
            (2, "auth0|62a1ae88b6bba16c6dbc6870", "admin", 403),
            (2, "auth0|62a1ae88b6bba16c6dbc6870", "banished", 403),
            (3, "auth0|62a1ae88b6bba16c6dbc6870", "regular", 404),
        ],
    )
    async def test_errors(self, account, user, status, code):
        body = {"account": account, "user": user, "status": status}
        await self._request(code, json=body)

    async def test_banish_user_with_token(self, sdb: Database) -> None:
        await models_insert(
            sdb, UserAccountFactory(user_id="auth0|GG"), UserTokenFactory(user_id="auth0|GG"),
        )

        body = {"account": DEFAULT_ACCOUNT_ID, "user": "auth0|GG", "status": "banished"}
        await self._request(json=body)
        await self._assert_user_banished(sdb, "auth0|GG")
        await assert_missing_row(sdb, UserToken, user_id="auth0|GG")

    async def _request(self, assert_status: int = 200, **kwargs: Any) -> dict:
        path = "/v1/account/user"
        response = await self.client.request(
            method="PUT", path=path, headers=self.headers, **kwargs,
        )
        assert response.status == assert_status
        return await response.json()

    async def _assert_user_banished(self, sdb: Database, user_id: str) -> None:
        await assert_missing_row(sdb, UserAccount, user_id=user_id, account_id=DEFAULT_ACCOUNT_ID)
        await assert_existing_row(
            sdb, BanishedUserAccount, user_id=user_id, account_id=DEFAULT_ACCOUNT_ID,
        )
