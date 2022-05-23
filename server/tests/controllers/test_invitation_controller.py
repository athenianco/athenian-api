from datetime import datetime, timedelta, timezone
import json
from random import randint
import re

from freezegun import freeze_time
import morcilla
import pytest
from sqlalchemy import and_, delete, insert, select, update

from athenian.api.controllers.invitation_controller import admin_backdoor, decode_slug, \
    encode_slug, jira_url_template, url_prefix
from athenian.api.ffx import decrypt
from athenian.api.models.metadata.github import FetchProgress
from athenian.api.models.metadata.jira import Progress as JIRAProgress
from athenian.api.models.state.models import Account, AccountFeature, AccountGitHubAccount, \
    AccountJiraInstallation, BanishedUserAccount, Feature, FeatureComponent, God, Invitation, \
    ReleaseSetting, RepositorySet, UserAccount, UserToken, WorkType
from athenian.api.models.web import InvitedUser


async def clean_state(sdb: morcilla.Database) -> int:
    await sdb.execute(delete(RepositorySet))
    await sdb.execute(delete(AccountFeature))
    await sdb.execute(delete(UserAccount))
    await sdb.execute(delete(Invitation))
    await sdb.execute(delete(AccountGitHubAccount))
    await sdb.execute(delete(UserToken))
    await sdb.execute(delete(ReleaseSetting))
    await sdb.execute(delete(AccountJiraInstallation))
    await sdb.execute(delete(WorkType))
    await sdb.execute(delete(Account).where(Account.id != admin_backdoor))
    if sdb.url.dialect != "sqlite":
        await sdb.execute("ALTER SEQUENCE accounts_id_seq RESTART;")

    return await sdb.execute(
        insert(Invitation).values(
            Invitation(salt=888, account_id=admin_backdoor)
            .create_defaults().explode()))


async def test_empty_db_account_creation(client, headers, sdb, eiso, disable_default_user, app):
    iid = await clean_state(sdb)
    body = {
        "url": url_prefix + encode_slug(iid, 888, app.app["auth"].key),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == 200
    body = json.loads((await response.read()).decode("utf-8"))

    del body["user"]["updated"]
    assert body == {
        "account": 1,
        "user": {
            "id": "auth0|5e1f6e2e8bfa520ea5290741",
            "name": "Eiso Kant",
            "login": "eiso",
            "native_id": "5e1f6e2e8bfa520ea5290741",
            "email": "eiso@athenian.co",
            "picture": "https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png", # noqa
            "accounts": {
                "1": {"is_admin": True,
                      "expired": False,
                      "has_ci": False,
                      "has_jira": False,
                      "has_deployments": True,
                      }},
        },
    }
    # the second is admin backdoor
    assert len(await sdb.fetch_all(select([Account]))) == 2
    assert len(await sdb.fetch_all(select([RepositorySet]))) == 0
    response = await client.request(
        method="GET", path="/v1/reposets/1", headers=headers, json={},
    )
    assert response.status == 200
    reposets = await sdb.fetch_all(select([RepositorySet]))
    assert len(reposets) == 1
    assert "github.com/src-d/go-git" in {r[0] for r in reposets[0]["items"]}


async def test_gen_user_invitation_new(client, headers, sdb, app):
    response = await client.request(
        method="GET", path="/v1/invite/generate/1", headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    prefix = url_prefix
    assert body["url"].startswith(prefix)
    x = body["url"][len(prefix):]
    iid, salt = decode_slug(x, app.app["auth"].key)
    inv = await sdb.fetch_one(
        select([Invitation])
        .where(and_(Invitation.id == iid, Invitation.salt == salt)))
    assert inv is not None
    assert inv[Invitation.is_active.name]
    assert inv[Invitation.accepted.name] == 0
    assert inv[Invitation.account_id.name] == 1
    assert inv[Invitation.created_by.name] == "auth0|5e1f6dfb57bc640ea390557b"
    try:
        assert inv[Invitation.created_at.name] > datetime.now(timezone.utc) - timedelta(minutes=1)
    except TypeError:
        assert inv[Invitation.created_at.name] > datetime.utcnow() - timedelta(minutes=1)


async def test_gen_user_invitation_no_admin(client, headers):
    response = await client.request(
        method="GET", path="/v1/invite/generate/2", headers=headers, json={},
    )
    assert response.status == 200


async def test_gen_user_invitation_no_member(client, headers):
    response = await client.request(
        method="GET", path="/v1/invite/generate/3", headers=headers, json={},
    )
    assert response.status == 404


async def test_gen_user_invitation_existing(client, eiso, headers, app):
    response = await client.request(
        method="GET", path="/v1/invite/generate/3", headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    prefix = url_prefix
    assert body["url"].startswith(prefix)
    x = body["url"][len(prefix):]
    iid, salt = decode_slug(x, app.app["auth"].key)
    assert iid == 1
    assert salt == 777


async def test_gen_account_invitation_no_god(client, headers, sdb):
    response = await client.request(
        method="GET", path="/v1/invite/generate", headers=headers, json={},
    )
    assert response.status == 403, (await response.read()).decode("utf-8")


@freeze_time("2012-10-23T10:00:00")
async def test_gen_account_invitation_e2e(client, headers, sdb, god, disable_default_user):
    response = await client.request(
        method="GET", path="/v1/invite/generate", headers=headers, json={},
    )
    assert response.status == 200
    body = await response.json()
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    body = await response.json()
    assert response.status == 200, body
    invited_user = InvitedUser.from_dict(body)

    new_account_row = await sdb.fetch_one(
        select([Account]).where(Account.id == invited_user.account),
    )
    # expiration for the created account is based on TRIAL_PERIOD
    account_expires = new_account_row[Account.expires_at.name]
    if sdb.url.dialect == "sqlite":
        account_expires = account_expires.replace(tzinfo=timezone.utc)
    assert account_expires == datetime(2012, 11, 23, 14, tzinfo=timezone.utc)

    assert body["user"]["accounts"]["4"] == {
        "is_admin": True,
        "expired": False,
        "has_ci": False,
        "has_jira": False,
        "has_deployments": False,
    }


async def test_accept_invitation_smoke(client, headers, sdb, disable_default_user, app, faker):
    app._auth0._default_user = app._auth0._default_user.copy()
    app._auth0._default_user.login = "vmarkovtsev"
    num_accounts_before = len(await sdb.fetch_all(select([Account])))
    body = {
        "url": url_prefix + encode_slug(1, 777, app.app["auth"].key),
    }
    await sdb.execute(update(AccountGitHubAccount)
                      .values({AccountGitHubAccount.account_id: 3}))
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    rbody = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 200, rbody
    assert rbody["user"]["updated"]
    del rbody["user"]["updated"]
    assert rbody == {
        "account": 3,
        "user": {
            "id": "auth0|5e1f6dfb57bc640ea390557b",
            "name": "Vadim Markovtsev",
            "login": "vmarkovtsev",
            "native_id": "5e1f6dfb57bc640ea390557b",
            "email": "vadim@athenian.co",
            "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
            "accounts": {
                "1": {"is_admin": True,
                      "expired": False,
                      "has_ci": False,
                      "has_jira": True,
                      "has_deployments": True,
                      },
                "2": {"is_admin": False,
                      "expired": False,
                      "has_ci": False,
                      "has_jira": False,
                      "has_deployments": False,
                      },
                "3": {"is_admin": False,
                      "expired": False,
                      "has_ci": True,
                      "has_jira": False,
                      "has_deployments": False,
                      },
            },
        },
    }
    num_accounts_after = len(await sdb.fetch_all(select([Account])))
    assert num_accounts_after == num_accounts_before


@pytest.mark.flaky(reruns=5)
async def test_accept_invitation_user_profile(
        client, headers, disable_default_user, sdb, app, faker):
    app._auth0._default_user = app._auth0._default_user.copy()
    app._auth0._default_user.login = "vmarkovtsev"
    name = faker.name()
    email = faker.email()
    body = {
        "url": url_prefix + encode_slug(1, 777, app.app["auth"].key),
        "name": name,
        "email": email,
    }
    await sdb.execute(update(AccountGitHubAccount)
                      .values({AccountGitHubAccount.account_id: 3}))
    try:
        response = await client.request(
            method="PUT", path="/v1/invite/accept", headers=headers, json=body,
        )
        rbody = json.loads((await response.read()).decode("utf-8"))
        app._auth0._default_user_id = "auth0|5e1f6dfb57bc640ea390557b"
        app._auth0._default_user = None
        user = await app._auth0.default_user()
    finally:
        await app._auth0.update_user_profile(
            "auth0|5e1f6dfb57bc640ea390557b", name="Vadim Markovtsev",
            email="vadim@athenian.co")

    assert response.status == 200, rbody
    assert user.name == name
    decrypted = decrypt(user.email, b"vadim")
    assert decrypted.split(b"|")[0].decode() == email


async def test_accept_invitation_bad_email(
        client, headers, disable_default_user, sdb, app):
    app._auth0._default_user = app._auth0._default_user.copy()
    app._auth0._default_user.login = "vmarkovtsev"
    body = {
        "url": url_prefix + encode_slug(1, 777, app.app["auth"].key),
        "email": "!!!",
    }
    await sdb.execute(update(AccountGitHubAccount)
                      .values({AccountGitHubAccount.account_id: 3}))
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == 400


@pytest.mark.parametrize("installed", [False, True])
async def test_accept_invitation_disabled_membership_check(
        client, headers, sdb, disable_default_user, app, installed):
    app._auth0._default_user = app._auth0._default_user.copy()
    app._auth0._default_user.login = "panzerxxx"
    check_fid = await sdb.fetch_val(
        select([Feature.id])
        .where(and_(Feature.name == Feature.USER_ORG_MEMBERSHIP_CHECK,
                    Feature.component == FeatureComponent.server)))
    await sdb.execute(insert(AccountFeature).values(AccountFeature(
        account_id=3,
        feature_id=check_fid,
        enabled=False,
    ).create_defaults().explode(with_primary_keys=True)))
    if installed:
        await sdb.execute(update(AccountGitHubAccount)
                          .where(AccountGitHubAccount.account_id == 1)
                          .values({AccountGitHubAccount.account_id: 3}))
    body = {
        "url": url_prefix + encode_slug(1, 777, app.app["auth"].key),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    rbody = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 200, rbody


@pytest.mark.parametrize("installed", [False, True])
async def test_accept_invitation_enabled_membership_check(
        client, headers, sdb, disable_default_user, app, installed):
    app._auth0._default_user = app._auth0._default_user.copy()
    app._auth0._default_user.login = "panzerxxx"
    check_fid = await sdb.fetch_val(
        select([Feature.id])
        .where(and_(Feature.name == Feature.USER_ORG_MEMBERSHIP_CHECK,
                    Feature.component == FeatureComponent.server)))
    await sdb.execute(insert(AccountFeature).values(AccountFeature(
        account_id=3,
        feature_id=check_fid,
        enabled=True,
    ).create_defaults().explode(with_primary_keys=True)))
    if installed:
        await sdb.execute(update(AccountGitHubAccount)
                          .where(AccountGitHubAccount.account_id == 1)
                          .values({AccountGitHubAccount.account_id: 3}))
    body = {
        "url": url_prefix + encode_slug(1, 777, app.app["auth"].key),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    rbody = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 403 if installed else 200, rbody


async def test_accept_invitation_banished(
        client, headers, sdb, disable_default_user, app):
    app._auth0._default_user = app._auth0._default_user.copy()
    app._auth0._default_user.login = "vmarkovtsev"
    check_fid = await sdb.fetch_val(
        select([Feature.id])
        .where(and_(Feature.name == Feature.USER_ORG_MEMBERSHIP_CHECK,
                    Feature.component == FeatureComponent.server)))
    await sdb.execute(insert(AccountFeature).values(AccountFeature(
        account_id=3,
        feature_id=check_fid,
        enabled=False,
    ).create_defaults().explode(with_primary_keys=True)))
    await sdb.execute(insert(BanishedUserAccount).values(BanishedUserAccount(
        user_id="auth0|5e1f6dfb57bc640ea390557b",
        account_id=3,
    ).create_defaults().explode(with_primary_keys=True)))
    body = {
        "url": url_prefix + encode_slug(1, 777, app.app["auth"].key),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    rbody = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 403, rbody


async def test_accept_invitation_default_user(client, headers, app):
    body = {
        "url": url_prefix + encode_slug(1, 777, app.app["auth"].key),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == 403


async def test_accept_invitation_noop(client, eiso, headers, disable_default_user, app):
    body = {
        "url": url_prefix + encode_slug(1, 777, app.app["auth"].key),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    del body["user"]["updated"]
    assert body == {
        "account": 3,
        "user": {
            "id": "auth0|5e1f6e2e8bfa520ea5290741",
            "name": "Eiso Kant",
            "login": "eiso",
            "native_id": "5e1f6e2e8bfa520ea5290741",
            "email": "eiso@athenian.co",
            "picture": "https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
            "accounts": {
                "1": {"is_admin": False,
                      "expired": False,
                      "has_ci": True,
                      "has_jira": True,
                      "has_deployments": True,
                      },
                "3": {"is_admin": True,
                      "expired": False,
                      "has_ci": False,
                      "has_jira": False,
                      "has_deployments": False,
                      },
            },
        },
    }


@pytest.mark.parametrize("trash", ["0", "0" * 8, "a" * 8])
async def test_accept_invitation_trash(client, trash, headers, disable_default_user):
    body = {
        "url": url_prefix + "0" * 8,
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == 400


async def test_accept_invitation_inactive(client, headers, sdb, disable_default_user, app):
    await sdb.execute(
        update(Invitation).where(Invitation.id == 1).values({Invitation.is_active: False}))
    body = {
        "url": url_prefix + encode_slug(1, 777, app.app["auth"].key),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == 403


@pytest.mark.parametrize("as_github", [True, False])
async def test_accept_invitation_github_disabled(
        client, headers, sdb, disable_default_user, app, as_github):
    if as_github:
        app._auth0._default_user_id = "github|123456"
        await sdb.execute(
            update(Feature).where(Feature.name == Feature.USER_ORG_MEMBERSHIP_CHECK).values({
                Feature.enabled: False,
                Feature.updated_at: datetime.now(timezone.utc),
            }))

    await sdb.execute(update(Feature).where(Feature.name == Feature.GITHUB_LOGIN_ENABLED).values({
        Feature.enabled: False,
        Feature.updated_at: datetime.now(timezone.utc),
    }))
    body = {
        "url": url_prefix + encode_slug(1, 777, app.app["auth"].key),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == (403 if as_github else 200), (await response.read()).decode()


async def test_accept_invitation_admin_smoke(client, headers, sdb, disable_default_user, app):
    num_accounts_before = len(await sdb.fetch_all(select([Account])))
    iid = await sdb.execute(
        insert(Invitation).values(
            Invitation(salt=888, account_id=admin_backdoor)
            .create_defaults().explode()))
    body = {
        "url": url_prefix + encode_slug(iid, 888, app.app["auth"].key),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body["user"]["updated"]
    del body["user"]["updated"]
    assert body == {
        "account": 4,
        "user": {
            "id": "auth0|5e1f6dfb57bc640ea390557b",
            "name": "Vadim Markovtsev",
            "login": "vadim",
            "native_id": "5e1f6dfb57bc640ea390557b",
            "email": "vadim@athenian.co",
            "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png", # noqa
            "accounts": {
                "1": {"is_admin": True,
                      "expired": False,
                      "has_ci": True,
                      "has_jira": True,
                      "has_deployments": True,
                      },
                "2": {"is_admin": False,
                      "expired": False,
                      "has_ci": False,
                      "has_jira": False,
                      "has_deployments": False,
                      },
                "4": {"is_admin": True,
                      "expired": False,
                      "has_ci": False,
                      "has_jira": False,
                      "has_deployments": False,
                      }},
        },
    }
    accounts = await sdb.fetch_all(select([Account]))
    num_accounts_after = len(accounts)
    assert num_accounts_after == num_accounts_before + 1
    for row in accounts:
        if row[Account.id.name] == 4:
            assert row[Account.secret_salt.name] not in (None, 0)
            secret = row[Account.secret.name]
            assert isinstance(secret, str)
            assert len(secret) == 8
            assert secret != Account.missing_secret


async def test_accept_invitation_admin_duplicate_not_precomputed(
        client, headers, sdb, disable_default_user, app):
    await sdb.execute(update(RepositorySet)
                      .where(RepositorySet.id == 1)
                      .values({RepositorySet.precomputed: False,
                               RepositorySet.updates_count: RepositorySet.updates_count,
                               RepositorySet.updated_at: datetime.now(timezone.utc)}))
    num_accounts_before = len(await sdb.fetch_all(select([Account])))
    iid = await sdb.execute(
        insert(Invitation).values(
            Invitation(salt=888, account_id=admin_backdoor)
            .create_defaults().explode()))
    body = {
        "url": url_prefix + encode_slug(iid, 888, app.app["auth"].key),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == 429
    num_accounts_after = len(await sdb.fetch_all(select([Account])))
    assert num_accounts_after == num_accounts_before


async def test_accept_invitation_admin_duplicate_no_reposet(
        client, headers, sdb, disable_default_user, app):
    await sdb.execute(delete(RepositorySet))
    num_accounts_before = len(await sdb.fetch_all(select([Account])))
    iid = await sdb.execute(
        insert(Invitation).values(
            Invitation(salt=888, account_id=admin_backdoor)
            .create_defaults().explode()))
    body = {
        "url": url_prefix + encode_slug(iid, 888, app.app["auth"].key),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == 429
    num_accounts_after = len(await sdb.fetch_all(select([Account])))
    assert num_accounts_after == num_accounts_before


async def test_check_invitation(client, headers, app):
    body = {
        "url": url_prefix + encode_slug(1, 777, app.app["auth"].key),
    }
    response = await client.request(
        method="POST", path="/v1/invite/check", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == {"valid": True, "active": True, "type": "regular"}


async def test_check_invitation_not_exists(client, headers, app):
    body = {
        "url": url_prefix + encode_slug(1, 888, app.app["auth"].key),
    }
    response = await client.request(
        method="POST", path="/v1/invite/check", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == {"valid": False}


async def test_check_invitation_admin(client, headers, sdb, app):
    iid = await sdb.execute(
        insert(Invitation).values(
            Invitation(salt=888, account_id=admin_backdoor)
            .create_defaults().explode()))
    body = {
        "url": url_prefix + encode_slug(iid, 888, app.app["auth"].key),
    }
    response = await client.request(
        method="POST", path="/v1/invite/check", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == {"valid": True, "active": True, "type": "admin"}


async def test_check_invitation_inactive(client, headers, sdb, app):
    await sdb.execute(
        update(Invitation).where(Invitation.id == 1).values({Invitation.is_active: False}))
    body = {
        "url": url_prefix + encode_slug(1, 777, app.app["auth"].key),
    }
    response = await client.request(
        method="POST", path="/v1/invite/check", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == {"valid": True, "active": False, "type": "regular"}


async def test_check_invitation_malformed(client, headers):
    body = {
        "url": "https://athenian.co",
    }
    response = await client.request(
        method="POST", path="/v1/invite/check", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == {"valid": False}


async def test_accept_invitation_god(client, headers, sdb, app):
    await sdb.execute(insert(God).values(God(
        user_id="auth0|5e1f6dfb57bc640ea390557b",
        mapped_id="auth0|5e1f6e2e8bfa520ea5290741",
    ).create_defaults().explode(with_primary_keys=True)))
    iid = await sdb.execute(
        insert(Invitation).values(Invitation(
            salt=888, account_id=admin_backdoor).create_defaults().explode(),
        ))
    body = {
        "url": url_prefix + encode_slug(iid, 888, app.app["auth"].key),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == 403


def test_encode_decode(xapp):
    for _ in range(1000):
        iid = randint(0, admin_backdoor)
        salt = randint(0, (1 << 16) - 1)
        try:
            iid_back, salt_back = decode_slug(
                encode_slug(iid, salt, xapp.app["auth"].key), xapp.app["auth"].key)
        except Exception as e:
            print(iid, salt)
            raise e from None
        assert iid_back == iid
        assert salt_back == salt


complete_progress = {
    "started_date": "2020-03-10T09:53:41Z", "finished_date": "2020-03-10T14:46:29Z",
    "owner": "vmarkovtsev", "repositories": 19,
    "tables": [{"fetched": 44, "name": "AssignedEvent", "total": 44},
               {"fetched": 5, "name": "BaseRefChangedEvent", "total": 5},
               {"fetched": 40, "name": "BaseRefForcePushedEvent", "total": 40},
               {"fetched": 1, "name": "Bot", "total": 1},
               {"fetched": 1089, "name": "ClosedEvent", "total": 1089},
               {"fetched": 1, "name": "CommentDeletedEvent", "total": 1},
               {"fetched": 3308, "name": "Commit", "total": 3308},
               {"fetched": 654, "name": "CrossReferencedEvent", "total": 654},
               {"fetched": 8, "name": "DemilestonedEvent", "total": 8},
               {"fetched": 233, "name": "HeadRefDeletedEvent", "total": 233},
               {"fetched": 662, "name": "HeadRefForcePushedEvent", "total": 662},
               {"fetched": 1, "name": "HeadRefRestoredEvent", "total": 1},
               {"fetched": 607, "name": "Issue", "total": 607},
               {"fetched": 2661, "name": "IssueComment", "total": 2661},
               {"fetched": 561, "name": "LabeledEvent", "total": 561},
               {"fetched": 1, "name": "Language", "total": 1},
               {"fetched": 1, "name": "License", "total": 1},
               {"fetched": 1042, "name": "MentionedEvent", "total": 1042},
               {"fetched": 554, "name": "MergedEvent", "total": 554},
               {"fetched": 47, "name": "MilestonedEvent", "total": 47},
               {"fetched": 14, "name": "Organization", "total": 14},
               {"fetched": 682, "name": "PullRequest", "total": 682},
               {"fetched": 2369, "name": "PullRequestCommit", "total": 2369},
               {"fetched": 16, "name": "PullRequestCommitCommentThread", "total": 16},
               {"fetched": 1352, "name": "PullRequestReview", "total": 1352},
               {"fetched": 1786, "name": "PullRequestReviewComment", "total": 1786},
               {"fetched": 1095, "name": "PullRequestReviewThread", "total": 1095},
               {"fetched": 864, "name": "Reaction", "total": 864},
               {"fetched": 1, "name": "ReadyForReviewEvent", "total": 1},
               {"fetched": 54, "name": "Ref", "total": 54},
               {"fetched": 1244, "name": "ReferencedEvent", "total": 1244},
               {"fetched": 53, "name": "Release", "total": 53},
               {"fetched": 228, "name": "RenamedTitleEvent", "total": 228},
               {"fetched": 24, "name": "ReopenedEvent", "total": 24},
               {"fetched": 288, "name": "Repository", "total": 288},
               {"fetched": 8, "name": "ReviewDismissedEvent", "total": 8},
               {"fetched": 9, "name": "ReviewRequestRemovedEvent", "total": 9},
               {"fetched": 439, "name": "ReviewRequestedEvent", "total": 439},
               {"fetched": 1045, "name": "SubscribedEvent", "total": 1045},
               {"fetched": 4, "name": "UnassignedEvent", "total": 4},
               {"fetched": 32, "name": "UnlabeledEvent", "total": 32},
               {"fetched": 1, "name": "UnsubscribedEvent", "total": 1},
               {"fetched": 910, "name": "User", "total": 910},
               {"fetched": 1, "name": "precomputed", "total": 1}],
}


async def test_progress_200(client, headers, app, client_cache):
    for _ in range(2):
        response = await client.request(
            method="GET", path="/v1/invite/progress/1", headers=headers, json={},
        )
        assert response.status == 200
        body = json.loads((await response.read()).decode("utf-8"))
        assert body == complete_progress


@pytest.mark.parametrize("account, code", [(2, 422), (3, 404)])
async def test_progress_errors(client, headers, account, code):
    response = await client.request(
        method="GET", path="/v1/invite/progress/%d" % account, headers=headers, json={},
    )
    assert response.status == code


async def test_progress_idle(client, headers, mdb_rw):
    await mdb_rw.execute(update(FetchProgress).values({
        FetchProgress.nodes_total: FetchProgress.nodes_total * 2}))
    try:
        response = await client.request(
            method="GET", path="/v1/invite/progress/1", headers=headers, json={},
        )
        assert response.status == 200
        body = json.loads((await response.read()).decode("utf-8"))
        idle_complete_progress = complete_progress.copy()
        idle_complete_progress["finished_date"] = "2020-03-10T17:46:29Z"
        assert body == idle_complete_progress
    finally:
        await mdb_rw.execute(update(FetchProgress).values({
            FetchProgress.nodes_total: FetchProgress.nodes_total / 2}))


async def test_progress_no_precomputed(client, headers, sdb):
    await sdb.execute(update(RepositorySet).where(RepositorySet.id == 1).values({
        RepositorySet.precomputed: False,
        RepositorySet.updated_at: datetime.now(timezone.utc),
        RepositorySet.updates_count: 2,
        RepositorySet.created_at: datetime.now(timezone.utc) - timedelta(days=1),
    }))
    response = await client.request(
        method="GET", path="/v1/invite/progress/1", headers=headers, json={},
    )
    assert response.status == 200
    body = json.loads((await response.read()).decode("utf-8"))
    progress = complete_progress.copy()
    progress["finished_date"] = None
    progress["tables"][-1]["fetched"] = 0
    assert body == progress


async def test_jira_progress_200(client, headers, app, client_cache):
    for _ in range(2):
        response = await client.request(
            method="GET", path="/v1/invite/jira_progress/1", headers=headers, json={},
        )
        assert response.status == 200
        body = json.loads((await response.read()).decode("utf-8"))
        assert body == {
            "started_date": "2020-01-22T13:29:00Z",
            "finished_date": "2020-01-23T13:29:00Z",
            "tables": [{"fetched": 10, "name": "issue", "total": 10}],
        }


async def test_jira_progress_not_finished(client, headers, mdb_rw):
    await mdb_rw.execute(update(JIRAProgress).values({
        JIRAProgress.total: JIRAProgress.total * 2}))
    try:
        response = await client.request(
            method="GET", path="/v1/invite/jira_progress/1", headers=headers, json={},
        )
        assert response.status == 200
        body = json.loads((await response.read()).decode("utf-8"))
        assert body == {
            "started_date": "2020-01-22T13:29:00Z",
            "finished_date": None,
            "tables": [{"fetched": 10, "name": "issue", "total": 20}],
        }
    finally:
        await mdb_rw.execute(update(JIRAProgress).values({
            JIRAProgress.total: JIRAProgress.total / 2}))


@pytest.mark.parametrize("account, code", [(2, 422), (3, 404)])
async def test_jira_progress_errors(client, headers, account, code):
    response = await client.request(
        method="GET", path="/v1/invite/jira_progress/%d" % account, headers=headers, json={},
    )
    assert response.status == code


async def test_gen_jira_link_smoke(client, headers):
    response = await client.request(
        method="GET", path="/v1/invite/jira/1", headers=headers, json={},
    )
    assert response.status == 200
    body = json.loads((await response.read()).decode("utf-8"))
    url = body["url"]
    assert re.match(jira_url_template % "[a-z0-9]{8}", url)
    body = json.loads((await response.read()).decode("utf-8"))
    assert url == body["url"]


@pytest.mark.parametrize("account, code", [(2, 403), (10, 404)])
async def test_gen_jira_link_errors(client, headers, account, code):
    response = await client.request(
        method="GET", path="/v1/invite/jira/%d" % account, headers=headers, json={},
    )
    assert response.status == code
