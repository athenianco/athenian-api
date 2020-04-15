from datetime import datetime, timedelta, timezone
import json
from random import randint

import pytest
from sqlalchemy import and_, delete, insert, select, update

from athenian.api.controllers import invitation_controller
from athenian.api.models.state.models import Account, Installation, Invitation, RepositorySet, \
    UserAccount


async def test_empty_db_account_creation(client, headers, sdb):
    await sdb.execute(delete(RepositorySet))
    await sdb.execute(delete(UserAccount))
    await sdb.execute(delete(Invitation))
    await sdb.execute(delete(Installation))
    await sdb.execute(delete(Account).where(Account.id != invitation_controller.admin_backdoor))
    if sdb.url.dialect != "sqlite":
        await sdb.execute("ALTER SEQUENCE accounts_id_seq RESTART;")

    iid = await sdb.execute(
        insert(Invitation).values(
            Invitation(salt=888, account_id=invitation_controller.admin_backdoor)
            .create_defaults().explode()))
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(iid, 888),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )

    body = json.loads((await response.read()).decode("utf-8"))

    del body["user"]["updated"]
    assert body == {
        "account": 1,
        "user": {
            "id": "auth0|5e1f6dfb57bc640ea390557b",
            "name": "Vadim Markovtsev",
            "native_id": "5e1f6dfb57bc640ea390557b",
            "email": "vadim@athenian.co",
            "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png", # noqa
            "accounts": {"1": True},
        },
    }
    # the second is admin backdoor
    assert len(await sdb.fetch_all(select([Account]))) == 2


async def test_gen_invitation_new(client, headers, sdb):
    response = await client.request(
        method="GET", path="/v1/invite/generate/1", headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    prefix = invitation_controller.url_prefix
    assert body["url"].startswith(prefix)
    x = body["url"][len(prefix):]
    iid, salt = invitation_controller.decode_slug(x)
    inv = await sdb.fetch_one(
        select([Invitation])
        .where(and_(Invitation.id == iid, Invitation.salt == salt)))
    assert inv is not None
    assert inv[Invitation.is_active.key]
    assert inv[Invitation.accepted.key] == 0
    assert inv[Invitation.account_id.key] == 1
    assert inv[Invitation.created_by.key] == "auth0|5e1f6dfb57bc640ea390557b"
    try:
        assert inv[Invitation.created_at.key] > datetime.now(timezone.utc) - timedelta(minutes=1)
    except TypeError:
        assert inv[Invitation.created_at.key] > datetime.utcnow() - timedelta(minutes=1)


async def test_gen_invitation_no_admin(client, headers):
    response = await client.request(
        method="GET", path="/v1/invite/generate/2", headers=headers, json={},
    )
    assert response.status == 403


async def test_gen_invitation_no_member(client, headers):
    response = await client.request(
        method="GET", path="/v1/invite/generate/3", headers=headers, json={},
    )
    assert response.status == 404


async def test_gen_invitation_existing(client, eiso, headers):
    response = await client.request(
        method="GET", path="/v1/invite/generate/3", headers=headers, json={},
    )
    body = json.loads((await response.read()).decode("utf-8"))
    prefix = invitation_controller.url_prefix
    assert body["url"].startswith(prefix)
    x = body["url"][len(prefix):]
    iid, salt = invitation_controller.decode_slug(x)
    assert iid == 1
    assert salt == 777


async def test_accept_invitation_smoke(client, headers, sdb):
    num_accounts_before = len(await sdb.fetch_all(select([Account])))
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(1, 777),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    del body["user"]["updated"]
    assert body == {
        "account": 3,
        "user": {
            "id": "auth0|5e1f6dfb57bc640ea390557b",
            "name": "Vadim Markovtsev",
            "native_id": "5e1f6dfb57bc640ea390557b",
            "email": "vadim@athenian.co",
            "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
            "accounts": {"1": True, "2": False, "3": False},
        },
    }
    num_accounts_after = len(await sdb.fetch_all(select([Account])))
    assert num_accounts_after == num_accounts_before


async def test_accept_invitation_noop(client, eiso, headers):
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(1, 777),
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
            "native_id": "5e1f6e2e8bfa520ea5290741",
            "email": "eiso@athenian.co",
            "picture": "https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
            "accounts": {"1": False, "3": True},
        },
    }


@pytest.mark.parametrize("trash", ["0", "0" * 8, "a" * 8])
async def test_accept_invitation_trash(client, trash, headers):
    body = {
        "url": invitation_controller.url_prefix + "0" * 8,
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == 400


async def test_accept_invitation_inactive(client, headers, sdb):
    await sdb.execute(
        update(Invitation).where(Invitation.id == 1).values({Invitation.is_active: False}))
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(1, 777),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == 403


async def test_accept_invitation_admin(client, headers, sdb):
    # avoid 429 cooldown
    cooldowntd = invitation_controller.accept_admin_cooldown
    await sdb.execute(update(UserAccount)
                      .where(UserAccount.user_id == "auth0|5e1f6dfb57bc640ea390557b")
                      .values({UserAccount.created_at: datetime.now(timezone.utc) - cooldowntd}))
    num_accounts_before = len(await sdb.fetch_all(select([Account])))
    iid = await sdb.execute(
        insert(Invitation).values(
            Invitation(salt=888, account_id=invitation_controller.admin_backdoor)
            .create_defaults().explode()))
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(iid, 888),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    del body["user"]["updated"]
    assert body == {
        "account": 4,
        "user": {
            "id": "auth0|5e1f6dfb57bc640ea390557b",
            "name": "Vadim Markovtsev",
            "native_id": "5e1f6dfb57bc640ea390557b",
            "email": "vadim@athenian.co",
            "picture": "https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png", # noqa
            "accounts": {"1": True, "2": False, "4": True},
        },
    }
    num_accounts_after = len(await sdb.fetch_all(select([Account])))
    assert num_accounts_after == num_accounts_before + 1


async def test_accept_invitation_admin_cooldown(client, headers, sdb):
    await sdb.execute(update(UserAccount)
                      .where(and_(UserAccount.user_id == "auth0|5e1f6dfb57bc640ea390557b",
                                  UserAccount.is_admin))
                      .values({UserAccount.created_at: datetime.now(timezone.utc)}))
    num_accounts_before = len(await sdb.fetch_all(select([Account])))
    iid = await sdb.execute(
        insert(Invitation).values(
            Invitation(salt=888, account_id=invitation_controller.admin_backdoor)
            .create_defaults().explode()))
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(iid, 888),
    }
    response = await client.request(
        method="PUT", path="/v1/invite/accept", headers=headers, json=body,
    )
    assert response.status == 429
    num_accounts_after = len(await sdb.fetch_all(select([Account])))
    assert num_accounts_after == num_accounts_before


async def test_check_invitation(client, headers):
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(1, 777),
    }
    response = await client.request(
        method="POST", path="/v1/invite/check", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == {"valid": True, "active": True, "type": "regular"}


async def test_check_invitation_not_exists(client, headers):
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(1, 888),
    }
    response = await client.request(
        method="POST", path="/v1/invite/check", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == {"valid": False}


async def test_check_invitation_admin(client, headers, sdb):
    iid = await sdb.execute(
        insert(Invitation).values(
            Invitation(salt=888, account_id=invitation_controller.admin_backdoor)
            .create_defaults().explode()))
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(iid, 888),
    }
    response = await client.request(
        method="POST", path="/v1/invite/check", headers=headers, json=body,
    )
    body = json.loads((await response.read()).decode("utf-8"))
    assert body == {"valid": True, "active": True, "type": "admin"}


async def test_check_invitation_inactive(client, headers, sdb):
    await sdb.execute(
        update(Invitation).where(Invitation.id == 1).values({Invitation.is_active: False}))
    body = {
        "url": invitation_controller.url_prefix + invitation_controller.encode_slug(1, 777),
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


def test_encode_decode():
    for _ in range(1000):
        iid = randint(0, invitation_controller.admin_backdoor)
        salt = randint(0, (1 << 16) - 1)
        try:
            iid_back, salt_back = invitation_controller.decode_slug(
                invitation_controller.encode_slug(iid, salt))
        except Exception as e:
            print(iid, salt)
            raise e from None
        assert iid_back == iid
        assert salt_back == salt


async def test_progress_200(client, headers, app, cache):
    app._cache = cache
    true_body = {
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
                   {"fetched": 910, "name": "User", "total": 910}],
    }
    for _ in range(2):
        response = await client.request(
            method="GET", path="/v1/invite/progress/1", headers=headers, json={},
        )
        assert response.status == 200
        body = json.loads((await response.read()).decode("utf-8"))
        assert body == true_body


@pytest.mark.parametrize("account, code", [(2, 422), (3, 404)])
async def test_progress_errors(client, headers, account, code):
    response = await client.request(
        method="GET", path="/v1/invite/progress/%d" % account, headers=headers, json={},
    )
    assert response.status == code
