import json

import pytest
from sqlalchemy import select, update

from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.models.state.models import Account, ReleaseSetting, UserAccount


async def validate_settings(body, response, sdb, exhaustive: bool):
    assert response.status == 200
    repos = json.loads((await response.read()).decode("utf-8"))
    assert len(repos) > 0
    assert repos[0].startswith("github.com/")
    df = await read_sql_query(
        select([ReleaseSetting]), sdb, ReleaseSetting,
        index=[ReleaseSetting.repository.key, ReleaseSetting.account_id.key])
    if exhaustive:
        assert len(df) == len(repos)
    for r in repos:
        s = df.loc[r, body["account"]]
        assert s["branches"] == body["branches"]
        assert s["tags"] == body["tags"]
        assert s["match"] == (body["match"] == "tag")
    return repos


async def test_set_release_match_overwrite(client, headers, sdb):
    body = {
        "repositories": ["{1}"],
        "account": 1,
        "branches": "master",
        "tags": ".*",
        "match": "tag",
    }
    response = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body)
    repos = await validate_settings(body, response, sdb, True)
    assert repos == ["github.com/src-d/gitbase", "github.com/src-d/go-git"]
    body.update({
        "branches": ".*",
        "tags": "v.*",
        "match": "branch",
    })
    response = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body)
    repos = await validate_settings(body, response, sdb, True)
    assert repos == ["github.com/src-d/gitbase", "github.com/src-d/go-git"]


async def test_set_release_match_different_accounts(client, headers, sdb):
    body1 = {
        "repositories": ["github.com/src-d/go-git"],
        "account": 1,
        "branches": "master",
        "tags": ".*",
        "match": "tag",
    }
    response1 = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body1)
    await sdb.execute(update(Account).where(Account.id == 1).values(
        {Account.installation_id.key: None}))
    await sdb.execute(update(Account).where(Account.id == 2).values(
        {Account.installation_id.key: 6366825}))
    await sdb.execute(update(UserAccount).where(UserAccount.account_id == 2).values(
        {UserAccount.is_admin.key: True}))
    body2 = {
        "repositories": ["github.com/src-d/go-git"],
        "account": 2,
        "branches": ".*",
        "tags": "v.*",
        "match": "branch",
    }
    response2 = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body2)
    await validate_settings(body1, response1, sdb, False)
    await validate_settings(body2, response2, sdb, False)


@pytest.mark.parametrize("code, account, repositories, branches, tags, match", [
    (400, 1, ["{1}"], None, ".*", "branch"),
    (200, 1, ["{1}"], "", ".*", "tag"),
    (400, 1, ["{1}"], None, ".*", "tag"),
    (400, 1, ["{1}"], "", ".*", "branch"),
    (400, 1, ["{1}"], ".*", None, "tag"),
    (400, 1, ["{1}"], ".*", None, "branch"),
    (200, 1, ["{1}"], ".*", "", "branch"),
    (400, 1, ["{1}"], ".*", "", "tag"),
    (403, 2, ["{1}"], ".*", ".*", "branch"),
    (404, 3, ["{1}"], ".*", ".*", "branch"),
    (403, 1, ["{2}"], ".*", ".*", "branch"),
    (400, 1, ["{1}"], ".*", ".*", "whatever"),
    (400, 1, ["{1}"], ".*", ".*", None),
    (400, 1, ["{1}"], ".*", ".*", ""),
])
async def test_set_release_match_nasty_input(
        client, headers, code, account, repositories, branches, tags, match):
    body = {
        "repositories": repositories,
        "account": account,
        "branches": branches,
        "tags": tags,
        "match": match,
    }
    response = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body)
    assert response.status == code
