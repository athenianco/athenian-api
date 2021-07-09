import json

from aiohttp.web_runner import GracefulExit
import pytest
from sqlalchemy import delete, insert, select, update

from athenian.api import auth
from athenian.api.async_utils import read_sql_query
from athenian.api.models.metadata.github import NodeUser
from athenian.api.models.state.models import AccountGitHubAccount, MappedJIRAIdentity, \
    ReleaseSetting, RepositorySet, UserAccount
from athenian.api.models.web import JIRAProject, MappedJIRAIdentity as WebMappedJIRAIdentity, \
    ReleaseMatchSetting, ReleaseMatchStrategy
from athenian.api.serialization import FriendlyJson


async def validate_settings(body, response, sdb, exhaustive: bool):
    assert response.status == 200
    repos = json.loads((await response.read()).decode("utf-8"))
    assert len(repos) > 0
    assert repos[0].startswith("github.com/")
    df = await read_sql_query(
        select([ReleaseSetting]), sdb, ReleaseSetting,
        index=[ReleaseSetting.repository.name, ReleaseSetting.account_id.name])
    if exhaustive:
        assert len(df) == len(repos)
    for r in repos:
        s = df.loc[r, body["account"]]
        assert s["branches"] == body["branches"]
        assert s["tags"] == body["tags"]
        assert s["match"] == (body["match"] == ReleaseMatchStrategy.TAG)
    return repos


async def test_set_release_match_overwrite(client, headers, sdb, disable_default_user):
    body = {
        "repositories": ["{1}"],
        "account": 1,
        "branches": "master",
        "tags": ".*",
        "match": ReleaseMatchStrategy.TAG,
    }
    response = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body)
    repos = await validate_settings(body, response, sdb, True)
    assert repos == ["github.com/src-d/gitbase", "github.com/src-d/go-git"]
    body.update({
        "branches": ".*",
        "tags": "v.*",
        "match": ReleaseMatchStrategy.BRANCH,
    })
    response = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body)
    repos = await validate_settings(body, response, sdb, True)
    assert repos == ["github.com/src-d/gitbase", "github.com/src-d/go-git"]


async def test_set_release_match_different_accounts(client, headers, sdb, disable_default_user):
    body1 = {
        "repositories": ["github.com/src-d/go-git"],
        "account": 1,
        "branches": "master",
        "tags": ".*",
        "match": ReleaseMatchStrategy.TAG,
    }
    response1 = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body1)
    await sdb.execute(delete(AccountGitHubAccount)
                      .where(AccountGitHubAccount.account_id == 1))
    await sdb.execute(insert(AccountGitHubAccount).values(
        AccountGitHubAccount(id=6366825, account_id=2).explode(with_primary_keys=True)))
    await sdb.execute(update(UserAccount).where(UserAccount.account_id == 2).values(
        {UserAccount.is_admin.name: True}))
    body2 = {
        "repositories": ["github.com/src-d/go-git"],
        "account": 2,
        "branches": ".*",
        "tags": "v.*",
        "match": ReleaseMatchStrategy.BRANCH,
    }
    response2 = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body2)
    await validate_settings(body1, response1, sdb, False)
    await validate_settings(body2, response2, sdb, False)


async def test_set_release_match_default_user(client, headers):
    body1 = {
        "repositories": ["github.com/src-d/go-git"],
        "account": 1,
        "branches": "master",
        "tags": ".*",
        "match": ReleaseMatchStrategy.TAG,
    }
    response = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body1)
    assert response.status == 403


@pytest.mark.parametrize("code, account, repositories, branches, tags, match", [
    (400, 1, ["{1}"], None, ".*", ReleaseMatchStrategy.BRANCH),
    (400, 1, ["{1}"], "", ".*", ReleaseMatchStrategy.TAG_OR_BRANCH),
    (200, 1, ["{1}"], "", ".*", ReleaseMatchStrategy.TAG),
    (200, 1, [], "", ".*", ReleaseMatchStrategy.TAG),
    (400, 1, ["{1}"], None, ".*", ReleaseMatchStrategy.TAG),
    (400, 1, ["{1}"], "", ".*", ReleaseMatchStrategy.BRANCH),
    (400, 1, ["{1}"], "(f", ".*", ReleaseMatchStrategy.BRANCH),
    (400, 1, ["{1}"], ".*", None, ReleaseMatchStrategy.TAG),
    (400, 1, ["{1}"], ".*", None, ReleaseMatchStrategy.BRANCH),
    (200, 1, ["{1}"], ".*", "", ReleaseMatchStrategy.BRANCH),
    (400, 1, ["{1}"], ".*", "", ReleaseMatchStrategy.TAG),
    (400, 1, ["{1}"], ".*", "", ReleaseMatchStrategy.TAG_OR_BRANCH),
    (400, 1, ["{1}"], ".*", "(f", ReleaseMatchStrategy.TAG),
    (422, 2, ["{1}"], ".*", ".*", ReleaseMatchStrategy.BRANCH),
    (403, 2, ["github.com/athenianco/athenian-api"], ".*", ".*", ReleaseMatchStrategy.BRANCH),
    (404, 3, ["{1}"], ".*", ".*", ReleaseMatchStrategy.BRANCH),
    (403, 1, ["{2}"], ".*", ".*", ReleaseMatchStrategy.BRANCH),
    (400, 1, ["{1}"], ".*", ".*", "whatever"),
    (400, 1, ["{1}"], ".*", ".*", None),
    (400, 1, ["{1}"], ".*", ".*", ""),
])
async def test_set_release_match_nasty_input(
        client, headers, sdb, code, account, repositories, branches, tags, match,
        disable_default_user):
    if account == 2 and code == 403:
        await sdb.execute(insert(AccountGitHubAccount).values({
            AccountGitHubAccount.id: 1,
            AccountGitHubAccount.account_id: 2,
        }))
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


async def cleanup_gkwillie(sdb, with_accounts: bool):
    await sdb.execute(delete(RepositorySet))
    if with_accounts:
        await sdb.execute(delete(AccountGitHubAccount))
    await sdb.execute(insert(UserAccount).values(UserAccount(
        user_id="github|60340680", account_id=1, is_admin=True,
    ).create_defaults().explode(with_primary_keys=True)))


async def test_set_release_match_login_failure(
        client, headers, sdb, lazy_gkwillie, disable_default_user):
    await cleanup_gkwillie(sdb, False)
    await sdb.execute(delete(AccountGitHubAccount))
    body = {
        "repositories": [],
        "account": 1,
        "branches": ".*",
        "tags": ".*",
        "match": ReleaseMatchStrategy.TAG_OR_BRANCH,
    }
    auth.GracefulExit = ValueError
    try:
        response = await client.request(
            method="PUT", path="/v1/settings/release_match", headers=headers, json=body)
    finally:
        auth.GracefulExit = GracefulExit
    assert response.status == 403, await response.read()


@pytest.mark.parametrize("code", [200, 422])
async def test_set_release_match_422(
        client, headers, sdb, mdb_rw, gkwillie, disable_default_user, code):
    mdb = mdb_rw
    await cleanup_gkwillie(sdb, True)
    if code == 422:
        await mdb.execute(update(NodeUser)
                          .where(NodeUser.login == "gkwillie")
                          .values({NodeUser.login: "gkwillie2"}))
    try:
        body = {
            "repositories": [],
            "account": 1,
            "branches": ".*",
            "tags": ".*",
            "match": ReleaseMatchStrategy.TAG_OR_BRANCH,
        }
        response = await client.request(
            method="PUT", path="/v1/settings/release_match", headers=headers, json=body)
        assert response.status == code, await response.read()
    finally:
        if code == 422:
            await mdb.execute(update(NodeUser)
                              .where(NodeUser.login == "gkwillie2")
                              .values({NodeUser.login: "gkwillie"}))


async def test_get_release_match_settings_defaults(client, headers):
    response = await client.request(
        method="GET", path="/v1/settings/release_match/1", headers=headers)
    assert response.status == 200
    settings = {}
    for k, v in json.loads((await response.read()).decode("utf-8")).items():
        if v["default_branch"] is None:
            v["default_branch"] = "whatever"
        settings[k] = ReleaseMatchSetting.from_dict(v)
    defset = ReleaseMatchSetting(
        branches="{{default}}",
        tags=".*",
        match=ReleaseMatchStrategy.TAG_OR_BRANCH,
        default_branch="master",
    )
    assert settings["github.com/src-d/go-git"] == defset
    for k, v in settings.items():
        if v.default_branch == "whatever":
            v.default_branch = "master"
        assert v == defset, k


async def test_get_release_match_settings_existing(client, headers, sdb):
    await sdb.execute(insert(ReleaseSetting).values(
        ReleaseSetting(repository="github.com/src-d/go-git",
                       account_id=1,
                       branches="master",
                       tags="v.*",
                       match=1).create_defaults().explode(with_primary_keys=True)))
    response = await client.request(
        method="GET", path="/v1/settings/release_match/1", headers=headers)
    assert response.status == 200
    settings = {}
    for k, v in json.loads((await response.read()).decode("utf-8")).items():
        if k != "github.com/src-d/go-git":
            v["default_branch"] = "whatever"
        settings[k] = ReleaseMatchSetting.from_dict(v)
    assert settings["github.com/src-d/go-git"] == ReleaseMatchSetting(
        branches="master",
        tags="v.*",
        match=ReleaseMatchStrategy.TAG,
        default_branch="master",
    )
    defset = ReleaseMatchSetting(
        branches="{{default}}",
        tags=".*",
        match=ReleaseMatchStrategy.TAG_OR_BRANCH,
        default_branch="whatever",
    )
    del settings["github.com/src-d/go-git"]
    for k, v in settings.items():
        assert v == defset, k


@pytest.mark.parametrize("account, code", [(2, 422), (3, 404)])
async def test_get_release_match_settings_nasty_input(client, headers, sdb, account, code):
    await sdb.execute(delete(RepositorySet))
    response = await client.request(
        method="GET", path="/v1/settings/release_match/%d" % account, headers=headers)
    assert response.status == code


JIRA_PROJECTS = [
    JIRAProject(name="Content", key="CON", enabled=True,
                avatar_url="https://athenianco.atlassian.net/secure/projectavatar?pid=10013&avatarId=10424"),  # noqa
    JIRAProject(name="Customer Success", key="CS", enabled=True,
                avatar_url="https://athenianco.atlassian.net/secure/projectavatar?pid=10012&avatarId=10419"),  # noqa
    JIRAProject(name="Product Development", key="DEV", enabled=False,
                avatar_url="https://athenianco.atlassian.net/secure/projectavatar?pid=10009&avatarId=10551"),  # noqa
    JIRAProject(name="Engineering", key="ENG", enabled=True,
                avatar_url="https://athenianco.atlassian.net/secure/projectavatar?pid=10003&avatarId=10404"),  # noqa
    JIRAProject(name="Growth", key="GRW", enabled=True,
                avatar_url="https://athenianco.atlassian.net/secure/projectavatar?pid=10008&avatarId=10419"),  # noqa
    JIRAProject(name="Operations", key="OPS", enabled=True,
                avatar_url="https://athenianco.atlassian.net/secure/projectavatar?pid=10002&avatarId=10421"),  # noqa
    JIRAProject(name="Product", key="PRO", enabled=True,
                avatar_url="https://athenianco.atlassian.net/secure/projectavatar?pid=10001&avatarId=10414"),  # noqa
]


async def test_get_jira_projects_smoke(client, headers, disabled_dev):
    response = await client.request(
        method="GET", path="/v1/settings/jira/projects/1", headers=headers)
    assert response.status == 200
    body = [JIRAProject.from_dict(i) for i in json.loads((await response.read()).decode("utf-8"))]
    assert body == JIRA_PROJECTS


@pytest.mark.parametrize("account, code", [[2, 422], [3, 404], [4, 404]])
async def test_get_jira_projects_nasty_input(client, headers, account, code):
    response = await client.request(
        method="GET", path="/v1/settings/jira/projects/%d" % account, headers=headers)
    assert response.status == code


async def test_set_jira_projects_smoke(client, headers, disable_default_user):
    body = {
        "account": 1,
        "projects": {
            "DEV": False,
        },
    }
    response = await client.request(
        method="PUT", path="/v1/settings/jira/projects", json=body, headers=headers)
    assert response.status == 200
    body = [JIRAProject.from_dict(i) for i in json.loads((await response.read()).decode("utf-8"))]
    assert body == JIRA_PROJECTS


async def test_set_jira_projects_default_user(client, headers):
    body = {
        "account": 1,
        "projects": {
            "DEV": False,
        },
    }
    response = await client.request(
        method="PUT", path="/v1/settings/jira/projects", json=body, headers=headers)
    assert response.status == 403


@pytest.mark.parametrize("account, key, code", [[2, "DEV", 403], [3, "DEV", 404], [1, "XXX", 400]])
async def test_set_jira_projects_nasty_input(
        client, headers, disable_default_user, account, key, code):
    body = {
        "account": account,
        "projects": {
            key: False,
        },
    }
    response = await client.request(
        method="PUT", path="/v1/settings/jira/projects", json=body, headers=headers)
    assert response.status == code


async def test_get_jira_identities_smoke(client, headers, sdb, denys_id_mapping):
    response = await client.request(
        method="GET", path="/v1/settings/jira/identities/1", headers=headers)
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    ids = [WebMappedJIRAIdentity.from_dict(item) for item in FriendlyJson.loads(body)]
    assert len(ids) == 16
    has_match = False
    for user_map in ids:
        match = user_map == WebMappedJIRAIdentity(developer_id="github.com/dennwc",
                                                  developer_name="Denys Smirnov",
                                                  jira_name="Denys Smirnov",
                                                  confidence=1.0)
        unmapped = user_map.developer_id is None and user_map.developer_name is None
        assert match or unmapped
        has_match |= match
    assert has_match


async def test_get_jira_identities_empty(client, headers):
    response = await client.request(
        method="GET", path="/v1/settings/jira/identities/1", headers=headers)
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    ids = [WebMappedJIRAIdentity.from_dict(item) for item in FriendlyJson.loads(body)]
    assert len(ids) == 16
    for user_map in ids:
        assert user_map.developer_id is None
        assert user_map.developer_name is None
        assert user_map.jira_name is not None


@pytest.mark.parametrize("account, code", [[2, 422], [3, 404], [4, 404]])
async def test_get_jira_identities_nasty_input(client, headers, account, code):
    response = await client.request(
        method="GET", path="/v1/settings/jira/identities/%d" % account, headers=headers)
    assert response.status == code


async def test_set_jira_identities_smoke(client, headers, sdb, denys_id_mapping):
    body = {
        "account": 1,
        "changes": [{
            "developer_id": "github.com/dennwc",
            "jira_name": "Vadim Markovtsev",
        }],
    }
    response = await client.request(
        method="PATCH", path="/v1/settings/jira/identities", headers=headers, json=body)
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    ids = [WebMappedJIRAIdentity.from_dict(item) for item in FriendlyJson.loads(body)]
    assert WebMappedJIRAIdentity(developer_id="github.com/dennwc",
                                 developer_name="Denys Smirnov",
                                 jira_name="Vadim Markovtsev",
                                 confidence=1.0) in ids
    assert len(ids) == 16
    rows = await sdb.fetch_all(select([MappedJIRAIdentity]))
    assert len(rows) == 1
    assert rows[0][MappedJIRAIdentity.account_id.name] == 1
    assert rows[0][MappedJIRAIdentity.github_user_id.name] == 40294
    assert rows[0][MappedJIRAIdentity.jira_user_id.name] == "5de5049e2c5dd20d0f9040c1"
    assert rows[0][MappedJIRAIdentity.confidence.name] == 1


async def test_set_jira_identities_reset_cache(client, headers, denys_id_mapping, client_cache):
    async def fetch_contribs():
        return sorted(json.loads(
            (await (await client.get(path="/v1/get/contributors/1", headers=headers)).read())
            .decode("utf-8")), key=lambda u: u["login"])

    contribs1 = await fetch_contribs()
    contribs2 = await fetch_contribs()
    assert contribs1 == contribs2
    body = {
        "account": 1,
        "changes": [{
            "developer_id": "github.com/dennwc",
            "jira_name": "Vadim Markovtsev",
        }],
    }
    response = await client.request(
        method="PATCH", path="/v1/settings/jira/identities", headers=headers, json=body)
    assert response.status == 200
    contribs2 = await fetch_contribs()
    assert contribs1 != contribs2
    contribs3 = await fetch_contribs()
    assert contribs2 == contribs3

    body = {
        "account": 1,
        "changes": [{
            "developer_id": "github.com/dennwc",
            "jira_name": None,
        }],
    }
    response = await client.request(
        method="PATCH", path="/v1/settings/jira/identities", headers=headers, json=body)
    assert response.status == 200
    contribs2 = await fetch_contribs()
    assert contribs1 != contribs2
    assert contribs2 != contribs3
    contribs3 = await fetch_contribs()
    assert contribs2 == contribs3


async def test_set_jira_identities_delete(client, headers, sdb, denys_id_mapping):
    body = {
        "account": 1,
        "changes": [{
            "developer_id": "github.com/dennwc",
            "jira_name": None,
        }],
    }
    response = await client.request(
        method="PATCH", path="/v1/settings/jira/identities", headers=headers, json=body)
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    ids = [WebMappedJIRAIdentity.from_dict(item) for item in FriendlyJson.loads(body)]
    assert len(ids) == 16
    for user_map in ids:
        assert user_map.developer_id is None
        assert user_map.developer_name is None
        assert user_map.jira_name is not None
    rows = await sdb.fetch_all(select([MappedJIRAIdentity]))
    assert len(rows) == 0


@pytest.mark.parametrize("account, github, jira, code", [
    [2, "github.com/vmarkovtsev", "Vadim Markovtsev", 403],
    [2, "github.com/vmarkovtsev", "Vadim Markovtsev", 422],
    [3, "github.com/vmarkovtsev", "Vadim Markovtsev", 404],
    [4, "github.com/vmarkovtsev", "Vadim Markovtsev", 404],
    [1, None, "Vadim Markovtsev", 400],
    [1, "", "Vadim Markovtsev", 400],
    [1, "github.com", "Vadim Markovtsev", 400],
    [1, "github.com/incognito", "Vadim Markovtsev", 400],
    [1, "github.com/vmarkovtsev", "Vadim", 400],
    [1, "github.com/vmarkovtsev", "", 400],
])
async def test_set_jira_identities_nasty_input(client, headers, account, github, jira, code, sdb):
    if account == 2 and code == 422:
        await sdb.execute(update(UserAccount)
                          .where(UserAccount.account_id == 2)
                          .values({UserAccount.is_admin: True}))
    body = {
        "account": account,
        "changes": [{
            "developer_id": github,
            "jira_name": jira,
        }],
    }
    response = await client.request(
        method="PATCH", path="/v1/settings/jira/identities", json=body, headers=headers)
    assert response.status == code
