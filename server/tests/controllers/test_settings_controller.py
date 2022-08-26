from datetime import datetime, timezone
import json

from aiohttp.web_runner import GracefulExit
import pytest
from sqlalchemy import and_, delete, insert, select, update

from athenian.api import auth
from athenian.api.async_utils import read_sql_query
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.logical_repos import (
    coerce_logical_repos,
    contains_logical_repos,
    drop_logical_repo,
)
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseMatch, Settings
from athenian.api.models.metadata.github import PullRequest
from athenian.api.models.precomputed.models import (
    GitHubDeploymentFacts,
    GitHubDonePullRequestFacts,
    GitHubOpenPullRequestFacts,
    GitHubPullRequestDeployment,
    GitHubRelease,
    GitHubReleaseDeployment,
)
from athenian.api.models.state.models import (
    AccountGitHubAccount,
    LogicalRepository,
    MappedJIRAIdentity,
    ReleaseSetting,
    RepositorySet,
    UserAccount,
    WorkType,
)
from athenian.api.models.web import (
    JIRAProject,
    LogicalRepository as WebLogicalRepository,
    MappedJIRAIdentity as WebMappedJIRAIdentity,
    ReleaseMatchSetting,
    ReleaseMatchStrategy,
    WorkType as WebWorkType,
)
from athenian.api.response import ResponseError
from athenian.api.serialization import FriendlyJson
from tests.testutils.db import assert_missing_row, model_insert_stmt
from tests.testutils.factory.precomputed import (
    GitHubDonePullRequestFactsFactory,
    GitHubOpenPullRequestFactsFactory,
    GitHubReleaseFactory,
)
from tests.testutils.factory.state import LogicalRepositoryFactory


async def validate_release_settings(body, response, sdb, exhaustive: bool):
    assert response.status == 200
    repos = json.loads((await response.read()).decode("utf-8"))
    assert len(repos) > 0
    assert repos[0].startswith("github.com/")
    df = await read_sql_query(
        select([ReleaseSetting]),
        sdb,
        ReleaseSetting,
        index=[ReleaseSetting.repository.name, ReleaseSetting.account_id.name],
    )
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
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body,
    )
    repos = await validate_release_settings(body, response, sdb, True)
    assert repos == ["github.com/src-d/gitbase", "github.com/src-d/go-git"]
    body.update(
        {
            "branches": ".*",
            "tags": "v.*",
            "match": ReleaseMatchStrategy.BRANCH,
        },
    )
    response = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body,
    )
    repos = await validate_release_settings(body, response, sdb, True)
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
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body1,
    )
    await sdb.execute(delete(AccountGitHubAccount).where(AccountGitHubAccount.account_id == 1))
    await sdb.execute(
        insert(AccountGitHubAccount).values(
            AccountGitHubAccount(id=6366825, account_id=2).explode(with_primary_keys=True),
        ),
    )
    await sdb.execute(
        update(UserAccount)
        .where(UserAccount.account_id == 2)
        .values({UserAccount.is_admin.name: True}),
    )
    body2 = {
        "repositories": ["github.com/src-d/go-git"],
        "account": 2,
        "branches": ".*",
        "tags": "v.*",
        "match": ReleaseMatchStrategy.BRANCH,
    }
    response2 = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body2,
    )
    await validate_release_settings(body1, response1, sdb, False)
    await validate_release_settings(body2, response2, sdb, False)


async def test_set_release_match_default_user(client, headers):
    body1 = {
        "repositories": ["github.com/src-d/go-git"],
        "account": 1,
        "branches": "master",
        "tags": ".*",
        "match": ReleaseMatchStrategy.TAG,
    }
    response = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body1,
    )
    assert response.status == 403


@pytest.mark.parametrize(
    "code, account, repositories, branches, tags, events, match",
    [
        (400, 1, ["{1}"], None, ".*", {}, ReleaseMatchStrategy.BRANCH),
        (400, 1, ["{1}"], "", ".*", {}, ReleaseMatchStrategy.TAG_OR_BRANCH),
        (200, 1, ["{1}"], "", ".*", {}, ReleaseMatchStrategy.TAG),
        (200, 1, [], "", ".*", {}, ReleaseMatchStrategy.TAG),
        (400, 1, ["{1}"], None, ".*", {}, ReleaseMatchStrategy.TAG),
        (400, 1, ["{1}"], "", ".*", {}, ReleaseMatchStrategy.BRANCH),
        (400, 1, ["{1}"], "(f", ".*", {}, ReleaseMatchStrategy.BRANCH),
        (400, 1, ["{1}"], ".*", None, {}, ReleaseMatchStrategy.TAG),
        (400, 1, ["{1}"], ".*", None, {}, ReleaseMatchStrategy.BRANCH),
        (200, 1, ["{1}"], ".*", "", {}, ReleaseMatchStrategy.BRANCH),
        (400, 1, ["{1}"], ".*", "", {}, ReleaseMatchStrategy.TAG),
        (400, 1, ["{1}"], ".*", "", {}, ReleaseMatchStrategy.TAG_OR_BRANCH),
        (400, 1, ["{1}"], ".*", "(f", {}, ReleaseMatchStrategy.TAG),
        (422, 2, ["{1}"], ".*", ".*", {}, ReleaseMatchStrategy.BRANCH),
        (
            403,
            2,
            ["github.com/athenianco/athenian-api"],
            ".*",
            ".*",
            {},
            ReleaseMatchStrategy.BRANCH,
        ),
        (404, 3, ["{1}"], ".*", ".*", {}, ReleaseMatchStrategy.BRANCH),
        (403, 1, ["{2}"], ".*", ".*", {}, ReleaseMatchStrategy.BRANCH),
        (400, 1, ["{1}"], ".*", ".*", {}, "whatever"),
        (400, 1, ["{1}"], ".*", ".*", {}, None),
        (400, 1, ["{1}"], ".*", ".*", {}, ""),
        (400, 1, [], "", ".*", {"events": "(f"}, ReleaseMatchStrategy.TAG),
    ],
)
async def test_set_release_match_nasty_input(
    client,
    headers,
    sdb,
    code,
    account,
    repositories,
    branches,
    tags,
    events,
    match,
    disable_default_user,
):
    if account == 2 and code == 403:
        await sdb.execute(
            insert(AccountGitHubAccount).values(
                {
                    AccountGitHubAccount.id: 1,
                    AccountGitHubAccount.account_id: 2,
                },
            ),
        )
    body = {
        "repositories": repositories,
        "account": account,
        "branches": branches,
        "tags": tags,
        "match": match,
        **events,
    }
    response = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body,
    )
    assert response.status == code


async def cleanup_gkwillie(sdb):
    await sdb.execute(
        insert(UserAccount).values(
            UserAccount(user_id="github|60340680", account_id=1, is_admin=True)
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )


async def test_set_release_match_login_failure(
    client,
    headers,
    sdb,
    lazy_gkwillie,
    disable_default_user,
):
    await cleanup_gkwillie(sdb)
    await sdb.execute(delete(RepositorySet))
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
            method="PUT", path="/v1/settings/release_match", headers=headers, json=body,
        )
    finally:
        auth.GracefulExit = GracefulExit
    assert response.status == 403, await response.read()


@pytest.mark.parametrize(
    "code, clear_kind",
    [
        (200, None),
        (200, "reposets"),
        (422, "account"),
        (422, "repos"),
    ],
)
async def test_set_release_match_422(
    client,
    headers,
    sdb,
    gkwillie,
    disable_default_user,
    code,
    clear_kind,
):
    await cleanup_gkwillie(sdb)
    if clear_kind == "reposets":
        await sdb.execute(delete(RepositorySet))
    elif clear_kind in ("account", "repos"):
        await sdb.execute(delete(AccountGitHubAccount))
    body = {
        "repositories": ["github.com/src-d/go-git"] if clear_kind == "repos" else [],
        "account": 1,
        "branches": ".*",
        "tags": ".*",
        "match": ReleaseMatchStrategy.TAG_OR_BRANCH,
    }
    response = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body,
    )
    assert response.status == code, await response.read()


async def test_set_release_match_logical(
    client,
    headers,
    sdb,
    disable_default_user,
    release_match_setting_tag_logical_db,
):
    body = {
        "repositories": ["github.com/src-d/go-git/alpha"],
        "account": 1,
        "branches": "master",
        "tags": ".*",
        "match": ReleaseMatchStrategy.EVENT,
    }
    response = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body,
    )
    assert response.status == 200, await response.read()
    match = await sdb.fetch_val(
        select([ReleaseSetting.match]).where(
            ReleaseSetting.repository == "github.com/src-d/go-git/alpha",
        ),
    )
    assert match == 3


async def test_set_release_match_logical_fail(
    client,
    headers,
    sdb,
    disable_default_user,
    release_match_setting_tag_logical_db,
):
    body = {
        "repositories": ["github.com/src-d/go-git/alpha"],
        "account": 1,
        "branches": "master",
        "tags": ".*",
        "match": ReleaseMatchStrategy.TAG_OR_BRANCH,
    }
    response = await client.request(
        method="PUT", path="/v1/settings/release_match", headers=headers, json=body,
    )
    assert response.status == 400, await response.read()


async def test_get_release_match_settings_defaults(client, headers):
    response = await client.request(
        method="GET", path="/v1/settings/release_match/1", headers=headers,
    )
    assert response.status == 200
    settings = {}
    for k, v in json.loads((await response.read()).decode("utf-8")).items():
        if v["default_branch"] is None:
            v["default_branch"] = "whatever"
        settings[k] = ReleaseMatchSetting.from_dict(v)
    defset = ReleaseMatchSetting(
        branches="{{default}}",
        tags=".*",
        events=".*",
        match=ReleaseMatchStrategy.TAG_OR_BRANCH,
        default_branch="master",
    )
    assert settings["github.com/src-d/go-git"] == defset
    for k, v in settings.items():
        if v.default_branch == "whatever":
            v.default_branch = "master"
        assert v == defset, k


async def test_get_release_match_settings_existing(client, headers, sdb):
    await sdb.execute(
        insert(ReleaseSetting).values(
            ReleaseSetting(
                repository="github.com/src-d/go-git",
                account_id=1,
                branches="master",
                tags="v.*",
                events=".*",
                match=ReleaseMatch.tag,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    response = await client.request(
        method="GET", path="/v1/settings/release_match/1", headers=headers,
    )
    assert response.status == 200
    settings = {}
    for k, v in json.loads((await response.read()).decode("utf-8")).items():
        if k != "github.com/src-d/go-git":
            v["default_branch"] = "whatever"
        settings[k] = ReleaseMatchSetting.from_dict(v)
    assert settings["github.com/src-d/go-git"] == ReleaseMatchSetting(
        branches="master",
        tags="v.*",
        events=".*",
        match=ReleaseMatchStrategy.TAG,
        default_branch="master",
    )
    defset = ReleaseMatchSetting(
        branches="{{default}}",
        tags=".*",
        events=".*",
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
        method="GET", path="/v1/settings/release_match/%d" % account, headers=headers,
    )
    assert response.status == code


async def test_get_release_match_settings_logical_fail(sdb, mdb, logical_settings_db):
    settings = Settings.from_account(1, sdb, mdb, None, None)
    with pytest.raises(ResponseError, match="424"):
        await settings.list_release_matches(["github.com/src-d/go-git/alpha"])


async def test_get_release_match_settings_logical_success(
    sdb,
    mdb,
    logical_settings_db,
    release_match_setting_tag_logical_db,
):
    settings = Settings.from_account(1, sdb, mdb, None, None)
    await settings.list_release_matches(["github.com/src-d/go-git/alpha"])


JIRA_PROJECTS = [
    JIRAProject(
        name="Content",
        key="CON",
        id="10013",
        enabled=True,
        last_update=None,
        issues_count=0,
        avatar_url=(
            "https://athenianco.atlassian.net/secure/projectavatar?pid=10013&avatarId=10424"
        ),
    ),  # noqa
    JIRAProject(
        name="Customer Success",
        key="CS",
        id="10012",
        enabled=True,
        last_update=None,
        issues_count=0,
        avatar_url=(
            "https://athenianco.atlassian.net/secure/projectavatar?pid=10012&avatarId=10419"
        ),
    ),  # noqa
    JIRAProject(
        name="Product Development",
        key="DEV",
        id="10009",
        enabled=False,
        issues_count=1001,
        last_update=datetime(2020, 10, 22, 11, 0, 9, tzinfo=timezone.utc),
        avatar_url=(
            "https://athenianco.atlassian.net/secure/projectavatar?pid=10009&avatarId=10551"
        ),
    ),  # noqa
    JIRAProject(
        name="Engineering",
        key="ENG",
        id="10003",
        enabled=True,
        issues_count=862,
        last_update=datetime(2020, 9, 1, 13, 7, 56, tzinfo=timezone.utc),
        avatar_url=(
            "https://athenianco.atlassian.net/secure/projectavatar?pid=10003&avatarId=10404"
        ),
    ),  # noqa
    JIRAProject(
        name="Growth",
        key="GRW",
        id="10008",
        enabled=True,
        last_update=None,
        issues_count=0,
        avatar_url=(
            "https://athenianco.atlassian.net/secure/projectavatar?pid=10008&avatarId=10419"
        ),
    ),  # noqa
    JIRAProject(
        name="Operations",
        key="OPS",
        id="10002",
        enabled=True,
        last_update=None,
        issues_count=0,
        avatar_url=(
            "https://athenianco.atlassian.net/secure/projectavatar?pid=10002&avatarId=10421"
        ),
    ),  # noqa
    JIRAProject(
        name="Product",
        key="PRO",
        id="10001",
        enabled=True,
        last_update=None,
        issues_count=0,
        avatar_url=(
            "https://athenianco.atlassian.net/secure/projectavatar?pid=10001&avatarId=10414"
        ),
    ),  # noqa
]


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_get_jira_projects_smoke(client, headers, disabled_dev):
    response = await client.request(
        method="GET", path="/v1/settings/jira/projects/1", headers=headers,
    )
    assert response.status == 200
    body = [JIRAProject.from_dict(i) for i in json.loads((await response.read()).decode("utf-8"))]
    assert body == JIRA_PROJECTS


@pytest.mark.parametrize("account, code", [[2, 422], [3, 404], [4, 404]])
async def test_get_jira_projects_nasty_input(client, headers, account, code):
    response = await client.request(
        method="GET", path="/v1/settings/jira/projects/%d" % account, headers=headers,
    )
    assert response.status == code


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_set_jira_projects_smoke(client, headers, disable_default_user):
    body = {
        "account": 1,
        "projects": {
            "DEV": False,
        },
    }
    response = await client.request(
        method="PUT", path="/v1/settings/jira/projects", json=body, headers=headers,
    )
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
        method="PUT", path="/v1/settings/jira/projects", json=body, headers=headers,
    )
    assert response.status == 403


@pytest.mark.parametrize("account, key, code", [[2, "DEV", 403], [3, "DEV", 404], [1, "XXX", 400]])
async def test_set_jira_projects_nasty_input(
    client,
    headers,
    disable_default_user,
    account,
    key,
    code,
):
    body = {
        "account": account,
        "projects": {
            key: False,
        },
    }
    response = await client.request(
        method="PUT", path="/v1/settings/jira/projects", json=body, headers=headers,
    )
    assert response.status == code


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_get_jira_identities_smoke(client, headers, sdb, denys_id_mapping):
    response = await client.request(
        method="GET", path="/v1/settings/jira/identities/1", headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    ids = [WebMappedJIRAIdentity.from_dict(item) for item in FriendlyJson.loads(body)]
    assert len(ids) == 16
    has_match = False
    for user_map in ids:
        match = user_map == WebMappedJIRAIdentity(
            developer_id="github.com/dennwc",
            developer_name="Denys Smirnov",
            jira_name="Denys Smirnov",
            confidence=1.0,
        )
        unmapped = user_map.developer_id is None and user_map.developer_name is None
        assert match or unmapped
        has_match |= match
    assert has_match


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_get_jira_identities_empty(client, headers):
    response = await client.request(
        method="GET", path="/v1/settings/jira/identities/1", headers=headers,
    )
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
        method="GET", path="/v1/settings/jira/identities/%d" % account, headers=headers,
    )
    assert response.status == code


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_set_jira_identities_smoke(client, headers, sdb, denys_id_mapping):
    body = {
        "account": 1,
        "changes": [
            {
                "developer_id": "github.com/dennwc",
                "jira_name": "Vadim Markovtsev",
            },
        ],
    }
    response = await client.request(
        method="PATCH", path="/v1/settings/jira/identities", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    ids = [WebMappedJIRAIdentity.from_dict(item) for item in FriendlyJson.loads(body)]
    assert (
        WebMappedJIRAIdentity(
            developer_id="github.com/dennwc",
            developer_name="Denys Smirnov",
            jira_name="Vadim Markovtsev",
            confidence=1.0,
        )
        in ids
    )
    assert len(ids) == 16
    rows = await sdb.fetch_all(select([MappedJIRAIdentity]))
    assert len(rows) == 1
    assert rows[0][MappedJIRAIdentity.account_id.name] == 1
    assert rows[0][MappedJIRAIdentity.github_user_id.name] == 40294
    assert rows[0][MappedJIRAIdentity.jira_user_id.name] == "5de5049e2c5dd20d0f9040c1"
    assert rows[0][MappedJIRAIdentity.confidence.name] == 1


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_set_jira_identities_reset_cache(client, headers, denys_id_mapping, client_cache):
    async def fetch_contribs():
        return sorted(
            json.loads(
                (
                    await (await client.get(path="/v1/get/contributors/1", headers=headers)).read()
                ).decode("utf-8"),
            ),
            key=lambda u: u["login"],
        )

    contribs1 = await fetch_contribs()
    contribs2 = await fetch_contribs()
    assert contribs1 == contribs2
    body = {
        "account": 1,
        "changes": [
            {
                "developer_id": "github.com/dennwc",
                "jira_name": "Vadim Markovtsev",
            },
        ],
    }
    response = await client.request(
        method="PATCH", path="/v1/settings/jira/identities", headers=headers, json=body,
    )
    assert response.status == 200
    contribs2 = await fetch_contribs()
    assert contribs1 != contribs2
    contribs3 = await fetch_contribs()
    assert contribs2 == contribs3

    body = {
        "account": 1,
        "changes": [
            {
                "developer_id": "github.com/dennwc",
                "jira_name": None,
            },
        ],
    }
    response = await client.request(
        method="PATCH", path="/v1/settings/jira/identities", headers=headers, json=body,
    )
    assert response.status == 200, (await response.read()).decode()
    contribs2 = await fetch_contribs()
    assert contribs1 != contribs2
    assert contribs2 != contribs3
    contribs3 = await fetch_contribs()
    assert contribs2 == contribs3


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_set_jira_identities_delete(client, headers, sdb, denys_id_mapping):
    body = {
        "account": 1,
        "changes": [
            {
                "developer_id": "github.com/dennwc",
                "jira_name": None,
            },
        ],
    }
    response = await client.request(
        method="PATCH", path="/v1/settings/jira/identities", headers=headers, json=body,
    )
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


@pytest.mark.parametrize(
    "account, github, jira, code",
    [
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
    ],
)
async def test_set_jira_identities_nasty_input(client, headers, account, github, jira, code, sdb):
    if account == 2 and code == 422:
        await sdb.execute(
            update(UserAccount)
            .where(UserAccount.account_id == 2)
            .values({UserAccount.is_admin: True}),
        )
    body = {
        "account": account,
        "changes": [
            {
                "developer_id": github,
                "jira_name": jira,
            },
        ],
    }
    response = await client.request(
        method="PATCH", path="/v1/settings/jira/identities", json=body, headers=headers,
    )
    assert response.status == code


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_get_work_type_smoke(client, headers):
    body = {
        "account": 1,
        "name": "Bug Fixing",
    }
    response = await client.request(
        method="POST", path="/v1/settings/work_type/get", json=body, headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    wt = WebWorkType.from_dict(FriendlyJson.loads(body))
    assert wt.to_dict() == {
        "color": "FF0000",
        "name": "Bug Fixing",
        "rules": [{"body": ["bug", "fix"], "name": "pull_request/label_include"}],
    }


@pytest.mark.parametrize(
    "body, status",
    [
        ({"account": 2, "name": "Bug Fixing"}, 404),
        ({"account": 3, "name": "Bug Fixing"}, 404),
        ({"account": 1, "name": "Bug Making"}, 404),
        ({"account": 1, "name": ""}, 400),
        ({"account": 1}, 400),
    ],
)
async def test_get_work_type_nasty_input(client, headers, body, status):
    response = await client.request(
        method="POST", path="/v1/settings/work_type/get", json=body, headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == status, "Response body is : " + body


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_delete_work_type_smoke(client, headers, sdb):
    body = {
        "account": 1,
        "name": "Bug Fixing",
    }
    response = await client.request(
        method="POST", path="/v1/settings/work_type/delete", json=body, headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    assert body == ""
    row = await sdb.fetch_one(
        select([WorkType]).where(
            and_(
                WorkType.account_id == 1,
                WorkType.name == "Bug Fixing",
            ),
        ),
    )
    assert row is None


@pytest.mark.parametrize(
    "body, status",
    [
        ({"account": 2, "name": "Bug Fixing"}, 404),
        ({"account": 3, "name": "Bug Fixing"}, 404),
        ({"account": 1, "name": "Bug Making"}, 404),
        ({"account": 1, "name": ""}, 400),
        ({"account": 1}, 400),
    ],
)
async def test_delete_work_type_nasty_input(client, headers, body, status, sdb):
    response = await client.request(
        method="POST", path="/v1/settings/work_type/delete", json=body, headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == status, "Response body is : " + body
    row = await sdb.fetch_one(
        select([WorkType]).where(
            and_(
                WorkType.account_id == 1,
                WorkType.name == "Bug Fixing",
            ),
        ),
    )
    assert row is not None


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_list_work_types_smoke(client, headers):
    response = await client.request(
        method="GET", path="/v1/settings/work_types/1", headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    models = [WebWorkType.from_dict(i) for i in FriendlyJson.loads(body)]
    assert len(models) == 1
    assert models[0].to_dict() == {
        "color": "FF0000",
        "name": "Bug Fixing",
        "rules": [{"body": ["bug", "fix"], "name": "pull_request/label_include"}],
    }


@pytest.mark.parametrize("acc, status", [(2, 200), (3, 404)])
async def test_list_work_types_nasty_input(client, headers, acc, status):
    response = await client.request(
        method="GET", path=f"/v1/settings/work_types/{acc}", headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == status, "Response body is : " + body
    if status == 200:
        assert body == "[]"


async def test_set_work_type_create(client, headers, sdb):
    body = {
        "account": 1,
        "work_type": {
            "name": "Bug Making",
            "color": "00ff00",
            "rules": [
                {
                    "name": "xxx",
                    "body": {"arg": 777},
                },
            ],
        },
    }
    response = await client.request(
        method="PUT", path="/v1/settings/work_type", json=body, headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    wt1 = WebWorkType.from_dict(FriendlyJson.loads(body))
    assert wt1.to_dict() == {
        "color": "00ff00",
        "name": "Bug Making",
        "rules": [{"body": {"arg": 777}, "name": "xxx"}],
    }
    row = await sdb.fetch_one(
        select([WorkType.name, WorkType.color, WorkType.rules]).where(
            and_(
                WorkType.account_id == 1,
                WorkType.name == "Bug Making",
            ),
        ),
    )
    assert dict(row) == {
        "color": "00ff00",
        "name": "Bug Making",
        "rules": [["xxx", {"arg": 777}]],
    }


async def test_set_work_type_update(client, headers, sdb):
    body = {
        "account": 1,
        "work_type": {
            "name": "Bug Fixing",
            "color": "00ff00",
            "rules": [
                {
                    "name": "xxx",
                    "body": {"arg": 777},
                },
            ],
        },
    }
    response = await client.request(
        method="PUT", path="/v1/settings/work_type", json=body, headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    wt1 = WebWorkType.from_dict(FriendlyJson.loads(body))
    assert wt1.to_dict() == {
        "color": "00ff00",
        "name": "Bug Fixing",
        "rules": [{"body": {"arg": 777}, "name": "xxx"}],
    }
    rows = await sdb.fetch_all(
        select([WorkType.name, WorkType.color, WorkType.rules]).where(
            and_(
                WorkType.account_id == 1,
                WorkType.name == "Bug Fixing",
            ),
        ),
    )
    assert len(rows) == 1
    assert dict(rows[0]) == {
        "color": "00ff00",
        "name": "Bug Fixing",
        "rules": [["xxx", {"arg": 777}]],
    }


@pytest.mark.parametrize(
    "acc, name, color, status",
    [
        (3, "Bug Fixing", {"color": "00FF00"}, 404),
        (1, "", {"color": "00FF00"}, 400),
        (1, "Bug Fixing", {}, 400),
    ],
)
async def test_set_work_type_nasty_input(client, headers, sdb, acc, name, color, status):
    body = {
        "account": acc,
        "work_type": {
            "name": name,
            **color,
            "rules": [
                {
                    "name": "xxx",
                    "body": {"arg": 777},
                },
            ],
        },
    }
    response = await client.request(
        method="PUT", path="/v1/settings/work_type", json=body, headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == status, "Response body is : " + body
    rows = await sdb.fetch_all(
        select([WorkType.name, WorkType.color, WorkType.rules]).where(
            and_(WorkType.name == "Bug Fixing"),
        ),
    )
    assert len(rows) == 1
    assert dict(rows[0]) == {
        "color": "FF0000",
        "name": "Bug Fixing",
        "rules": [["pull_request/label_include", ["bug", "fix"]]],
    }


async def test_set_work_type_empty_rules(client, headers, sdb):
    body = {
        "account": 2,
        "work_type": {
            "name": "Bug Fixing",
            "color": "0000ee",
            "rules": [],
        },
    }
    response = await client.request(
        method="PUT", path="/v1/settings/work_type", json=body, headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    rows = await sdb.fetch_all(
        select([WorkType.account_id, WorkType.name, WorkType.color, WorkType.rules]).where(
            and_(WorkType.name == "Bug Fixing"),
        ),
    )
    assert len(rows) == 2
    rows = [row for row in rows if row[WorkType.account_id.name] == 2]
    assert len(rows) == 1
    assert dict(rows[0]) == {
        "account_id": 2,
        "color": "0000ee",
        "name": "Bug Fixing",
        "rules": [],
    }


async def test_drop_logical_repo():
    assert drop_logical_repo("src-d/go-git") == "src-d/go-git"
    assert drop_logical_repo("src-d/go-git/alpha") == "src-d/go-git"
    assert drop_logical_repo("") == ""


async def test_coerce_logical_repos():
    assert coerce_logical_repos(["src-d/go-git"]) == {"src-d/go-git": {"src-d/go-git"}}
    assert coerce_logical_repos(["src-d/go-git/alpha", "src-d/go-git/beta"]) == {
        "src-d/go-git": {"src-d/go-git/alpha", "src-d/go-git/beta"},
    }
    assert coerce_logical_repos([]) == {}


async def test_contains_logical_repos():
    assert not contains_logical_repos([])
    assert not contains_logical_repos(["src-d/go-git"])
    assert contains_logical_repos(["src-d/go-git/"])
    assert contains_logical_repos(["src-d/go-git/alpha"])


@pytest.mark.parametrize("with_title", [False, True])
@pytest.mark.parametrize("with_labels", [False, True])
async def test_logical_settings_smoke(sdb, mdb, prefixer, with_title, with_labels):
    await sdb.execute(
        insert(LogicalRepository).values(
            LogicalRepository(
                account_id=1,
                name="alpha",
                repository_id=40550,
                prs={
                    **({"title": ".*[Ff]ix"} if with_title else {}),
                    **({"labels": ["bug"]} if with_labels else {}),
                },
                deployments={
                    **({"title": "prod"} if with_title else {}),
                    **({"labels": {"repo": ["alpha"]}} if with_labels else {}),
                },
            )
            .create_defaults()
            .explode(),
        ),
    )
    settings = Settings.from_account(1, sdb, mdb, None, None)
    logical_settings = await settings.list_logical_repositories(prefixer)
    any_with = with_labels or with_title
    assert logical_settings.has_logical_prs() == any_with
    assert logical_settings.has_logical_deployments() == any_with
    assert logical_settings.with_logical_prs([]) == (
        {"src-d/go-git", "src-d/go-git/alpha"} if any_with else set()
    )
    assert logical_settings.with_logical_deployments([]) == (
        {"src-d/go-git", "src-d/go-git/alpha"} if any_with else set()
    )
    assert logical_settings.has_prs_by_label(["src-d/go-git"]) == with_labels
    if not any_with:
        with pytest.raises(KeyError):
            logical_settings.prs("src-d/go-git")
        with pytest.raises(KeyError):
            logical_settings.deployments("src-d/go-git")
        return
    repo_settings = logical_settings.prs("src-d/go-git")
    assert repo_settings
    assert repo_settings.has_labels == with_labels
    assert repo_settings.has_titles == with_title
    assert (
        repo_settings.logical_repositories == {"src-d/go-git", "src-d/go-git/alpha"}
        if any_with
        else {"src-d/go-git"}
    )

    repo_settings = logical_settings.deployments("src-d/go-git")
    assert repo_settings
    assert repo_settings.has_labels == with_labels
    assert repo_settings.has_titles == with_title
    assert (
        repo_settings.logical_repositories == {"src-d/go-git", "src-d/go-git/alpha"}
        if any_with
        else {"src-d/go-git"}
    )


@pytest.fixture(scope="function")
async def logical_settings_with_labels(sdb):
    await sdb.execute(
        insert(LogicalRepository).values(
            LogicalRepository(
                account_id=1,
                name="alpha",
                repository_id=40550,
                prs={"title": ".*[Ff]ix", "labels": ["fix"]},
                deployments={"title": "test", "labels": {"repo": ["alpha"]}},
            )
            .create_defaults()
            .explode(),
        ),
    )
    await sdb.execute(
        insert(LogicalRepository).values(
            LogicalRepository(
                account_id=1,
                name="beta",
                repository_id=40550,
                prs={"title": ".*[Aa]dd"},
                deployments={"title": "prod"},
            )
            .create_defaults()
            .explode(),
        ),
    )


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_list_logical_repositories_smoke(
    client,
    headers,
    sdb,
    logical_settings_with_labels,
    release_match_setting_tag_logical_db,
    logical_reposet,
):
    response = await client.request(
        method="GET", path="/v1/settings/logical_repositories/1", headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    repos = [WebLogicalRepository.from_dict(d) for d in json.loads(body)]
    assert len(repos) == 2
    assert repos[0].name == "alpha"
    assert repos[0].parent == "github.com/src-d/go-git"
    assert repos[0].prs.title == ".*[Ff]ix"
    assert repos[0].prs.labels_include == ["fix"]
    assert repos[0].deployments.title == "test"
    assert repos[0].deployments.labels_include == {"repo": ["alpha"]}
    assert repos[0].releases.match == ReleaseMatch.tag.name
    assert repos[0].releases.branches == "master"
    assert repos[0].releases.tags == ".*"
    assert repos[0].releases.events == ".*"
    assert repos[1].name == "beta"
    assert repos[1].parent == "github.com/src-d/go-git"
    assert repos[1].prs.title == ".*[Aa]dd"
    assert repos[1].prs.labels_include is None
    assert repos[1].deployments.title == "prod"
    assert repos[1].deployments.labels_include is None
    assert repos[1].releases.match == ReleaseMatch.tag.name
    assert repos[1].releases.branches == "master"
    assert repos[1].releases.tags == r"v4\..*"
    assert repos[1].releases.events == ".*"


@pytest.mark.parametrize("account, code", [(2, 422), (3, 404)])
async def test_list_logical_repositories_nasty_input(
    client,
    headers,
    sdb,
    logical_settings_with_labels,
    release_match_setting_tag_logical_db,
    logical_reposet,
    account,
    code,
):
    response = await client.request(
        method="GET", path=f"/v1/settings/logical_repositories/{account}", headers=headers,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == code, "Response body is : " + body


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_delete_logical_repository_smoke(
    client,
    headers,
    logical_settings_db,
    release_match_setting_tag_logical_db,
    sdb,
):
    body = {
        "account": 1,
        "name": "github.com/src-d/go-git/alpha",
    }
    response = await client.request(
        method="POST",
        path="/v1/settings/logical_repository/delete",
        headers=headers,
        json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is: " + body
    rows = await sdb.fetch_all(select([LogicalRepository.name]))
    assert len(rows) == 1
    assert rows[0][0] == "beta"
    row = await sdb.fetch_one(select([RepositorySet]).where(RepositorySet.id == 1))
    assert row[RepositorySet.items.name] == [
        ["github.com/src-d/gitbase", 39652769],
        ["github.com/src-d/go-git", 40550],
        ["github.com/src-d/go-git/beta", 40550],
    ]
    row = await sdb.fetch_one(
        select([ReleaseSetting]).where(
            ReleaseSetting.repository == "github.com/src-d/go-git/alpha",
        ),
    )
    assert row is None


@pytest.mark.app_validate_responses(False)
async def test_delete_logical_repository_clean_deployments(
    client,
    headers,
    sdb,
    pdb,
    logical_settings_db,
    release_match_setting_tag_logical_db,
) -> None:
    body = {"account": 1, "name": "github.com/src-d/go-git/alpha"}
    # create some deployments in pdb
    await pdb.execute(
        insert(GitHubReleaseDeployment).values(
            acc_id=1,
            repository_full_name="src-d/go-git/alpha",
            release_match="v1.2.3",
            deployment_name="my-deployment",
            release_id=1,
        ),
    )
    await pdb.execute(
        insert(GitHubPullRequestDeployment).values(
            acc_id=1,
            repository_full_name="src-d/go-git",
            deployment_name="my-deployment2",
            pull_request_id=123,
            finished_at=datetime(2012, 1, 1),
        ),
    )
    await pdb.execute(
        insert(GitHubDeploymentFacts).values(
            acc_id=1,
            deployment_name="my-deployment",
            release_matches="v1.23",
            data=b"",
            format_version=2,
        ),
    )

    response = await client.request(
        method="POST",
        path="/v1/settings/logical_repository/delete",
        headers=headers,
        json=body,
    )
    assert response.status == 200
    logical_repo_row = await sdb.fetch_one(
        select([LogicalRepository]).where(LogicalRepository.name == "alpha"),
    )
    assert logical_repo_row is None

    rel_depl_row = await pdb.fetch_one(
        select(GitHubReleaseDeployment).where(
            GitHubReleaseDeployment.repository_full_name == "src-d/go-git/alpha",
        ),
    )
    assert rel_depl_row is None

    pr_depl_row = await pdb.fetch_one(
        select(GitHubPullRequestDeployment).where(
            GitHubPullRequestDeployment.repository_full_name == "src-d/go-git",
        ),
    )
    assert pr_depl_row is None

    depl_facts_row = await pdb.fetch_one(
        select(GitHubDeploymentFacts).where(
            GitHubDeploymentFacts.deployment_name == "my-deployment",
        ),
    )
    assert depl_facts_row is None


@pytest.mark.app_validate_responses(False)
async def test_delete_logical_repo_clean_physical_repo_facts(client, headers, sdb, pdb) -> None:
    logical_repo = LogicalRepositoryFactory(
        name="alpha", repository_id=40550, prs={"title": ".*[Ff]ix"},
    )
    await sdb.execute(model_insert_stmt(logical_repo, with_primary_keys=False))
    await sdb.execute(
        update(RepositorySet)
        .where(RepositorySet.owner_id == 1)
        .values(
            {
                RepositorySet.items: [
                    ["github.com/src-d/go-git", 40550],
                    ["github.com/src-d/go-git/alpha", 40550],
                ],
                RepositorySet.updated_at: datetime.now(timezone.utc),
                RepositorySet.updates_count: RepositorySet.updates_count + 1,
            },
        ),
    )

    done_pr_facts = [
        GitHubDonePullRequestFactsFactory(repository_full_name="src-d/go-git"),
        GitHubDonePullRequestFactsFactory(repository_full_name="src-d/go-git/alpha"),
        GitHubDonePullRequestFactsFactory(repository_full_name="src-d/go-git/beta"),
    ]
    for done_pr_fact in done_pr_facts:
        await pdb.execute(model_insert_stmt(done_pr_fact))

    open_pr_facts = [
        GitHubOpenPullRequestFactsFactory(repository_full_name="src-d/go-git"),
        GitHubOpenPullRequestFactsFactory(repository_full_name="src-d/go-git/alpha"),
        GitHubOpenPullRequestFactsFactory(repository_full_name="src-d/go-git2"),
    ]
    for open_pr_fact in open_pr_facts:
        await pdb.execute(model_insert_stmt(open_pr_fact))

    await pdb.execute(model_insert_stmt(GitHubReleaseFactory(repository_full_name="src-d/go-git")))

    body = {"account": 1, "name": "github.com/src-d/go-git/alpha"}
    response = await client.request(
        method="POST",
        path="/v1/settings/logical_repository/delete",
        headers=headers,
        json=body,
    )
    assert response.status == 200

    await assert_missing_row(
        pdb, GitHubDonePullRequestFacts, repository_full_name="src-d/go-git/alpha",
    )
    row = pdb.fetch_one(
        select(GitHubDonePullRequestFacts).where(
            GitHubDonePullRequestFacts.repository_full_name == "src-d/go-git/beta",
        ),
    )
    assert row is not None
    await assert_missing_row(pdb, GitHubDonePullRequestFacts, repository_full_name="src-d/go-git")

    await assert_missing_row(
        pdb, GitHubOpenPullRequestFacts, repository_full_name="src-d/go-git/alpha",
    )
    row = pdb.fetch_one(
        select(GitHubOpenPullRequestFacts).where(
            GitHubOpenPullRequestFacts.repository_full_name == "src-d/go-git2",
        ),
    )
    assert row is not None
    await assert_missing_row(pdb, GitHubOpenPullRequestFacts, repository_full_name="src-d/go-git")

    row = await pdb.fetch_one(
        select(GitHubRelease).where(GitHubRelease.repository_full_name == "src-d/go-git"),
    )
    assert row is None


@pytest.mark.parametrize(
    "account, name, code",
    [
        (2, "", 403),
        (3, "", 404),
        (1, "xxx", 400),
        (1, "github.com/src-d/go-git", 400),
        (1, "github.com/src-d/go-git/xxx", 404),
    ],
)
async def test_delete_logical_repository_nasty_input(
    client,
    headers,
    logical_settings_db,
    account,
    name,
    code,
):
    body = {
        "account": account,
        "name": name or "github.com/src-d/go-git/alpha",
    }
    response = await client.request(
        method="POST",
        path="/v1/settings/logical_repository/delete",
        headers=headers,
        json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == code, "Response body is: " + body


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize("with_precomputed_reset_check", [False, True])
@with_defer
async def test_set_logical_repository_smoke(
    client,
    headers,
    metrics_calculator_factory,
    sdb,
    mdb,
    bots,
    release_match_setting_tag,
    prefixer,
    with_precomputed_reset_check,
):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2016, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2021, 1, 1, tzinfo=timezone.utc)
    if with_precomputed_reset_check:
        await metrics_calculator_no_cache.calc_pull_request_facts_github(
            time_from,
            time_to,
            {"src-d/go-git"},
            {},
            LabelFilter.empty(),
            JIRAFilter.empty(),
            False,
            bots,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            prefixer,
            False,
            False,
        )
        await wait_deferred()
    await _test_set_logical_repository(client, headers, sdb, 1)
    settings = Settings.from_account(1, sdb, mdb, None, None)
    df_post = await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from,
        time_to,
        {"src-d/go-git", "src-d/go-git/alpha"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        bots,
        await settings.list_release_matches(),
        await settings.list_logical_repositories(prefixer),
        prefixer,
        False,
        False,
    )
    assert (
        df_post[PullRequest.repository_full_name.name].values == "src-d/go-git/alpha"
    ).sum() == 90


async def _test_set_logical_repository(client, headers, sdb, n):
    title = "[Ff]ix.*"
    labels = ["BUG", "fiX", "Plumbing", "enhancement"]
    body = {
        "account": 1,
        "name": "alpha",
        "parent": "github.com/src-d/go-git",
        "prs": {
            "title": title,
            "labels_include": labels,
        },
        "releases": {
            "branches": "master",
            "tags": "v.*",
            "match": "tag",
        },
    }
    response = await client.request(
        method="PUT",
        path="/v1/settings/logical_repository",
        headers=headers,
        json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is: " + body
    rows = await sdb.fetch_all(
        select([LogicalRepository.name, LogicalRepository.prs]).order_by(LogicalRepository.name),
    )
    assert len(rows) == n
    assert rows[0][LogicalRepository.name.name] == "alpha"
    assert rows[0][LogicalRepository.prs.name] == {
        "title": title,
        "labels": [v.lower() for v in labels],
    }
    row = await sdb.fetch_one(select([RepositorySet]).where(RepositorySet.id == 1))
    assert row[RepositorySet.items.name][:3] == [
        ["github.com/src-d/gitbase", 39652769],
        ["github.com/src-d/go-git", 40550],
        ["github.com/src-d/go-git/alpha", 40550],
    ]
    assert len(row[RepositorySet.items.name]) == 2 + n
    row = await sdb.fetch_one(
        select([ReleaseSetting]).where(
            ReleaseSetting.repository == "github.com/src-d/go-git/alpha",
        ),
    )
    assert row[ReleaseSetting.branches.name] == "master"
    assert row[ReleaseSetting.tags.name] == "v.*"
    assert row[ReleaseSetting.match.name] == ReleaseMatch.tag


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_set_logical_repository_replace(
    client,
    headers,
    logical_settings_db,
    release_match_setting_tag_logical_db,
    sdb,
):
    await _test_set_logical_repository(client, headers, sdb, 2)


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_set_logical_repository_replace_identical(client, headers, sdb, logical_settings_db):
    # make the logical repositiry equal to the body
    await sdb.execute(
        update(LogicalRepository)
        .where(LogicalRepository.name == "alpha")
        .values(
            prs={"title": ".*[Aa]argh", "labels": ["bug", "fix"]},
            updated_at=LogicalRepository.updated_at,
        ),
    )
    # create a matching ReleaseSetting for the logical repository
    await sdb.execute(
        insert(ReleaseSetting).values(
            ReleaseSetting(
                repository="github.com/src-d/go-git/alpha",
                account_id=1,
                branches="master",
                tags="v.*",
                events=".*",
                match=ReleaseMatch.tag,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )

    body = {
        "account": 1,
        "name": "alpha",
        "parent": "github.com/src-d/go-git",
        "prs": {"title": ".*[Aa]argh", "labels_include": ["bug", "fix"]},
        "releases": {"branches": "master", "tags": "v.*", "match": "tag", "events": ".*"},
    }
    response = await client.request(
        method="PUT",
        path="/v1/settings/logical_repository",
        headers=headers,
        json=body,
    )
    assert response.status == 200


@pytest.mark.app_validate_responses(False)
async def test_set_logical_repository_clean_deployments(
    client,
    headers,
    sdb,
    logical_settings_db,
    pdb,
) -> None:
    await pdb.execute(
        insert(GitHubPullRequestDeployment).values(
            acc_id=1,
            repository_full_name="src-d/go-git",
            deployment_name="my-deployment2",
            pull_request_id=123,
            finished_at=datetime(2012, 1, 1),
        ),
    )

    body = {
        "account": 1,
        "name": "gamma",
        "parent": "github.com/src-d/go-git",
        "prs": {"title": ".*[Aa]argh"},
        "releases": {"branches": "master", "tags": "v.*", "match": "tag"},
    }
    response = await client.request(
        method="PUT",
        path="/v1/settings/logical_repository",
        headers=headers,
        json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, body

    pr_depl_row = await pdb.fetch_one(
        select(GitHubPullRequestDeployment).where(
            GitHubPullRequestDeployment.repository_full_name == "src-d/go-git",
        ),
    )
    assert pr_depl_row is None


@pytest.mark.parametrize(
    "account, name, parent, prs, match, extra, code",
    [
        (2, "alpha", "github.com/src-d/go-git", None, "tag", {}, 403),
        (3, "alpha", "github.com/src-d/go-git", None, "tag", {}, 404),
        (1, "alpha", "github.com/athenianco/athenian-api", None, "tag", {}, 403),
        (1, "", "github.com/src-d/go-git", None, "tag", {}, 400),
        (1, "alpha", "github.com/src-d/go-git", None, "branch", {}, 400),
        (
            1,
            "alpha",
            "github.com/src-d/go-git",
            None,
            "tag",
            {"deployments": {"title": "(f*+"}},
            400,
        ),
        (1, "alpha", "github.com/src-d/go-git", "(f*+", "tag", {}, 400),
    ],
)
async def test_set_logical_repository_nasty_input(
    client,
    headers,
    account,
    name,
    parent,
    prs,
    match,
    extra,
    code,
):
    body = {
        "account": account,
        "name": name,
        "parent": parent,
        "prs": {
            "title": ".*[Aa]argh" if not prs else prs,
        },
        "releases": {
            "branches": "master",
            "tags": "v.*",
            "match": match,
        },
        **extra,
    }
    response = await client.request(
        method="PUT",
        path="/v1/settings/logical_repository",
        headers=headers,
        json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == code, "Response body is: " + body
