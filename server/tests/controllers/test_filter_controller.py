from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from itertools import chain
import json
from operator import itemgetter
from typing import Collection, Dict, Optional, Set

from aiohttp import ClientResponse
import dateutil
import pytest
from sqlalchemy import delete, insert, select

from athenian.api.cache import CACHE_VAR_NAME, setup_cache_metrics
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.deployment import mine_deployments
from athenian.api.internal.miners.github.release_mine import mine_releases, override_first_releases
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseMatch
from athenian.api.models.metadata.github import Release
from athenian.api.models.persistentdata.models import (
    DeployedComponent as DBDeployedComponent,
    DeploymentNotification as DBDeploymentNotification,
    ReleaseNotification,
)
from athenian.api.models.precomputed.models import GitHubRelease
from athenian.api.models.state.models import AccountJiraInstallation, ReleaseSetting
from athenian.api.models.web import (
    CommitsList,
    DeployedComponent,
    DeploymentNotification,
    FilteredCodeCheckRuns,
    FilteredEnvironment,
    FilteredLabel,
    PullRequestEvent,
    PullRequestParticipant,
    PullRequestSet,
    PullRequestStage,
    ReleaseSet,
)
from athenian.api.models.web.diffed_releases import DiffedReleases
from athenian.api.models.web.filtered_deployments import FilteredDeployments
from athenian.api.prometheus import PROMETHEUS_REGISTRY_VAR_NAME
from athenian.api.typing_utils import wraps
from tests.conftest import FakeCache
from tests.controllers.conftest import with_only_master_branch


@pytest.fixture(scope="function")
async def with_event_releases(sdb, rdb):
    await sdb.execute(
        insert(ReleaseSetting).values(
            ReleaseSetting(
                repository="github.com/src-d/go-git",
                account_id=1,
                branches="master",
                tags=".*",
                events=".*",
                match=ReleaseMatch.event.value,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    await rdb.execute(
        insert(ReleaseNotification).values(
            ReleaseNotification(
                account_id=1,
                repository_node_id=40550,
                commit_hash_prefix="8d20cc5",
                name="Pushed!",
                author_node_id=40020,
                url="www",
                published_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )


@pytest.mark.filter_repositories
async def test_filter_repositories_no_repos(client, headers):
    body = {
        "date_from": "2015-01-12",
        "date_to": "2015-01-12",
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body,
    )
    assert response.status == 200
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == []


@pytest.mark.filter_repositories
@with_defer
async def test_filter_repositories_smoke(
    metrics_calculator_factory,
    client,
    headers,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2017, 9, 15, tzinfo=timezone.utc)
    time_to = datetime(2017, 9, 18, tzinfo=timezone.utc)
    args = (
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
    await metrics_calculator_no_cache.calc_pull_request_facts_github(*args)
    await wait_deferred()
    body = {
        "date_from": "2017-09-16",
        "date_to": "2017-09-17",
        "timezone": 60,
        "account": 1,
        "in": ["github.com/src-d/go-git"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body,
    )
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == ["github.com/src-d/go-git"]
    body["in"] = ["github.com/src-d/gitbase"]
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body,
    )
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == []


@pytest.mark.filter_repositories
@with_defer
async def test_filter_repositories_exclude_inactive_precomputed(
    metrics_calculator_factory,
    client,
    headers,
    release_match_setting_tag,
    prefixer,
    bots,
):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2017, 9, 15, tzinfo=timezone.utc)
    time_to = datetime(2017, 9, 18, tzinfo=timezone.utc)
    args = (
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
    await metrics_calculator_no_cache.calc_pull_request_facts_github(*args)
    await wait_deferred()
    body = {
        "date_from": "2017-09-16",
        "date_to": "2017-09-17",
        "timezone": 60,
        "account": 1,
        "in": ["github.com/src-d/go-git"],
        "exclude_inactive": True,
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body,
    )
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == []


@pytest.mark.filter_repositories
async def test_filter_repositories_exclude_inactive_cache(client, headers, client_cache):
    body = {
        "date_from": "2017-09-16",
        "date_to": "2017-09-17",
        "timezone": 60,
        "account": 1,
        "in": ["github.com/src-d/go-git"],
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body,
    )
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == ["github.com/src-d/go-git"]
    body["exclude_inactive"] = True
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body,
    )
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == []


@pytest.mark.filter_repositories
async def test_filter_repositories_fuck_up(client, headers, sdb, pdb):
    await sdb.execute(
        insert(ReleaseSetting).values(
            ReleaseSetting(
                repository="github.com/src-d/go-git",
                account_id=1,
                branches="master",
                tags=".*",
                events=".*",
                match=ReleaseMatch.branch.value,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    await pdb.execute(
        insert(GitHubRelease).values(
            GitHubRelease(
                node_id=1,
                acc_id=1,
                release_match="branch|whatever",
                repository_full_name="src-d/go-git",
                repository_node_id=222,
                name="release",
                published_at=datetime(2017, 1, 1, hour=12, tzinfo=timezone.utc),
                url="url",
                sha="sha",
                commit_id=111,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    body = {
        "date_from": "2017-01-01",
        "date_to": "2017-01-01",
        "timezone": 60,
        "account": 1,
        "in": ["github.com/src-d/go-git"],
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body,
    )
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == []


@pytest.mark.filter_repositories
@pytest.mark.parametrize(
    "account, date_to, in_, code",
    [
        (3, "2020-01-23", None, 404),
        (2, "2020-01-23", None, 422),
        (10, "2020-01-23", None, 404),
        (1, "2015-10-13", None, 200),
        (1, "2010-01-11", None, 400),
        (1, "2020-01-32", None, 400),
        (1, "2015-10-13", ["github.com/athenianco/athenian-api"], 403),
    ],
)
async def test_filter_repositories_nasty_input(client, headers, account, date_to, in_, code):
    body = {
        "date_from": "2015-10-13",
        "date_to": date_to,
        "account": account,
    }
    if in_ is not None:
        body["in"] = in_
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body,
    )
    assert response.status == code


@pytest.mark.filter_repositories
@with_defer
async def test_filter_repositories_logical(
    metrics_calculator_factory,
    client,
    headers,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag_logical,
    release_match_setting_tag_logical_db,
    prefixer,
    bots,
    logical_settings,
    logical_settings_db,
):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2017, 9, 15, tzinfo=timezone.utc)
    time_to = datetime(2018, 1, 18, tzinfo=timezone.utc)
    args = (
        time_from,
        time_to,
        {"src-d/go-git", "src-d/go-git/alpha"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        bots,
        release_match_setting_tag_logical,
        logical_settings,
        prefixer,
        False,
        False,
    )
    await metrics_calculator_no_cache.calc_pull_request_facts_github(*args)
    await wait_deferred()
    body = {
        "date_from": "2017-09-16",
        "date_to": "2018-01-17",
        "timezone": 60,
        "account": 1,
        "in": [],
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body,
    )
    repos = json.loads((await response.read()).decode("utf-8"))
    assert set(repos) == {"github.com/src-d/go-git", "github.com/src-d/go-git/alpha"}
    body["in"] = ["github.com/src-d/go-git/alpha"]
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body,
    )
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == ["github.com/src-d/go-git/alpha"]


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_contributors
@pytest.mark.parametrize("in_", [{}, {"in": []}])
async def test_filter_contributors_no_repos(client, headers, in_):
    body = {
        "date_from": "2015-01-12",
        "date_to": "2020-01-23",
        "account": 1,
        **in_,
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body,
    )
    assert response.status == 200
    contribs = await response.json()
    assert len(contribs) == 202
    assert len(set(c["login"] for c in contribs)) == len(contribs)
    assert all(c["login"].startswith("github.com/") for c in contribs)
    contribs = {c["login"]: c for c in contribs}
    assert "github.com/mcuadros" in contribs
    body["date_to"] = body["date_from"]
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body,
    )
    assert response.status == 200
    contribs = await response.json()
    assert contribs == []


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_contributors
async def test_filter_contributors(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "timezone": 60,
        "account": 1,
        "in": ["github.com/src-d/go-git"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body,
    )
    assert response.status == 200
    contribs = await response.json()
    assert len(contribs) == 199
    assert len(set(c["login"] for c in contribs)) == len(contribs)
    assert all(c["login"].startswith("github.com/") for c in contribs)
    contribs = {c["login"]: c for c in contribs}
    assert "github.com/mcuadros" in contribs
    assert "github.com/author_login" not in contribs
    assert "github.com/committer_login" not in contribs
    assert contribs["github.com/mcuadros"]["avatar"]
    assert contribs["github.com/mcuadros"]["name"] == "Máximo Cuadros"
    topics = set()
    for c in contribs.values():
        for v in c["updates"]:
            topics.add(v)
    assert topics == {
        "prs",
        "commenter",
        "commit_author",
        "commit_committer",
        "reviewer",
        "releaser",
    }
    body["in"] = ["github.com/src-d/gitbase"]
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body,
    )
    assert response.status == 200
    contribs = await response.json()
    assert contribs == []


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_contributors
async def test_filter_contributors_merger_only(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "timezone": 60,
        "account": 1,
        "in": ["github.com/src-d/go-git"],
        "as": ["merger"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body,
    )
    assert response.status == 200
    mergers = await response.json()
    mergers_logins = {c["login"] for c in mergers}

    assert len(mergers) == 8
    assert len(mergers_logins) == len(mergers)
    assert all(x.startswith("github.com/") for x in mergers_logins)

    expected_mergers = {
        "github.com/ajnavarro",
        "github.com/alcortesm",
        "github.com/erizocosmico",
        "github.com/jfontan",
        "github.com/mcuadros",
        "github.com/orirawlings",
        "github.com/smola",
        "github.com/strib",
    }
    assert mergers_logins == expected_mergers


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.flaky(reruns=3, reruns_delay=1)
@pytest.mark.filter_contributors
async def test_filter_contributors_with_empty_and_full_roles(client, headers):
    all_roles = [
        "author",
        "reviewer",
        "commit_author",
        "commit_committer",
        "commenter",
        "merger",
        "releaser",
    ]

    base_body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "timezone": 60,
        "account": 1,
        "in": ["github.com/src-d/go-git"],
    }

    body_empty_roles = {**base_body, "as": []}
    body_all_roles = {**base_body, "as": all_roles}

    response_empty_roles = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body_empty_roles,
    )
    response_all_roles = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body_all_roles,
    )

    parsed_empty_roles = json.loads((await response_empty_roles.read()).decode("utf-8"))
    parsed_all_roles = json.loads((await response_all_roles.read()).decode("utf-8"))

    assert sorted(parsed_empty_roles, key=itemgetter("login")) == sorted(
        parsed_all_roles, key=itemgetter("login"),
    )


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_contributors
@pytest.mark.parametrize(
    "account, date_to, in_, code",
    [
        (3, "2020-01-23", None, 404),
        (2, "2020-01-23", None, 422),
        (10, "2020-01-23", None, 404),
        (1, "2015-10-13", None, 200),
        (1, "2010-01-11", None, 400),
        (1, "2020-01-32", None, 400),
        (1, "2015-10-13", ["github.com/athenianco/athenian-api"], 403),
        (1, "2015-10-13", ["athenian-api"], 400),
    ],
)
async def test_filter_contributors_nasty_input(client, headers, account, date_to, in_, code):
    body = {
        "date_from": "2015-10-13",
        "date_to": date_to,
        "account": account,
    }
    if in_ is not None:
        body["in"] = in_
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body,
    )
    assert response.status == code


@pytest.fixture(scope="module")
def filter_prs_single_cache():
    fc = FakeCache()
    setup_cache_metrics({CACHE_VAR_NAME: fc, PROMETHEUS_REGISTRY_VAR_NAME: None})
    for v in fc.metrics["context"].values():
        v.set(defaultdict(int))
    return fc


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_pull_requests
@pytest.mark.parametrize(
    "stage",
    sorted(set(PullRequestStage) - {PullRequestStage.RELEASE_IGNORED}),
)
@with_only_master_branch
async def test_filter_prs_single_stage(
    # do not remove "mdb_rw", it is required by the decorators
    client,
    headers,
    mdb_rw,
    stage,
    app,
    filter_prs_single_cache,
):
    app.app[CACHE_VAR_NAME] = filter_prs_single_cache
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-04-23",
        "account": 1,
        "in": [],
        "stages": [stage],
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    await validate_prs_response(
        response, {stage}, set(), {}, datetime(year=2020, month=4, day=23, tzinfo=timezone.utc),
    )


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_pull_requests
@with_only_master_branch
async def test_filter_prs_stage_deployed(
    # do not remove "mdb_rw", it is required by the decorators
    client,
    headers,
    mdb_rw,
    app,
    precomputed_deployments,
    detect_deployments,
):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-04-23",
        "account": 1,
        "in": [],
        "stages": [PullRequestStage.DEPLOYED],
        "exclude_inactive": True,
        "environments": ["production"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    prs = PullRequestSet.from_dict(json.loads(text))
    assert len(prs.data) == 513


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_pull_requests
@pytest.mark.parametrize("event", sorted(PullRequestEvent))
@with_only_master_branch
async def test_filter_prs_single_event(
    # do not remove "mdb_rw", it is required by the decorators
    client,
    headers,
    mdb_rw,
    event,
    app,
    filter_prs_single_cache,
):
    app.app[CACHE_VAR_NAME] = filter_prs_single_cache
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-04-23",
        "account": 1,
        "in": [],
        "events": [event],
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    await validate_prs_response(
        response, set(), {event}, {}, datetime(year=2020, month=4, day=23, tzinfo=timezone.utc),
    )


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_pull_requests
@with_only_master_branch
async def test_filter_prs_event_deployed(
    # do not remove "mdb_rw", it is required by the decorators
    client,
    headers,
    mdb_rw,
    app,
    precomputed_deployments,
    detect_deployments,
):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-04-23",
        "account": 1,
        "in": [],
        "events": [PullRequestEvent.DEPLOYED],
        "exclude_inactive": False,
        "environments": ["production"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    prs = PullRequestSet.from_dict(json.loads(text))
    assert len(prs.data) == 513


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_pull_requests
@with_only_master_branch
async def test_filter_prs_no_stages(client, headers, mdb_rw):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-04-23",
        "updated_from": "2015-10-13",
        "updated_to": "2020-05-01",
        "timezone": 60,
        "account": 1,
        "in": [],
        "stages": list(PullRequestStage),
        "exclude_inactive": False,
    }
    time_to = datetime(year=2020, month=4, day=23, tzinfo=timezone.utc)
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    await validate_prs_response(response, set(PullRequestStage), set(), {}, time_to, 682)
    body["stages"] = []
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    assert response.status == 400
    del body["stages"]
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    assert response.status == 400


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_pull_requests
@with_only_master_branch
async def test_filter_prs_shot_updated(
    client,
    headers,
    mdb_rw,
    release_match_setting_tag_logical_db,
):
    # release_match_setting_tag_logical_db is here to save time and test that everything works
    body = {
        "date_from": "2016-10-13",
        "date_to": "2018-01-23",
        "timezone": 60,
        "account": 1,
        "in": [],
        "events": [PullRequestEvent.MERGED],
        "with": {
            "author": ["github.com/mcuadros"],
        },
        "updated_from": "2017-01-01",
        "updated_to": "2018-01-24",
        "exclude_inactive": False,
    }
    time_to = datetime(year=2018, month=1, day=24, tzinfo=timezone.utc)
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    await validate_prs_response(
        response,
        set(),
        {PullRequestEvent.MERGED},
        {"author": ["github.com/mcuadros"]},
        time_to,
        52,
    )
    # it is 75 without the constraints


@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_pull_requests
@with_only_master_branch
async def test_filter_prs_shot_team(
    client,
    headers,
    mdb_rw,
    release_match_setting_tag_logical_db,
    sample_team,
):
    # release_match_setting_tag_logical_db is here to save time and test that everything works
    team_str = "{%d}" % sample_team
    body = {
        "date_from": "2016-10-13",
        "date_to": "2018-01-23",
        "timezone": 60,
        "account": 1,
        "in": [],
        "events": [PullRequestEvent.MERGED],
        "with": {
            "author": [team_str],
        },
        "updated_from": "2017-01-01",
        "updated_to": "2018-01-24",
        "exclude_inactive": False,
    }
    time_to = datetime(year=2018, month=1, day=24, tzinfo=timezone.utc)
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    await validate_prs_response(
        response, set(), {PullRequestEvent.MERGED}, {"author": [team_str]}, time_to, 83,
    )


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_pull_requests
async def test_filter_prs_labels_include(client, headers):
    body = {
        "date_from": "2018-09-01",
        "date_to": "2018-11-30",
        "timezone": 0,
        "account": 1,
        "in": [],
        "events": [PullRequestEvent.MERGED],
        "labels_include": ["bug"],
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200
    prs = PullRequestSet.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert len(prs.data) == 2
    for pr in prs.data:
        assert "bug" in {label.name for label in pr.labels}


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_pull_requests
@pytest.mark.parametrize("timezone, must_match", [(120, True), (60, True), (0, False)])
async def test_filter_prs_merged_timezone(client, headers, timezone, must_match):
    body = {
        "date_from": "2017-07-08",
        "date_to": "2017-07-10",
        "timezone": timezone,
        "account": 1,
        "in": [],
        "events": [PullRequestEvent.MERGED],
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200
    obj = json.loads((await response.read()).decode("utf-8"))
    prs = PullRequestSet.from_dict(obj)  # type: PullRequestSet
    matched = False
    for pr in prs.data:
        if pr.number == 467:  # merged 2017-07-08 01:37 GMT+2 = 2017-07-07 23:37 UTC
            matched = True
    assert matched == must_match


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_pull_requests
@pytest.mark.parametrize("timezone, must_match", [(-7 * 60, False), (-8 * 60, True)])
async def test_filter_prs_created_timezone(client, headers, timezone, must_match):
    body = {
        "date_from": "2017-07-15",
        "date_to": "2017-07-16",
        "timezone": timezone,
        "account": 1,
        "in": [],
        "stages": list(PullRequestStage),
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200
    obj = json.loads((await response.read()).decode("utf-8"))
    prs = PullRequestSet.from_dict(obj)  # type: PullRequestSet
    matched = False
    for pr in prs.data:
        if pr.number == 485:  # created 2017-07-17 09:02 GMT+2 = 2017-07-17 07:02 UTC
            matched = True
    assert matched == must_match


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_filter_prs_event_releases(client, headers, with_event_releases):
    body = {
        "date_from": "2018-10-13",
        "date_to": "2019-02-23",
        "account": 1,
        "in": [],
        "events": [PullRequestEvent.MERGED],
        "exclude_inactive": True,
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    prs = PullRequestSet.from_dict(json.loads(text))
    assert len(prs.data) == 37


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_filter_prs_deployments_missing_env(
    client,
    headers,
    precomputed_deployments,
    detect_deployments,
):
    body = {
        "date_from": "2018-10-13",
        "date_to": "2019-02-23",
        "account": 1,
        "in": [],
        "events": [PullRequestEvent.MERGED],
        "exclude_inactive": True,
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    prs = PullRequestSet.from_dict(json.loads(text))
    assert len(prs.data) == 37
    counts = [0, 0, 0, 0]
    for pr in prs.data:
        assert pr.stage_timings.deploy["production"] > timedelta(0)
        counts[0] += "deployed" in pr.events_now
        counts[1] += "deployed" in pr.events_time_machine
        counts[2] += "deployed" in pr.stages_now
        counts[3] += "deployed" in pr.stages_time_machine
    assert counts == [37, 0, 37, 0]


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_filter_prs_deployments_with_env(
    client,
    headers,
    precomputed_deployments,
    detect_deployments,
):
    body = {
        "date_from": "2018-10-13",
        "date_to": "2019-02-23",
        "account": 1,
        "in": [],
        "events": [PullRequestEvent.MERGED],
        "exclude_inactive": True,
        "environments": ["production"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    prs = PullRequestSet.from_dict(json.loads(text))
    assert len(prs.data) == 37
    deployed_margin = datetime(2019, 11, 1, 12, 15) - datetime(2015, 5, 2)
    undeployed_margin = (datetime.now() - datetime(2019, 2, 23)) - timedelta(seconds=60)
    deps = 0
    for pr in prs.data:
        if PullRequestEvent.DEPLOYED in pr.events_now:
            deps += 1
            assert PullRequestStage.DEPLOYED in pr.stages_now
            assert pr.stage_timings.deploy["production"] < deployed_margin
        else:
            assert pr.stage_timings.deploy["production"] > undeployed_margin
        assert PullRequestEvent.DEPLOYED not in pr.events_time_machine
        assert PullRequestStage.DEPLOYED not in pr.stages_time_machine
    assert deps == 37


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_filter_prs_jira(client, headers, app, filter_prs_single_cache):
    app.app[CACHE_VAR_NAME] = filter_prs_single_cache
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-04-23",
        "account": 1,
        "in": [],
        "events": [PullRequestEvent.MERGED],
        "exclude_inactive": False,
    }
    if len(filter_prs_single_cache.mem) == 0:
        response = await client.request(
            method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
        )
        text = (await response.read()).decode("utf-8")
        assert response.status == 200, text
    body["jira"] = {
        "epics": ["DEV-149", "DEV-776", "DEV-737", "DEV-667", "DEV-140"],
        "labels_include": ["performance", "enhancement"],
        "labels_exclude": ["security"],
        "issue_types": ["Task"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    prs = PullRequestSet.from_dict(json.loads(text))
    data1 = prs.data
    assert len(prs.data) == 2
    filter_prs_single_cache.mem.clear()
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    prs = PullRequestSet.from_dict(json.loads(text))
    data2 = prs.data
    for pr in chain(data1, data2):
        assert pr.stage_timings.deploy["production"]
        pr.stage_timings.deploy["production"] = timedelta(0)
    assert data1 == data2


async def test_filter_prs_jira_disabled_projects(client, headers, disabled_dev):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-04-23",
        "account": 1,
        "in": [],
        "events": [PullRequestEvent.MERGED],
        "exclude_inactive": False,
        "jira": {
            "epics": ["DEV-149", "DEV-776", "DEV-737", "DEV-667", "DEV-140"],
        },
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    prs = PullRequestSet.from_dict(json.loads(text))
    assert len(prs.data) == 0


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_pull_requests
async def test_filter_prs_logical(
    client,
    headers,
    logical_settings_db,
    release_match_setting_tag_logical_db,
):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-04-23",
        "account": 1,
        "in": ["github.com/src-d/go-git/alpha", "github.com/src-d/go-git/beta"],
        "stages": [PullRequestStage.DONE],
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200
    prs = PullRequestSet.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert len(prs.data) == 255
    for pr in prs.data:
        assert pr.repository in ["github.com/src-d/go-git/alpha", "github.com/src-d/go-git/beta"]


open_go_git_pr_numbers = {
    570,
    816,
    970,
    1273,
    1069,
    1086,
    1098,
    1139,
    1152,
    1153,
    1173,
    1238,
    1243,
    1246,
    1254,
    1270,
    1269,
    1272,
    1286,
    1291,
    1285,
}

rejected_go_git_pr_numbers = {
    3,
    8,
    75,
    13,
    46,
    52,
    53,
    86,
    103,
    85,
    101,
    119,
    127,
    129,
    156,
    154,
    257,
    291,
    272,
    280,
    281,
    353,
    329,
    330,
    382,
    383,
    474,
    392,
    399,
    407,
    494,
    419,
    420,
    503,
    437,
    446,
    1186,
    497,
    486,
    506,
    560,
    548,
    575,
    591,
    619,
    639,
    670,
    671,
    689,
    743,
    699,
    715,
    768,
    776,
    782,
    790,
    789,
    824,
    800,
    805,
    819,
    821,
    849,
    861,
    863,
    926,
    1185,
    867,
    872,
    880,
    878,
    1188,
    908,
    946,
    940,
    947,
    952,
    951,
    975,
    1007,
    1010,
    976,
    988,
    997,
    1003,
    1002,
    1016,
    1104,
    1120,
    1044,
    1062,
    1075,
    1078,
    1109,
    1103,
    1122,
    1187,
    1182,
    1168,
    1170,
    1183,
    1184,
    1213,
    1248,
    1247,
    1265,
    1276,
}

force_push_dropped_go_git_pr_numbers = {
    504,
    561,
    907,
    1,
    2,
    5,
    6,
    7,
    9,
    10,
    11,
    12,
    14,
    15,
    20,
    16,
    17,
    18,
    21,
    22,
    23,
    25,
    26,
    24,
    27,
    28,
    30,
    32,
    34,
    35,
    37,
    39,
    47,
    54,
    56,
    55,
    58,
    61,
    64,
    66,
    63,
    68,
    69,
    70,
    74,
    78,
    79,
    83,
    84,
    87,
    88,
    89,
    92,
    93,
    94,
    95,
    96,
    97,
    90,
    91,
    104,
    105,
    106,
    108,
    99,
    100,
    102,
    116,
    117,
    118,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    124,
    121,
    122,
    130,
    131,
    132,
    133,
    135,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    153,
    136,
    138,
    140,
    141,
    142,
    143,
    144,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    176,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    185,
    186,
    187,
    188,
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,
    190,
    191,
    192,
    189,
    200,
    201,
    204,
    205,
    207,
    209,
    210,
    212,
    213,
    214,
    215,
    218,
    219,
    221,
    224,
    227,
    229,
    230,
    237,
    240,
    241,
    233,
    235,
    244,
}

rebased_resolved_pr_numbers = frozenset(
    [
        1,
        2,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
        14,
        15,
        16,
        17,
        18,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        30,
        32,
        34,
        35,
        37,
        39,
        47,
        54,
        55,
        56,
        58,
        61,
        63,
        64,
        66,
        68,
        69,
        70,
        74,
        83,
        84,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        99,
        100,
        102,
        104,
        105,
        106,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        121,
        122,
        124,
        130,
        131,
        132,
        133,
        135,
        136,
        138,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        150,
        151,
        153,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        169,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        185,
        186,
        187,
        188,
        189,
        190,
        191,
        192,
        200,
        201,
        204,
        205,
        207,
        209,
        210,
        212,
        213,
        214,
        215,
        218,
        219,
        221,
        224,
        227,
        229,
        230,
        233,
        235,
        237,
        240,
        241,
        244,
        907,
    ],
)

force_push_dropped_go_git_pr_numbers = frozenset(
    force_push_dropped_go_git_pr_numbers - rebased_resolved_pr_numbers,
)

will_never_be_released_go_git_pr_numbers = {
    1180,
    1195,
    1204,
    1205,
    1206,
    1208,
    1214,
    1225,
    1226,
    1235,
    1231,
}


async def validate_prs_response(
    response: ClientResponse,
    stages: Set[str],
    events: Set[str],
    parts: Dict[str, Collection[str]],
    time_to: datetime,
    count: Optional[int] = None,
) -> int:
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    obj = json.loads(text)
    prs = PullRequestSet.from_dict(obj)
    if stages == {PullRequestStage.DEPLOYED} or events == {PullRequestEvent.DEPLOYED}:
        assert len(prs.data) == 0
        return
    users = prs.include.users
    assert len(users) > 0, text
    for user in users:
        assert user.startswith("github.com/")
        assert len(user.split("/")) == 2
    assert len(prs.data) > 0, text
    numbers = set()
    total_comments = (
        total_commits
    ) = (
        total_review_comments
    ) = (
        total_released
    ) = total_rejected = total_review_requests = total_reviews = total_force_push_dropped = 0
    tdz = timedelta(0)
    timings = defaultdict(lambda: tdz)
    if count is not None:
        assert len(prs.data) == count
    failed_check_runs = defaultdict(int)
    for pr in prs.data:
        assert pr.title
        assert pr.repository == "github.com/src-d/go-git", str(pr)

        assert pr.number > 0, str(pr)
        assert pr.number not in numbers, str(pr)
        numbers.add(pr.number)

        # >= because there are closed PRs with 0 commits
        assert pr.size_added >= 0, str(pr)
        assert pr.size_removed >= 0, str(pr)
        assert pr.files_changed >= 0, str(pr)
        total_comments += pr.comments
        total_commits += pr.commits
        total_review_comments += pr.review_comments
        total_reviews += pr.reviews
        if pr.files_changed > 0:
            assert pr.commits > 0, str(pr)
        if pr.size_added > 0 or pr.size_removed > 0:
            assert pr.files_changed > 0, str(pr)
        if pr.review_comments > 0:
            assert pr.reviews > 0, str(pr)

        assert pr.created, str(pr)
        assert pr.created < time_to
        if pr.closed is None:
            assert pr.merged is None
        else:
            assert pr.closed > pr.created
        if pr.merged:
            assert pr.closed is not None
            assert abs(pr.merged - pr.closed) < timedelta(seconds=60)
        if pr.review_requested is not None:
            assert PullRequestEvent.REVIEW_REQUESTED in pr.events_now, str(pr)
        if PullRequestEvent.REVIEW_REQUESTED in pr.events_now:
            assert pr.review_requested is not None
            assert pr.review_requested >= pr.created, str(pr)
        if pr.first_review is not None:
            assert PullRequestEvent.REVIEWED in pr.events_now, str(pr)
            assert pr.reviews > 0, str(pr)
            assert pr.first_review > pr.created, str(pr)
        if pr.reviews > 0:
            assert pr.first_review is not None, str(pr)
        if pr.approved is not None:
            assert pr.first_review <= pr.approved, str(pr)

        assert stages.intersection(set(pr.stages_time_machine)) or events.intersection(
            set(pr.events_time_machine),
        ), str(pr)
        assert PullRequestEvent.CREATED in pr.events_now, str(pr)
        if pr.number not in open_go_git_pr_numbers:
            assert pr.closed is not None
            if pr.number not in will_never_be_released_go_git_pr_numbers:
                assert PullRequestStage.DONE in pr.stages_now, str(pr)
            else:
                assert PullRequestEvent.MERGED in pr.events_now
            if (
                pr.number not in rejected_go_git_pr_numbers
                and pr.number not in will_never_be_released_go_git_pr_numbers
                and pr.number not in force_push_dropped_go_git_pr_numbers
            ):
                assert PullRequestEvent.RELEASED in pr.events_now, str(pr)
            else:
                assert PullRequestEvent.RELEASED not in pr.events_now
                assert PullRequestEvent.RELEASED not in pr.events_time_machine
                if pr.number in rejected_go_git_pr_numbers:
                    assert PullRequestEvent.MERGED not in pr.events_now, str(pr)
                    assert PullRequestEvent.MERGED not in pr.events_time_machine, str(pr)
                if pr.number in force_push_dropped_go_git_pr_numbers:
                    assert PullRequestStage.FORCE_PUSH_DROPPED in pr.stages_now, str(pr)
                    assert PullRequestEvent.MERGED in pr.events_now, str(pr)
        else:
            assert pr.closed is None

        if PullRequestStage.WIP in pr.stages_now:
            assert PullRequestEvent.COMMITTED in pr.events_now, str(pr)
        if PullRequestStage.REVIEWING in pr.stages_now:
            assert PullRequestEvent.COMMITTED in pr.events_now, str(pr)
            assert pr.stage_timings.review is not None
        total_review_requests += PullRequestEvent.REVIEW_REQUESTED in pr.events_now
        if PullRequestStage.MERGING in pr.stages_now:
            assert PullRequestEvent.APPROVED in pr.events_now, str(pr)
            assert pr.stage_timings.merge is not None
        if PullRequestStage.RELEASING in pr.stages_now:
            assert PullRequestEvent.MERGED in pr.events_now, str(pr)
            assert PullRequestEvent.COMMITTED in pr.events_now, str(pr)
            assert pr.stage_timings.release is not None, str(pr)
        if PullRequestStage.DONE in pr.stages_now:
            assert pr.closed is not None, str(pr)
            if PullRequestEvent.MERGED in pr.events_now:
                if pr.number not in force_push_dropped_go_git_pr_numbers:
                    assert PullRequestEvent.RELEASED in pr.events_now, str(pr)
                else:
                    assert PullRequestStage.FORCE_PUSH_DROPPED in pr.stages_now, str(pr)
                    total_force_push_dropped += 1
            else:
                assert PullRequestEvent.REJECTED in pr.events_now, str(pr)
                total_rejected += 1

        if PullRequestStage.WIP in pr.stages_time_machine:
            assert PullRequestEvent.COMMITTED in pr.events_time_machine, str(pr)
        if PullRequestStage.REVIEWING in pr.stages_time_machine:
            assert PullRequestEvent.COMMITTED in pr.events_time_machine, str(pr)
            assert pr.stage_timings.review is not None
        total_review_requests += PullRequestEvent.REVIEW_REQUESTED in pr.events_time_machine
        if PullRequestStage.MERGING in pr.stages_time_machine:
            assert PullRequestEvent.APPROVED in pr.events_time_machine, str(pr)
            assert pr.stage_timings.merge is not None
        if PullRequestStage.RELEASING in pr.stages_time_machine:
            assert PullRequestEvent.MERGED in pr.events_time_machine, str(pr)
            assert PullRequestEvent.COMMITTED in pr.events_time_machine, str(pr)
            assert pr.stage_timings.release is not None, str(pr)
        if PullRequestStage.DONE in pr.stages_time_machine:
            assert pr.closed is not None, str(pr)
            if PullRequestEvent.MERGED in pr.events_time_machine:
                if pr.number not in force_push_dropped_go_git_pr_numbers:
                    assert PullRequestEvent.RELEASED in pr.events_time_machine, str(pr)
                else:
                    assert PullRequestStage.FORCE_PUSH_DROPPED in pr.stages_time_machine, str(pr)
            else:
                assert PullRequestEvent.REJECTED in pr.events_time_machine, str(pr)

        assert pr.stage_timings.wip is not None, str(pr)
        if pr.stage_timings.wip == tdz:
            if pr.stage_timings.merge is None:
                # review requested at once, no new commits, not merged
                assert (pr.closed and not pr.merged) or pr.stage_timings.review > tdz, str(pr)
            else:
                # no new commits after opening the PR
                assert pr.stage_timings.merge > tdz, str(pr)
        else:
            assert pr.stage_timings.wip > tdz, str(pr)
        assert pr.stage_timings.review is None or pr.stage_timings.review >= tdz
        assert pr.stage_timings.merge is None or pr.stage_timings.merge >= tdz
        assert pr.stage_timings.release is None or pr.stage_timings.release >= tdz
        timings["wip"] += pr.stage_timings.wip
        if pr.stage_timings.review is not None:
            timings["review"] += pr.stage_timings.review
        if pr.stage_timings.merge is not None:
            timings["merge"] += pr.stage_timings.merge
        if pr.stage_timings.release is not None:
            timings["release"] += pr.stage_timings.release

        if PullRequestEvent.REVIEWED in pr.events_now:
            # pr.review_comments can be 0
            assert pr.stage_timings.review is not None
        if pr.review_comments > 0:
            assert PullRequestEvent.REVIEWED in pr.events_now, str(pr)
        if PullRequestEvent.APPROVED in pr.events_now:
            assert PullRequestEvent.REVIEWED in pr.events_now, str(pr)
        if PullRequestEvent.CHANGES_REQUESTED in pr.events_now:
            assert PullRequestEvent.REVIEWED in pr.events_now, str(pr)
        if PullRequestEvent.MERGED not in pr.events_now and pr.closed is not None:
            assert PullRequestStage.DONE in pr.stages_now
            if pr.stage_timings.merge is None:
                # https://github.com/src-d/go-git/pull/878
                assert pr.commits == 0 or (pr.closed and not pr.merged), str(pr)
            else:
                assert pr.stage_timings.merge > tdz or pr.stage_timings.review > tdz, str(pr)
        if PullRequestEvent.RELEASED in pr.events_now:
            assert PullRequestStage.DONE in pr.stages_now, str(pr)
            assert pr.released is not None, str(pr)
            assert pr.stage_timings.merge is not None, str(pr)
            assert pr.stage_timings.release is not None
        if pr.released is not None:
            if pr.number not in force_push_dropped_go_git_pr_numbers:
                assert PullRequestEvent.RELEASED in pr.events_now, str(pr)
                assert pr.release_url, str(pr)
            else:
                assert PullRequestStage.FORCE_PUSH_DROPPED in pr.stages_now, str(pr)
                assert pr.release_url is None, str(pr)
            assert PullRequestStage.DONE in pr.stages_now, str(pr)
            total_released += 1

        if PullRequestEvent.REVIEWED in pr.events_time_machine:
            # pr.review_comments can be 0
            assert pr.stage_timings.review is not None
        if pr.review_comments > 0:
            assert PullRequestEvent.REVIEWED in pr.events_time_machine, str(pr)
        if PullRequestEvent.APPROVED in pr.events_time_machine:
            assert PullRequestEvent.REVIEWED in pr.events_time_machine, str(pr)
        if PullRequestEvent.CHANGES_REQUESTED in pr.events_time_machine:
            assert PullRequestEvent.REVIEWED in pr.events_time_machine, str(pr)
        if PullRequestEvent.MERGED not in pr.events_time_machine and pr.closed is not None:
            assert PullRequestStage.DONE in pr.stages_time_machine
            if pr.stage_timings.merge is None:
                # https://github.com/src-d/go-git/pull/878
                assert pr.commits == 0 or (pr.closed and not pr.merged), str(pr)
            else:
                assert pr.stage_timings.merge > tdz or pr.stage_timings.review > tdz, str(pr)
        if PullRequestEvent.RELEASED in pr.events_time_machine:
            assert PullRequestStage.DONE in pr.stages_time_machine, str(pr)
            assert pr.released is not None, str(pr)
            assert pr.stage_timings.merge is not None, str(pr)
            assert pr.stage_timings.release is not None
        if pr.released is not None:
            if pr.number not in force_push_dropped_go_git_pr_numbers:
                assert PullRequestEvent.RELEASED in pr.events_time_machine, str(pr)
                assert pr.release_url, str(pr)
            else:
                assert PullRequestStage.FORCE_PUSH_DROPPED in pr.stages_time_machine, str(pr)
                assert pr.release_url is None, str(pr)
            assert PullRequestStage.DONE in pr.stages_time_machine, str(pr)

        assert len(pr.participants) > 0
        authors = 0
        reviewers = 0
        mergers = 0
        releasers = 0
        inverse_participants = defaultdict(set)
        for p in pr.participants:
            assert p.id.startswith("github.com/")
            is_author = PullRequestParticipant.STATUS_AUTHOR in p.status
            authors += is_author
            if is_author:
                assert PullRequestParticipant.STATUS_REVIEWER not in p.status, pr.number
            reviewers += PullRequestParticipant.STATUS_REVIEWER in p.status
            mergers += PullRequestParticipant.STATUS_MERGER in p.status
            releasers += PullRequestParticipant.STATUS_RELEASER in p.status
            for s in p.status:
                inverse_participants[s].add(p.id)
        if pr.number != 749:
            # the author of 749 is deleted on GitHub
            assert authors == 1
        if reviewers == 0:
            assert PullRequestEvent.REVIEWED not in pr.events_now
            assert PullRequestEvent.APPROVED not in pr.events_now
            assert PullRequestEvent.CHANGES_REQUESTED not in pr.events_now
        else:
            assert PullRequestEvent.REVIEWED in pr.events_now
        assert mergers <= 1
        if mergers == 1:
            assert PullRequestEvent.MERGED in pr.events_now
        assert releasers <= 1
        if releasers == 1:
            assert PullRequestEvent.RELEASED in pr.events_now
        if parts:
            passed = False
            for role, p in parts.items():
                if p == ["{1}"]:  # very dirty but works
                    p = ["github.com/mcuadros", "github.com/vmarkovtsev", "github.com/smola"]
                passed |= bool(inverse_participants[role].intersection(set(p)))
            assert passed
        if pr.merged_with_failed_check_runs:
            for name in pr.merged_with_failed_check_runs:
                failed_check_runs[name] += 1
        # we cannot cover all possible cases while keeping the test run time reasonable :(

    assert total_comments > 0
    assert total_commits > 0
    if stages not in ({PullRequestStage.WIP}, {PullRequestStage.MERGING}):
        assert total_review_comments > 0
    else:
        assert total_review_comments == 0
    if stages != {PullRequestStage.WIP}:
        assert total_reviews > 0
    if stages not in (
        {PullRequestStage.RELEASING},
        {PullRequestStage.MERGING},
        {PullRequestStage.REVIEWING},
        {PullRequestStage.WIP},
        {PullRequestStage.FORCE_PUSH_DROPPED},
    ) and events not in ({PullRequestEvent.RELEASED}, {PullRequestEvent.MERGED}):
        assert total_rejected > 0
    else:
        assert total_rejected == 0
    if stages not in (
        {PullRequestStage.RELEASING},
        {PullRequestStage.MERGING},
        {PullRequestStage.REVIEWING},
        {PullRequestStage.WIP},
        {PullRequestStage.FORCE_PUSH_DROPPED},
    ) and events not in ({PullRequestEvent.REJECTED},):
        assert total_released > 0
    else:
        assert total_released == 0
    if {PullRequestStage.REVIEWING}.intersection(stages) or {
        PullRequestEvent.CHANGES_REQUESTED,
        PullRequestEvent.REVIEWED,
        PullRequestEvent.APPROVED,
        PullRequestEvent.CHANGES_REQUESTED,
    }.intersection(events):
        assert total_review_requests > 0
    if PullRequestStage.FORCE_PUSH_DROPPED in stages:
        assert total_force_push_dropped > 0
    for k, v in timings.items():
        assert v > tdz, k
    if (not (events == {PullRequestEvent.REJECTED} and not stages)) and (
        not (not events and PullRequestStage.DONE not in stages)
    ):
        assert failed_check_runs
        for key in failed_check_runs:
            assert "/" in key or key in ("DCO", "signed-off-by")
    return len(prs.data)


@pytest.mark.filter_pull_requests
@pytest.mark.parametrize(
    "account, date_to, updated_from, in_, code",
    [
        (3, "2020-01-23", None, [], 404),
        (2, "2020-01-23", None, [], 422),
        (10, "2020-01-23", None, [], 404),
        (1, "2015-10-13", None, [], 200),
        (1, "2010-01-11", None, [], 400),
        (1, "2020-01-32", None, [], 400),
        (1, "2015-10-13", "2015-10-15", [], 400),
        (1, "2015-10-13", None, ["github.com/athenianco/athenian-api"], 403),
    ],
)
async def test_filter_prs_nasty_input(client, headers, account, date_to, updated_from, in_, code):
    body = {
        "date_from": "2015-10-13",
        "date_to": date_to,
        "account": account,
        "in": in_,
        "stages": list(PullRequestStage),
        "exclude_inactive": False,
    }
    if updated_from is not None:
        body["updated_from"] = updated_from
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    assert response.status == code


@pytest.mark.filter_pull_requests
async def test_filter_prs_david_bug(client, headers):
    body = {
        "account": 1,
        "date_from": "2019-02-22",
        "date_to": "2020-02-22",
        "in": ["github.com/src-d/go-git"],
        "stages": [
            PullRequestStage.WIP,
            PullRequestStage.REVIEWING,
            PullRequestStage.MERGING,
            PullRequestStage.RELEASING,
        ],
        "with": {
            "author": ["github.com/Junnplus"],
            "reviewer": ["github.com/Junnplus"],
            "commit_author": ["github.com/Junnplus"],
            "commit_committer": ["github.com/Junnplus"],
            "commenter": ["github.com/Junnplus"],
            "merger": ["github.com/Junnplus"],
        },
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_pull_requests
async def test_filter_prs_developer_filter(client, headers):
    body = {
        "date_from": "2017-07-15",
        "date_to": "2017-12-16",
        "account": 1,
        "in": [],
        "stages": list(PullRequestStage),
        "with": {
            "author": ["github.com/mcuadros"],
        },
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200
    obj = json.loads((await response.read()).decode("utf-8"))
    prs = PullRequestSet.from_dict(obj)
    assert len(prs.data) == 27
    for pr in prs.data:
        passed = False
        for part in pr.participants:
            if part.id == "github.com/mcuadros":
                assert PullRequestParticipant.STATUS_AUTHOR in part.status
                passed = True
        assert passed


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_pull_requests
async def test_filter_prs_exclude_inactive(client, headers):
    body = {
        "date_from": "2017-01-01",
        "date_to": "2017-01-11",
        "account": 1,
        "in": [],
        "stages": list(PullRequestStage),
        "exclude_inactive": True,
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200
    obj = json.loads((await response.read()).decode("utf-8"))
    prs = PullRequestSet.from_dict(obj)
    assert len(prs.data) == 6


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_pull_requests
@with_defer
async def test_filter_prs_release_ignored(
    client,
    headers,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    pr_miner,
    prefixer,
    branches,
    default_branches,
    metrics_calculator_factory,
    bots,
):
    time_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=12, day=1, tzinfo=timezone.utc)
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        True,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        False,
        False,
    )
    time_from = datetime(year=2017, month=6, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=12, day=1, tzinfo=timezone.utc)
    releases, _, _, _ = await mine_releases(
        ["src-d/go-git"],
        {},
        None,
        default_branches,
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        with_deployments=False,
    )
    await wait_deferred()
    ignored = await override_first_releases(
        releases, {}, release_match_setting_tag, 1, pdb, threshold_factor=0,
    )
    assert ignored == 1

    body = {
        "date_from": "2017-06-01",
        "date_to": "2018-01-01",
        "account": 1,
        "in": [],
        "events": [PullRequestEvent.MERGED],
        "exclude_inactive": True,
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body,
    )
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    prs = PullRequestSet.from_dict(json.loads(text))
    assert len(prs.data) == 141
    release_ignored = 0
    for pr in prs.data:
        if PullRequestStage.RELEASE_IGNORED not in pr.stages_now:
            continue
        release_ignored += 1
        assert pr.released is None
        assert PullRequestStage.DONE in pr.stages_now
        assert PullRequestStage.RELEASING not in pr.stages_now
        assert PullRequestStage.RELEASE_IGNORED in pr.stages_time_machine
    assert release_ignored == 14


def _test_cached_mdb_pdb(func):
    async def wrapped_test_cached_mdb(**kwargs):
        await func(**kwargs)
        for db in ("mdb", "pdb"):
            await kwargs["app"].app[db].disconnect()
        try:
            await func(**kwargs)
        finally:
            for db in ("mdb", "pdb"):
                del kwargs["app"].app[db]

    wraps(wrapped_test_cached_mdb, func)
    return wrapped_test_cached_mdb


@pytest.mark.filter_commits
@_test_cached_mdb_pdb
async def test_filter_commits_bypassing_prs_mcuadros(client, headers, app, client_cache):
    body = {
        "account": 1,
        "date_from": "2019-01-12",
        "date_to": "2020-02-22",
        "in": ["{1}"],
        "property": "bypassing_prs",
        "with_author": ["github.com/mcuadros"],
        "with_committer": ["github.com/mcuadros"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/commits", headers=headers, json=body,
    )
    assert response.status == 200
    commits = CommitsList.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert commits.to_dict() == {
        "data": [
            {
                "author": {
                    "email": "mcuadros@gmail.com",
                    "login": "github.com/mcuadros",
                    "name": "Máximo Cuadros",
                    "timestamp": datetime(2019, 4, 24, 13, 20, 51, tzinfo=timezone.utc),
                    "timezone": 2.0,
                },
                "committer": {
                    "email": "mcuadros@gmail.com",
                    "login": "github.com/mcuadros",
                    "name": "Máximo Cuadros",
                    "timestamp": datetime(2019, 4, 24, 13, 20, 51, tzinfo=timezone.utc),
                    "timezone": 2.0,
                },
                "files_changed": 1,
                "hash": "5c6d199dc675465f5e103ea36c0bfcb9d3ebc565",
                "message": (
                    "plumbing: commit.Stats, fix panic on empty chucks\n\n"
                    "Signed-off-by: Máximo Cuadros <mcuadros@gmail.com>"
                ),
                "repository": "github.com/src-d/go-git",
                "size_added": 4,
                "size_removed": 0,
            },
        ],
        "include": {
            "users": {
                "github.com/mcuadros": {
                    "avatar": "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4",
                },
            },
        },
    }


@pytest.mark.filter_commits
async def test_filter_commits_no_pr_merges_mcuadros(client, headers):
    body = {
        "account": 1,
        "date_from": "2019-01-12",
        "date_to": "2020-02-22",
        "timezone": 60,
        "in": ["{1}"],
        "property": "no_pr_merges",
        "with_author": ["github.com/mcuadros"],
        "with_committer": ["github.com/mcuadros"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/commits", headers=headers, json=body,
    )
    assert response.status == 200
    commits = CommitsList.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert len(commits.data) == 5
    assert len(commits.include.users) == 1
    for c in commits.data:
        assert c.author.login == "github.com/mcuadros"
        assert c.committer.login == "github.com/mcuadros"


@pytest.mark.filter_commits
@_test_cached_mdb_pdb
async def test_filter_commits_bypassing_prs_team(client, headers, app, client_cache, sample_team):
    team_str = "{%d}" % sample_team
    body = {
        "account": 1,
        "date_from": "2019-01-12",
        "date_to": "2020-02-22",
        "in": ["{1}"],
        "property": "bypassing_prs",
        "with_author": [team_str],
        "with_committer": [team_str],
    }
    response = await client.request(
        method="POST", path="/v1/filter/commits", headers=headers, json=body,
    )
    assert response.status == 200
    commits = CommitsList.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert commits.to_dict() == {
        "data": [
            {
                "author": {
                    "email": "mcuadros@gmail.com",
                    "login": "github.com/mcuadros",
                    "name": "Máximo Cuadros",
                    "timestamp": datetime(2019, 4, 24, 13, 20, 51, tzinfo=timezone.utc),
                    "timezone": 2.0,
                },
                "committer": {
                    "email": "mcuadros@gmail.com",
                    "login": "github.com/mcuadros",
                    "name": "Máximo Cuadros",
                    "timestamp": datetime(2019, 4, 24, 13, 20, 51, tzinfo=timezone.utc),
                    "timezone": 2.0,
                },
                "files_changed": 1,
                "hash": "5c6d199dc675465f5e103ea36c0bfcb9d3ebc565",
                "message": (
                    "plumbing: commit.Stats, fix panic on empty chucks\n\n"
                    "Signed-off-by: Máximo Cuadros <mcuadros@gmail.com>"
                ),
                "repository": "github.com/src-d/go-git",
                "size_added": 4,
                "size_removed": 0,
            },
        ],
        "include": {
            "users": {
                "github.com/mcuadros": {
                    "avatar": "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4",
                },
            },
        },
    }


@pytest.mark.filter_commits
async def test_filter_commits_bypassing_prs_merges(client, headers):
    body = {
        "account": 1,
        "date_from": "2019-01-12",
        "date_to": "2020-02-22",
        "in": ["{1}"],
        "property": "bypassing_prs",
        "with_author": [],
        "with_committer": [],
    }
    response = await client.request(
        method="POST", path="/v1/filter/commits", headers=headers, json=body,
    )
    assert response.status == 200
    commits = CommitsList.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert len(commits.data) == 2
    for c in commits.data:
        assert c.committer.email != "noreply@github.com"


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_commits
@pytest.mark.parametrize("only_default_branch, length", [(True, 375), (False, 450)])
async def test_filter_commits_bypassing_prs_only_default_branch(
    client,
    headers,
    only_default_branch,
    length,
):
    body = {
        "account": 1,
        "date_from": "2015-01-12",
        "date_to": "2017-02-22",
        "in": ["{1}"],
        "property": "bypassing_prs",
        "only_default_branch": only_default_branch,
    }
    response = await client.request(
        method="POST", path="/v1/filter/commits", headers=headers, json=body,
    )
    assert response.status == 200
    commits = CommitsList.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert len(commits.data) == length


@pytest.mark.filter_commits
async def test_filter_commits_bypassing_prs_empty(client, headers):
    body = {
        "account": 1,
        "date_from": "2020-01-12",
        "date_to": "2020-02-22",
        "in": ["{1}"],
        "property": "bypassing_prs",
        "with_author": ["github.com/mcuadros"],
        "with_committer": ["github.com/mcuadros"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/commits", headers=headers, json=body,
    )
    assert response.status == 200
    commits = CommitsList.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert len(commits.data) == 0
    assert len(commits.include.users) == 0


@pytest.mark.filter_commits
async def test_filter_commits_bypassing_prs_no_with(client, headers):
    body = {
        "account": 1,
        "date_from": "2019-11-01",
        "date_to": "2020-02-21",
        "in": ["{1}"],
        "property": "bypassing_prs",
    }
    response = await client.request(
        method="POST", path="/v1/filter/commits", headers=headers, json=body,
    )
    assert response.status == 200
    commits = CommitsList.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert len(commits.data) == 0
    assert len(commits.include.users) == 0
    body["date_from"] = "2019-06-01"
    response = await client.request(
        method="POST", path="/v1/filter/commits", headers=headers, json=body,
    )
    assert response.status == 200
    commits = CommitsList.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert len(commits.data) == 1
    assert commits.data[0].committer.timestamp == datetime(
        2019, 7, 25, 8, 56, 22, tzinfo=dateutil.tz.tzutc(),
    )


@pytest.mark.filter_commits
@pytest.mark.parametrize("cached", [False, True], ids=["no cache", "with cache"])
@pytest.mark.parametrize(
    "account, date_to, in_, code",
    [
        (3, "2020-02-22", "{1}", 404),
        (2, "2020-02-22", "github.com/src-d/go-git", 422),
        (10, "2020-02-22", "{1}", 404),
        (1, "2020-01-12", "{1}", 200),
        (1, "2010-01-11", "{1}", 400),
        (1, "2020-02-32", "{1}", 400),
        (1, "2020-01-12", "github.com/athenianco/athenian-api", 403),
    ],
)
async def test_filter_commits_bypassing_prs_nasty_input(
    client,
    cached,
    headers,
    account,
    date_to,
    in_,
    code,
):
    body = {
        "account": account,
        "date_from": "2020-01-12",
        "date_to": date_to,
        "in": [in_],
        "property": "bypassing_prs",
    }
    response = await client.request(
        method="POST", path="/v1/filter/commits", headers=headers, json=body,
    )
    assert response.status == code


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_releases
async def test_filter_releases_by_tag(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-01-12",
        "timezone": 60,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/releases", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 200, response_text
    releases = ReleaseSet.from_dict(json.loads(response_text))
    assert len(releases.include.users) == 78
    assert "github.com/mcuadros" in releases.include.users
    assert len(releases.include.jira) == 41
    with_labels = 0
    with_epics = 0
    for key, val in releases.include.jira.items():
        assert key.startswith("DEV-")
        assert key == val.id
        assert val.title
        assert val.type
        with_labels += bool(val.labels)
        with_epics += bool(val.epic)
    assert with_labels == 40
    assert with_epics == 3
    assert len(releases.data) == 21
    pr_numbers = set()
    jira_stats = defaultdict(int)
    nnz_publishers = 0
    for release in releases.data:
        nnz_publishers += (nnz_publisher := release.publisher is not None)
        assert not nnz_publisher or release.publisher.startswith("github.com/"), str(release)
        assert len(release.commit_authors) > 0, str(release)
        assert all(a.startswith("github.com/") for a in release.commit_authors), str(release)
        for a in release.commit_authors:
            assert a in releases.include.users
        assert release.commits > 0, str(release)
        assert release.url.startswith("http"), str(release)
        assert release.name, str(release)
        assert release.added_lines > 0, str(release)
        assert release.deleted_lines > 0, str(release)
        assert release.age > timedelta(0), str(release)
        assert release.published >= datetime(year=2018, month=1, day=12, tzinfo=timezone.utc), str(
            release,
        )
        assert release.repository.startswith("github.com/"), str(release)
        assert len(release.prs) > 0
        for pr in release.prs:
            assert pr.number > 0
            assert pr.number not in pr_numbers
            pr_numbers.add(pr.number)
            assert pr.title
            assert pr.additions + pr.deletions > 0 or pr.number in {804}
            assert (pr.author is None and pr.number in {749, 1203}) or pr.author.startswith(
                "github.com/",
            )
            if pr.jira is not None:
                jira_stats[len(pr.jira)] += 1
    assert nnz_publishers > 0
    assert jira_stats == {1: 44}


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_releases
async def test_filter_releases_by_branch_no_jira(client, headers, client_cache, app, sdb, mdb_rw):
    backup = await mdb_rw.fetch_all(select([Release]))
    backup = [dict(r) for r in backup]
    await sdb.execute(delete(AccountJiraInstallation))
    await mdb_rw.execute(delete(Release))
    try:
        body = {
            "account": 1,
            "date_from": "2018-01-01",
            "date_to": "2020-10-22",
            "in": ["{1}"],
        }
        response = await client.request(
            method="POST", path="/v1/filter/releases", headers=headers, json=body,
        )
        response_text = (await response.read()).decode("utf-8")
        assert response.status == 200, response_text
        releases = ReleaseSet.from_dict(json.loads(response_text))
        assert len(releases.data) == 188
    finally:
        await mdb_rw.execute(insert(Release).values(backup))


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_releases
async def test_filter_releases_by_event(client, headers, with_event_releases):
    body = {
        "account": 1,
        "date_from": "2019-01-12",
        "date_to": "2020-01-12",
        "timezone": 60,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/releases", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 200, response_text
    releases = ReleaseSet.from_dict(json.loads(response_text))
    assert len(releases.data) == 1
    assert len(releases.data[0].prs) == 520


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_releases
async def test_filter_releases_by_participants(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-01-12",
        "timezone": 60,
        "in": ["{1}"],
        "with": {
            "releaser": ["github.com/smola"],
            "pr_author": ["github.com/mcuadros"],
            "commit_author": ["github.com/smola"],
        },
    }
    response = await client.request(
        method="POST", path="/v1/filter/releases", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 200, response_text
    releases = ReleaseSet.from_dict(json.loads(response_text))
    releases.include.users = set(releases.include.users)
    assert len(releases.include.users) == 78
    assert "github.com/mcuadros" in releases.include.users
    assert len(releases.data) == 12
    for release in releases.data:
        match_releaser = release.publisher == "github.com/smola"
        match_pr_author = "github.com/mcuadros" in {pr.author for pr in release.prs}
        match_commit_author = "github.com/smola" in release.commit_authors
        assert match_releaser or match_pr_author or match_commit_author, release


@pytest.mark.filter_releases
@pytest.mark.parametrize(
    "account, date_to, in_, code",
    [
        (3, "2020-02-22", "{1}", 404),
        (2, "2020-02-22", "github.com/src-d/go-git", 422),
        (10, "2020-02-22", "{1}", 404),
        (1, "2020-01-12", "{1}", 200),
        (1, "2010-01-11", "{1}", 400),
        (1, "2020-02-32", "{1}", 400),
        (1, "2020-01-12", "github.com/athenianco/athenian-api", 403),
    ],
)
async def test_filter_releases_nasty_input(client, headers, account, date_to, in_, code):
    body = {
        "account": account,
        "date_from": "2020-01-12",
        "date_to": date_to,
        "in": [in_],
    }
    response = await client.request(
        method="POST", path="/v1/filter/releases", headers=headers, json=body,
    )
    assert response.status == code


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_releases
async def test_filter_releases_by_jira(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-01",
        "date_to": "2020-10-22",
        "in": ["{1}"],
        "jira": {
            "labels_include": ["Bug", "onBoarding", "Performance"],
        },
    }
    response = await client.request(
        method="POST", path="/v1/filter/releases", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 200, response_text
    releases = ReleaseSet.from_dict(json.loads(response_text))
    assert len(releases.data) == 8


@pytest.mark.filter_releases
async def test_filter_releases_by_labels(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-01",
        "date_to": "2020-10-22",
        "in": ["{1}"],
        "labels_include": ["Bug", "enhancement", "PLUMBING"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/releases", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 200, response_text
    releases = ReleaseSet.from_dict(json.loads(response_text))
    assert len(releases.data) == 3


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_releases
@with_defer
async def test_filter_releases_deployments(
    client,
    headers,
    release_match_setting_tag_or_branch,
    prefixer,
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    dummy_deployment_label,
):
    await mine_deployments(
        ["src-d/go-git"],
        {},
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        ["production", "staging"],
        [],
        {},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches,
        default_branches,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    await wait_deferred()
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-01-12",
        "timezone": 60,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/releases", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 200, response_text
    releases = ReleaseSet.from_dict(json.loads(response_text))
    assert releases.include.deployments == {
        "Dummy deployment": DeploymentNotification(
            components=[
                DeployedComponent(
                    repository="github.com/src-d/go-git",
                    reference="v4.13.1 (0d1a009cbb604db18be960db5f1525b99a55d727)",
                ),
            ],
            environment="production",
            name="Dummy deployment",
            url=None,
            date_started=datetime(2019, 11, 1, 12, 0, tzinfo=timezone.utc),
            date_finished=datetime(2019, 11, 1, 12, 15, tzinfo=timezone.utc),
            conclusion="SUCCESS",
            labels={"xxx": ["yyy"]},
        ),
    }
    assert releases.data[0].deployments == ["Dummy deployment"]


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_get_prs_smoke(client, headers):
    body = {
        "account": 1,
        "prs": [
            {
                "repository": "github.com/src-d/go-git",
                "numbers": list(range(1000, 1100)),
            },
        ],
    }
    response = await client.request(
        method="POST", path="/v1/get/pull_requests", headers=headers, json=body,
    )
    response_body = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 200, response_body
    model = PullRequestSet.from_dict(response_body)
    assert len(model.data) == 51


@pytest.mark.parametrize(
    "account, repo, numbers, status",
    [
        (1, "bitbucket.org", [1, 2, 3], 400),
        (2, "github.com/src-d/go-git", [1, 2, 3], 422),
        (3, "github.com/src-d/go-git", [1, 2, 3], 404),
        (4, "github.com/src-d/go-git", [1, 2, 3], 404),
        (1, "github.com/whatever/else", [1, 2, 3], 403),
    ],
)
async def test_get_prs_nasty_input(client, headers, account, repo, numbers, status):
    body = {
        "account": account,
        "prs": [
            {
                "repository": repo,
                "numbers": numbers,
            },
        ],
    }
    response = await client.request(
        method="POST", path="/v1/get/pull_requests", headers=headers, json=body,
    )
    response_body = json.loads((await response.read()).decode("utf-8"))
    assert response.status == status, response_body


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@with_defer
async def test_get_prs_deployments(
    client,
    headers,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    branches,
    default_branches,
    prefixer,
    precomputed_deployments,
    detect_deployments,
):
    body = {
        "account": 1,
        "prs": [
            {
                "repository": "github.com/src-d/go-git",
                "numbers": [1160, 1179, 1168],
            },
        ],
        "environments": ["production"],
    }
    response = await client.request(
        method="POST", path="/v1/get/pull_requests", headers=headers, json=body,
    )
    response_body = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 200, response_body
    prs = PullRequestSet.from_dict(response_body)

    for pr in prs.data:
        if pr.number in (1160, 1179):
            assert pr.stage_timings.deploy["production"] > timedelta(0)
            assert PullRequestEvent.DEPLOYED in pr.events_now
            assert PullRequestStage.DEPLOYED in pr.stages_now
        if pr.number == 1168:
            assert not pr.stage_timings.deploy
            assert PullRequestEvent.DEPLOYED not in pr.events_now
            assert PullRequestStage.DEPLOYED not in pr.stages_now
    assert prs.include.deployments == {
        "Dummy deployment": DeploymentNotification(
            components=[
                DeployedComponent(
                    repository="github.com/src-d/go-git",
                    reference="v4.13.1 (0d1a009cbb604db18be960db5f1525b99a55d727)",
                ),
            ],
            environment="production",
            name="Dummy deployment",
            url=None,
            date_started=datetime(2019, 11, 1, 12, 0, tzinfo=timezone.utc),
            date_finished=datetime(2019, 11, 1, 12, 15, tzinfo=timezone.utc),
            conclusion="SUCCESS",
            labels=None,
        ),
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_labels
async def test_filter_labels_smoke(client, headers):
    body = {
        "account": 1,
        "repositories": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/labels", headers=headers, json=body,
    )
    response_body = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 200, response_body
    labels = [FilteredLabel.from_dict(i) for i in response_body]
    assert all(labels[i - 1].used_prs >= labels[i].used_prs for i in range(1, len(labels)))
    assert len(labels) == 7
    assert labels[0].name == "enhancement"
    assert labels[0].color == "84b6eb"
    assert labels[0].used_prs == 7


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.filter_labels
@pytest.mark.parametrize(
    "account, repos, status",
    [
        (1, ["github.com/whatever/else"], 403),
        (2, ["github.com/src-d/go-git"], 422),
        (3, ["github.com/src-d/go-git"], 404),
        (4, ["github.com/src-d/go-git"], 404),
        (1, [], 200),
    ],
)
async def test_filter_labels_nasty_input(client, headers, account, repos, status):
    body = {
        "account": account,
        "repositories": repos,
    }
    response = await client.request(
        method="POST", path="/v1/filter/labels", headers=headers, json=body,
    )
    response_body = json.loads((await response.read()).decode("utf-8"))
    assert response.status == status, response_body


async def test_get_releases_smoke(client, headers):
    body = {
        "account": 1,
        "releases": [
            {
                "repository": "github.com/src-d/go-git",
                "names": ["v4.0.0", "v4.4.0"],
            },
        ],
    }
    response = await client.request(
        method="POST", path="/v1/get/releases", headers=headers, json=body,
    )
    response_body = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 200, response_body
    ReleaseSet.from_dict(response_body)
    assert response_body == {
        "include": {
            "users": {
                "github.com/mcuadros": {
                    "avatar": "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4",
                },
                "github.com/maguro": {
                    "avatar": "https://avatars3.githubusercontent.com/u/165060?s=600&v=4",
                },
                "github.com/chunyi1994": {
                    "avatar": "https://avatars0.githubusercontent.com/u/18182339?s=600&v=4",
                },
                "github.com/jfontan": {
                    "avatar": "https://avatars1.githubusercontent.com/u/1829?s=600&v=4",
                },
                "github.com/ajnavarro": {
                    "avatar": "https://avatars3.githubusercontent.com/u/1196465?s=600&v=4",
                },
                "github.com/eiso": {
                    "avatar": "https://avatars3.githubusercontent.com/u/1247608?s=600&v=4",
                },
                "github.com/erizocosmico": {
                    "avatar": "https://avatars2.githubusercontent.com/u/1312023?s=600&v=4",
                },
                "github.com/dimonomid": {
                    "avatar": "https://avatars1.githubusercontent.com/u/1329932?s=600&v=4",
                },
                "github.com/krylovsk": {
                    "avatar": "https://avatars2.githubusercontent.com/u/136714?s=600&v=4",
                },
                "github.com/smithrobs": {
                    "avatar": "https://avatars2.githubusercontent.com/u/245836?s=600&v=4",
                },
                "github.com/ZJvandeWeg": {
                    "avatar": "https://avatars0.githubusercontent.com/u/2529595?s=600&v=4",
                },
                "github.com/thoeni": {
                    "avatar": "https://avatars2.githubusercontent.com/u/2122700?s=600&v=4",
                },
                "github.com/balkian": {
                    "avatar": "https://avatars2.githubusercontent.com/u/213135?s=600&v=4",
                },
                "github.com/grunenwflorian": {
                    "avatar": "https://avatars3.githubusercontent.com/u/22022442?s=600&v=4",
                },
                "github.com/blacksails": {
                    "avatar": "https://avatars2.githubusercontent.com/u/3807831?s=600&v=4",
                },
                "github.com/dvrkps": {
                    "avatar": "https://avatars2.githubusercontent.com/u/4771727?s=600&v=4",
                },
                "github.com/antham": {
                    "avatar": "https://avatars1.githubusercontent.com/u/4854264?s=600&v=4",
                },
                "github.com/kuba--": {
                    "avatar": "https://avatars2.githubusercontent.com/u/4056521?s=600&v=4",
                },
                "github.com/taruti": {
                    "avatar": "https://avatars0.githubusercontent.com/u/42174?s=600&v=4",
                },
                "github.com/bzz": {
                    "avatar": "https://avatars0.githubusercontent.com/u/5582506?s=600&v=4",
                },
                "github.com/orirawlings": {
                    "avatar": "https://avatars0.githubusercontent.com/u/57213?s=600&v=4",
                },
                "github.com/matjam": {
                    "avatar": "https://avatars2.githubusercontent.com/u/578676?s=600&v=4",
                },
                "github.com/ferhatelmas": {
                    "avatar": "https://avatars2.githubusercontent.com/u/648018?s=600&v=4",
                },
                "github.com/josharian": {
                    "avatar": "https://avatars0.githubusercontent.com/u/67496?s=600&v=4",
                },
                "github.com/darkowlzz": {
                    "avatar": "https://avatars1.githubusercontent.com/u/614105?s=600&v=4",
                },
                "github.com/strib": {
                    "avatar": "https://avatars2.githubusercontent.com/u/8516691?s=600&v=4",
                },
                "github.com/wellsjo": {
                    "avatar": "https://avatars2.githubusercontent.com/u/823446?s=600&v=4",
                },
            },
            "jira": {
                "DEV-261": {
                    "id": "DEV-261",
                    "title": "Remove API deprecations",
                    "labels": ["code-quality"],
                    "type": "task",
                },
                "DEV-760": {
                    "id": "DEV-760",
                    "title": "Optimize loading releases by branch",
                    "labels": ["performance"],
                    "type": "task",
                },
                "DEV-772": {
                    "id": "DEV-772",
                    "title": "Support JIRA tables in the API + unit tests",
                    "epic": "DEV-149",
                    "labels": ["enhancement"],
                    "type": "task",
                },
                "DEV-638": {
                    "id": "DEV-638",
                    "title": "Optimize filter_commits SQL",
                    "labels": ["performance"],
                    "type": "task",
                },
            },
        },
        "data": [
            {
                "name": "v4.4.0",
                "repository": "github.com/src-d/go-git",
                "url": "https://github.com/src-d/go-git/releases/tag/v4.4.0",
                "published": "2018-05-16T10:34:04Z",
                "age": "2494701s",
                "added_lines": 453,
                "deleted_lines": 28,
                "commits": 6,
                "publisher": "github.com/mcuadros",
                "commit_authors": [
                    "github.com/jfontan",
                    "github.com/maguro",
                    "github.com/mcuadros",
                ],
                "prs": [
                    {
                        "number": 815,
                        "title": 'Fix for "Worktree Add function adds ".git" directory"',
                        "additions": 62,
                        "deletions": 8,
                        "author": "github.com/kuba--",
                        "jira": ["DEV-638"],
                    },
                    {
                        "number": 825,
                        "title": "Worktree: Provide ability to add excludes to worktree",
                        "additions": 285,
                        "deletions": 7,
                        "author": "github.com/maguro",
                        "jira": ["DEV-760"],
                    },
                    {
                        "number": 833,
                        "title": "git: remote, Do not iterate all references on update.",
                        "additions": 22,
                        "deletions": 2,
                        "author": "github.com/jfontan",
                        "jira": ["DEV-772"],
                    },
                ],
            },
            {
                "name": "v4.0.0",
                "repository": "github.com/src-d/go-git",
                "url": "https://github.com/src-d/go-git/releases/tag/v4.0.0",
                "published": "2018-01-08T13:07:18Z",
                "age": "10869148s",
                "added_lines": 10699,
                "deleted_lines": 3354,
                "commits": 181,
                "publisher": "github.com/mcuadros",
                "commit_authors": [
                    "github.com/ZJvandeWeg",
                    "github.com/ajnavarro",
                    "github.com/antham",
                    "github.com/balkian",
                    "github.com/blacksails",
                    "github.com/bzz",
                    "github.com/chunyi1994",
                    "github.com/darkowlzz",
                    "github.com/dimonomid",
                    "github.com/dvrkps",
                    "github.com/eiso",
                    "github.com/erizocosmico",
                    "github.com/ferhatelmas",
                    "github.com/grunenwflorian",
                    "github.com/jfontan",
                    "github.com/josharian",
                    "github.com/krylovsk",
                    "github.com/matjam",
                    "github.com/mcuadros",
                    "github.com/orirawlings",
                    "github.com/smithrobs",
                    "github.com/strib",
                    "github.com/taruti",
                    "github.com/thoeni",
                    "github.com/wellsjo",
                ],
                "prs": [
                    {
                        "number": 534,
                        "title": "plumbing: object, commit.Parent() method",
                        "additions": 24,
                        "deletions": 0,
                        "author": "github.com/josharian",
                    },
                    {
                        "number": 577,
                        "title": "Worktree.Add: Support Add deleted files, fixes #571",
                        "additions": 43,
                        "deletions": 0,
                        "author": "github.com/grunenwflorian",
                    },
                    {
                        "number": 579,
                        "title": "revlist: do not visit again already visited parents",
                        "additions": 38,
                        "deletions": 11,
                        "author": "github.com/erizocosmico",
                    },
                    {
                        "number": 580,
                        "title": "remote: iterate over references only once",
                        "additions": 73,
                        "deletions": 40,
                        "author": "github.com/erizocosmico",
                    },
                    {
                        "number": 582,
                        "title": "packfile: improve performance of delta generation",
                        "additions": 367,
                        "deletions": 56,
                        "author": "github.com/erizocosmico",
                    },
                    {
                        "number": 583,
                        "title": (
                            "Minor fix to grammatical error in error message for"
                            " ErrRepositoryNotExists"
                        ),
                        "additions": 1,
                        "deletions": 1,
                        "author": "github.com/matjam",
                    },
                    {
                        "number": 584,
                        "title": "revert: revlist: do not revisit already visited ancestors",
                        "additions": 3,
                        "deletions": 17,
                        "author": "github.com/erizocosmico",
                    },
                    {
                        "number": 585,
                        "title": "examples: update link to GoDoc in _examples/storage",
                        "additions": 1,
                        "deletions": 1,
                        "author": "github.com/bzz",
                    },
                    {
                        "number": 586,
                        "title": "plumbing: the commit walker can skip externally-seen commits",
                        "additions": 45,
                        "deletions": 15,
                        "author": "github.com/strib",
                    },
                    {
                        "number": 587,
                        "title": "config: support a configurable, and turn-off-able, pack.window",
                        "additions": 146,
                        "deletions": 42,
                        "author": "github.com/strib",
                    },
                    {
                        "number": 588,
                        "title": (
                            "revlist: do not revisit ancestors as long as all branches are visited"
                        ),
                        "additions": 84,
                        "deletions": 3,
                        "author": "github.com/erizocosmico",
                    },
                    {
                        "number": 608,
                        "title": "Add port to SCP Endpoints",
                        "additions": 23,
                        "deletions": 3,
                        "author": "github.com/balkian",
                    },
                    {
                        "number": 609,
                        "title": "remote: add support for ls-remote",
                        "additions": 70,
                        "deletions": 0,
                        "author": "github.com/darkowlzz",
                    },
                    {
                        "number": 610,
                        "title": "remote: add the last 100 commits for each ref in haves list",
                        "additions": 97,
                        "deletions": 4,
                        "author": "github.com/strib",
                    },
                    {
                        "number": 613,
                        "title": "Add Stats() to Commit",
                        "additions": 167,
                        "deletions": 0,
                        "author": "github.com/darkowlzz",
                    },
                    {
                        "number": 615,
                        "title": "Fix spelling Unstagged -> Unstaged",
                        "additions": 4,
                        "deletions": 4,
                        "author": "github.com/blacksails",
                    },
                    {
                        "number": 616,
                        "title": "Add support for signed commits",
                        "additions": 69,
                        "deletions": 0,
                        "author": "github.com/darkowlzz",
                    },
                    {
                        "number": 617,
                        "title": "Fix spelling",
                        "additions": 1,
                        "deletions": 1,
                        "author": "github.com/blacksails",
                    },
                    {
                        "number": 626,
                        "title": (
                            "packp/capability: Skip argument validations for unknown capabilities"
                        ),
                        "additions": 36,
                        "deletions": 11,
                        "author": "github.com/orirawlings",
                    },
                    {
                        "number": 631,
                        "title": "packfile: use buffer pool for diffs",
                        "additions": 13,
                        "deletions": 4,
                        "author": "github.com/strib",
                    },
                    {
                        "number": 632,
                        "title": "packfile: delete index maps from memory when no longer needed",
                        "additions": 6,
                        "deletions": 0,
                        "author": "github.com/strib",
                    },
                    {
                        "number": 633,
                        "title": "travis: update go versions",
                        "additions": 2,
                        "deletions": 2,
                        "author": "github.com/dvrkps",
                    },
                    {
                        "number": 638,
                        "title": "Updating reference to the git object model",
                        "additions": 1,
                        "deletions": 1,
                        "author": "github.com/thoeni",
                    },
                    {
                        "number": 640,
                        "title": "utils: merkletrie, filesystem fix symlinks to dir",
                        "additions": 40,
                        "deletions": 0,
                        "author": "github.com/dimonomid",
                    },
                    {
                        "number": 641,
                        "title": "fix: a range loop can break in advance",
                        "additions": 1,
                        "deletions": 0,
                        "author": "github.com/chunyi1994",
                    },
                    {
                        "number": 643,
                        "title": "Fix typo in the readme",
                        "additions": 1,
                        "deletions": 1,
                        "author": "github.com/ZJvandeWeg",
                    },
                    {
                        "number": 646,
                        "title": "format: packfile fix DecodeObjectAt when Decoder has type",
                        "additions": 29,
                        "deletions": 4,
                        "author": "github.com/mcuadros",
                    },
                    {
                        "number": 647,
                        "title": "examples,plumbing,utils: typo fixes",
                        "additions": 20,
                        "deletions": 20,
                        "author": "github.com/ferhatelmas",
                    },
                    {
                        "number": 649,
                        "title": (
                            "transport: made public all the fields and standardized AuthMethod"
                        ),
                        "additions": 59,
                        "deletions": 55,
                        "author": "github.com/mcuadros",
                    },
                    {
                        "number": 650,
                        "title": "transport: converts Endpoint interface into a struct",
                        "additions": 278,
                        "deletions": 269,
                        "author": "github.com/mcuadros",
                    },
                    {
                        "number": 651,
                        "title": "dotgit: remove ref cache for packed refs",
                        "additions": 29,
                        "deletions": 48,
                        "author": "github.com/erizocosmico",
                    },
                    {
                        "number": 652,
                        "title": "plumbing/object: do not eat error on tree decode",
                        "additions": 39,
                        "deletions": 2,
                        "author": "github.com/ferhatelmas",
                    },
                    {
                        "number": 653,
                        "title": "plumbing: object, new Commit.Verify method",
                        "additions": 95,
                        "deletions": 0,
                        "author": "github.com/darkowlzz",
                    },
                    {
                        "number": 655,
                        "title": "*: update to go-billy.v4 and go-git-fixtures.v3",
                        "additions": 84,
                        "deletions": 89,
                        "author": "github.com/mcuadros",
                    },
                    {
                        "number": 656,
                        "title": "plumbing: object, fix Commit.Verify test",
                        "additions": 3,
                        "deletions": 2,
                        "author": "github.com/darkowlzz",
                    },
                    {
                        "number": 657,
                        "title": "plumbing: transport/http, Close http.Body reader when needed",
                        "additions": 3,
                        "deletions": 1,
                        "author": "github.com/ajnavarro",
                    },
                    {
                        "number": 658,
                        "title": "plumbing: object/tag, add signature and verification support",
                        "additions": 172,
                        "deletions": 7,
                        "author": "github.com/darkowlzz",
                    },
                    {
                        "number": 659,
                        "title": "doc: Update compatibility for commit/tag verify",
                        "additions": 2,
                        "deletions": 2,
                        "author": "github.com/ferhatelmas",
                    },
                    {
                        "number": 660,
                        "title": "fix Repository.ResolveRevision for branch and tag",
                        "additions": 82,
                        "deletions": 54,
                        "author": "github.com/antham",
                    },
                    {
                        "number": 661,
                        "title": "all: fixes for ineffective assign",
                        "additions": 23,
                        "deletions": 2,
                        "author": "github.com/ferhatelmas",
                    },
                    {
                        "number": 663,
                        "title": "storage: filesystem, add support for git alternates",
                        "additions": 148,
                        "deletions": 1,
                        "author": "github.com/darkowlzz",
                    },
                    {
                        "number": 664,
                        "title": "plumbing/transport: Fix truncated comment in Endpoint",
                        "additions": 1,
                        "deletions": 1,
                        "author": "github.com/orirawlings",
                    },
                    {
                        "number": 665,
                        "title": "remote: support for non-force, fast-forward-only fetches",
                        "additions": 205,
                        "deletions": 22,
                        "author": "github.com/strib",
                    },
                    {
                        "number": 666,
                        "title": (
                            "dotgit: handle refs that exist in both packed-refs and a loose ref"
                            " file"
                        ),
                        "additions": 76,
                        "deletions": 6,
                        "author": "github.com/strib",
                    },
                    {
                        "number": 667,
                        "title": "all: simplification",
                        "additions": 56,
                        "deletions": 118,
                        "author": "github.com/ferhatelmas",
                    },
                    {
                        "number": 668,
                        "title": "Updating the outdated README example to the new one",
                        "additions": 6,
                        "deletions": 7,
                        "author": "github.com/eiso",
                    },
                    {
                        "number": 669,
                        "title": "storage/repository: add new functions for garbage collection",
                        "additions": 928,
                        "deletions": 74,
                        "author": "github.com/strib",
                    },
                    {
                        "number": 672,
                        "title": "all: gofmt -s",
                        "additions": 22,
                        "deletions": 22,
                        "author": "github.com/ferhatelmas",
                    },
                    {
                        "number": 674,
                        "title": "dotgit: use Equal method of time.Time for equality",
                        "additions": 1,
                        "deletions": 1,
                        "author": "github.com/ferhatelmas",
                    },
                    {
                        "number": 675,
                        "title": "git: worktree, add Clean() method for git clean",
                        "additions": 67,
                        "deletions": 0,
                        "author": "github.com/darkowlzz",
                    },
                    {
                        "number": 677,
                        "title": "object: patch, fix stats for submodules (fixes #654)",
                        "additions": 51,
                        "deletions": 5,
                        "author": "github.com/krylovsk",
                    },
                    {
                        "number": 680,
                        "title": (
                            "License upgrade, plus code of conduct and contributing guidelines"
                        ),
                        "additions": 336,
                        "deletions": 19,
                        "author": "github.com/mcuadros",
                    },
                    {
                        "number": 686,
                        "title": "git: worktree, add Grep() method for git grep",
                        "additions": 324,
                        "deletions": 1,
                        "author": "github.com/darkowlzz",
                    },
                    {
                        "number": 687,
                        "title": "check .ssh/config for host and port overrides; fixes #629",
                        "additions": 110,
                        "deletions": 0,
                        "author": "github.com/smithrobs",
                    },
                    {
                        "number": 688,
                        "title": "doc: update compatibility for clean",
                        "additions": 1,
                        "deletions": 1,
                        "author": "github.com/darkowlzz",
                    },
                    {
                        "number": 690,
                        "title": "README.md update",
                        "additions": 14,
                        "deletions": 18,
                        "author": "github.com/mcuadros",
                    },
                    {
                        "number": 695,
                        "title": "git: Worktree.Grep() support multiple patterns and pathspecs",
                        "additions": 124,
                        "deletions": 35,
                        "author": "github.com/darkowlzz",
                    },
                    {
                        "number": 696,
                        "title": "*: simplication",
                        "additions": 4,
                        "deletions": 11,
                        "author": "github.com/ferhatelmas",
                    },
                    {
                        "number": 697,
                        "title": "plumbing: packafile, improve delta reutilization",
                        "additions": 152,
                        "deletions": 29,
                        "author": "github.com/ajnavarro",
                    },
                    {
                        "number": 698,
                        "title": "plumbing: cache, enforce the use of cache in packfile decoder",
                        "additions": 113,
                        "deletions": 68,
                        "author": "github.com/jfontan",
                    },
                    {
                        "number": 700,
                        "title": (
                            "Add a setRef and rewritePackedRefsWhileLocked versions that supports"
                            " non rw fs"
                        ),
                        "additions": 142,
                        "deletions": 33,
                        "author": "github.com/jfontan",
                    },
                    {
                        "number": 710,
                        "title": "fix typo",
                        "additions": 1,
                        "deletions": 1,
                        "author": "github.com/wellsjo",
                        "jira": ["DEV-261"],
                    },
                ],
            },
        ],
    }


@pytest.mark.parametrize(
    "account, repo, names, code",
    [
        (3, "github.com/src-d/go-git", ["v4.0.0", "v4.4.0"], 404),
        (1, "github.com/athenianco/athenian-api", ["v4.0.0", "v4.4.0"], 403),
        (2, "github.com/src-d/go-git", ["v4.0.0", "v4.4.0"], 422),
        (1, "github.com/src-d/go-git", [], 200),
    ],
)
async def test_get_releases_nasty_input(client, headers, account, repo, names, code):
    body = {
        "account": account,
        "releases": [
            {
                "repository": repo,
                "names": names,
            },
        ],
    }
    response = await client.request(
        method="POST", path="/v1/get/releases", headers=headers, json=body,
    )
    response_body = json.loads((await response.read()).decode("utf-8"))
    assert response.status == code, response_body


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.flaky(reruns=3)
async def test_diff_releases_smoke(client, headers):
    body = {
        "account": 1,
        "borders": {
            "github.com/src-d/go-git": [{"old": "v4.0.0", "new": "v4.4.0"}],
        },
    }
    response = await client.request(
        method="POST", path="/v1/diff/releases", headers=headers, json=body,
    )
    response_body = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 200, response_body
    dr = DiffedReleases.from_dict(response_body)
    assert len(dr.include.users) == 41
    assert len(dr.include.jira) == 35
    assert len(dr.data["github.com/src-d/go-git"]) == 1
    assert {r.name for r in dr.data["github.com/src-d/go-git"][0].releases} == {
        "v4.3.1",
        "v4.1.0",
        "v4.2.0",
        "v4.3.0",
        "v4.2.1",
        "v4.1.1",
        "v4.4.0",
    }
    assert dr.data["github.com/src-d/go-git"][0].old == "v4.0.0"
    assert dr.data["github.com/src-d/go-git"][0].new == "v4.4.0"


@pytest.mark.flaky(reruns=3)
@with_defer
async def test_diff_releases_commits(
    client,
    headers,
    mdb,
    pdb,
    rdb,
    release_match_setting_branch,
    prefixer,
    branches,
    default_branches,
):
    # d105e15d91e7553d9d40d6e9fffe0a5008cf8afe
    # 31a249d0d5b71bc0f374d3297247d89808263a8b
    body = {
        "account": 1,
        "borders": {
            "github.com/src-d/go-git": [{"old": "d105e15", "new": "31a249d"}],
        },
    }
    response = await client.request(
        method="POST", path="/v1/diff/releases", headers=headers, json=body,
    )
    response_body = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 200, response_body
    dr = DiffedReleases.from_dict(response_body)
    assert len(dr.data) == 0

    time_from = datetime(year=2017, month=3, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=4, day=1, tzinfo=timezone.utc)
    releases, _, _, _ = await mine_releases(
        ["src-d/go-git"],
        {},
        branches,
        default_branches,
        time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        with_deployments=False,
        with_pr_titles=False,
    )
    await wait_deferred()

    response = await client.request(
        method="POST", path="/v1/diff/releases", headers=headers, json=body,
    )
    response_body = json.loads((await response.read()).decode("utf-8"))
    assert response.status == 200, response_body
    dr = DiffedReleases.from_dict(response_body)

    assert len(dr.data["github.com/src-d/go-git"]) == 1
    assert len(dr.include.users) == 8
    assert dr.include.jira is None
    assert {r.name for r in dr.data["github.com/src-d/go-git"][0].releases} == {
        "047a795df6d5a0d5dd0782297cea918e4a4a6e10",
        "199317f082082fb8f168ad40a5cae134acfe4a16",
        "31a249d0d5b71bc0f374d3297247d89808263a8b",
        "36c78b9d1b1eea682703fb1cbb0f4f3144354389",
        "4eef16a98d093057f1e4c560da4ed3bbba67cd76",
        "59335b69777f2ef311e63b7d3464459a3ac51d48",
        "5f4169fe242e7c80d779984a86a1de5a1eb78218",
        "62ad629b9a4213fdb8d33bcc7e0bea66d043fc41",
        "cfbd64f09f0d068d593f3dc3beb4ea7e62719e34",
        "e190c37cf51a2a320cabd81b25057859ed689a3b",
        "e512b0280d2747249acecdd8ba33b2ec80d0f364",
        "f51d4a8476f865eef27011a9d90e03566c43d59c",
        "f64e4b856865bc37f45e55ef094060481b53928e",
    }
    assert dr.data["github.com/src-d/go-git"][0].old == "d105e15"
    assert dr.data["github.com/src-d/go-git"][0].new == "31a249d"


@pytest.mark.parametrize(
    "account, repo, old, new, code, body",
    [
        (3, "github.com/src-d/go-git", "v4.0.0", "v4.4.0", 404, None),
        (1, "github.com/athenianco/athenian-api", "v4.0.0", "v4.4.0", 403, None),
        (2, "github.com/src-d/go-git", "v4.0.0", "v4.4.0", 422, None),
        (1, "github.com/src-d/go-git", "whatever", "v4.4.0", 200, None),
        (1, "github.com/src-d/go-git", "v4.0.0", "v4.0.0", 200, []),
        (1, "github.com/src-d/go-git", "v4.4.0", "v4.0.0", 200, None),
    ],
)
async def test_diff_releases_nasty_input(client, headers, account, repo, old, new, code, body):
    body = {
        "account": account,
        "borders": {
            repo: [{"old": old, "new": new}],
        },
    }
    response = await client.request(
        method="POST", path="/v1/diff/releases", headers=headers, json=body,
    )
    response_body = json.loads((await response.read()).decode("utf-8"))
    assert response.status == code, response_body
    if code == 200:
        if body is None:
            assert len(response_body["data"]) == 0
        elif not body:
            assert response_body["data"] == [[{"old": old, "new": new, "releases": []}]]


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_filter_check_runs_smoke(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-01-12",
        "timezone": 60,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/code_checks", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 200, response_text
    check_runs = FilteredCodeCheckRuns.from_dict(json.loads(response_text))
    timeline = check_runs.timeline
    assert timeline == [
        date(2018, 1, 11),
        date(2018, 2, 1),
        date(2018, 3, 1),
        date(2018, 4, 1),
        date(2018, 5, 1),
        date(2018, 6, 1),
        date(2018, 7, 1),
        date(2018, 8, 1),
        date(2018, 9, 1),
        date(2018, 10, 1),
        date(2018, 11, 1),
        date(2018, 12, 1),
        date(2019, 1, 1),
        date(2019, 2, 1),
        date(2019, 3, 1),
        date(2019, 4, 1),
        date(2019, 5, 1),
        date(2019, 6, 1),
        date(2019, 7, 1),
        date(2019, 8, 1),
        date(2019, 9, 1),
        date(2019, 10, 1),
        date(2019, 11, 1),
        date(2019, 12, 1),
        date(2020, 1, 1),
        date(2020, 1, 12),
    ]
    check_runs = check_runs.items
    assert len(check_runs) == 7
    assert {cr.title for cr in check_runs} == {
        "signed-off-by",
        "continuous-integration/travis-ci/push",
        "DCO",
        "codecov/project",
        "continuous-integration/appveyor/pr",
        "continuous-integration/appveyor/branch",
        "continuous-integration/travis-ci/pr",
    }
    assert {cr.repository for cr in check_runs} == {"github.com/src-d/go-git"}
    assert all(
        cr.last_execution_time > datetime(2018, 1, 12, tzinfo=timezone.utc) for cr in check_runs
    )
    assert all(cr.last_execution_url.startswith("https://") for cr in check_runs)
    assert len({cr.last_execution_url for cr in check_runs}) == 7
    assert all(len(cr.size_groups) > 0 for cr in check_runs)
    assert set(chain.from_iterable(cr.size_groups for cr in check_runs)) == {1, 2, 3, 4, 5, 6}
    nn_means = nn_medians = 0
    nn_means_timeline = nn_medians_timeline = nn_count_timeline = nn_successes_timeline = 0
    skips = 0
    for cr in check_runs:
        for stats in (cr.total_stats, cr.prs_stats):
            assert stats.count > 0
            assert stats.flaky_count == 0
            assert 0 <= stats.successes <= stats.count
            if stats.mean_execution_time is not None:
                nn_means += 1
            if stats.median_execution_time is not None:
                nn_medians += 1
            skips += stats.skips
            assert len(stats.mean_execution_time_timeline) == len(timeline) - 1
            nn_means_timeline += sum(
                1 for x in stats.mean_execution_time_timeline if x is not None
            )
            assert len(stats.median_execution_time_timeline) == len(timeline) - 1
            nn_medians_timeline += sum(
                1 for x in stats.median_execution_time_timeline if x is not None
            )
            assert len(stats.count_timeline) == len(timeline) - 1
            nn_count_timeline += sum(1 for x in stats.count_timeline if x > 0)
            assert len(stats.successes_timeline) == len(timeline) - 1
            nn_successes_timeline += sum(1 for x in stats.successes_timeline if x > 0)
            assert stats.count == sum(stats.count_timeline)
            assert stats.successes == sum(stats.successes_timeline)
    assert nn_means > 0
    assert nn_medians > 0
    assert skips == 0  # indeed, go-git has no skips
    assert nn_means_timeline > 0
    assert nn_medians_timeline > 0
    assert nn_count_timeline > 0
    assert nn_successes_timeline > 0


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_filter_check_runs_no_jira(client, headers, sdb):
    await sdb.execute(delete(AccountJiraInstallation))
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-01-12",
        "timezone": 60,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/code_checks", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 200, response_text


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize(
    "filters, count",
    [
        ({"labels_include": ["bug", "plumbing", "enhancement"]}, 4),
        ({"triggered_by": ["github.com/mcuadros"]}, 7),
        ({"triggered_by": ["{1}"]}, 7),
        ({"jira": {"labels_include": ["bug", "onboarding", "performance"]}}, 5),
    ],
)
async def test_filter_check_runs_filters(client, headers, filters, count, sample_team):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-01-12",
        "timezone": 60,
        "in": ["{1}"],
        **filters,
    }
    response = await client.request(
        method="POST", path="/v1/filter/code_checks", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 200, response_text
    check_runs = FilteredCodeCheckRuns.from_dict(json.loads(response_text))
    assert len(check_runs.items) == count


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_filter_check_runs_logical_repos(client, headers, logical_settings_db):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-01-12",
        "timezone": 60,
        "in": ["github.com/src-d/go-git/alpha"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/code_checks", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 200, response_text
    check_runs = FilteredCodeCheckRuns.from_dict(json.loads(response_text))
    assert len(check_runs.items) == 7


@pytest.mark.parametrize(
    "account, date_to, repo, quantiles, status",
    [
        (2, "2020-01-11", "github.com/src-d/go-git", [0, 1], 422),
        (3, "2020-01-11", "github.com/src-d/go-git", [0, 1], 404),
        (1, "2020-01-11", "github.com/athenianco/athenian-api", [0, 1], 403),
        (1, "2020-01-11", "github.com/src-d/go-git", [1, 0], 400),
        (1, "2018-01-11", "github.com/src-d/go-git", [0, 1], 400),
    ],
)
async def test_filter_check_runs_nasty_input(
    client,
    headers,
    account,
    date_to,
    repo,
    quantiles,
    status,
):
    body = {
        "account": account,
        "date_from": "2018-01-12",
        "date_to": date_to,
        "in": [repo],
        "quantiles": quantiles,
    }
    response = await client.request(
        method="POST", path="/v1/filter/code_checks", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == status, response_text


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_filter_deployments_smoke(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-01-12",
        "timezone": 60,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/deployments", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 200, response_text
    deps = FilteredDeployments.from_dict(json.loads(response_text))
    assert len(deps.include.users) == 124
    assert len(deps.include.jira) == 42
    deps = deps.deployments
    assert len(deps) == 1
    assert deps[0].code.to_dict() == {
        "commits_overall": {"github.com/src-d/go-git": 1508},
        "commits_prs": {"github.com/src-d/go-git": 1070},
        "jira": {
            "github.com/src-d/go-git": [
                "DEV-139",
                "DEV-164",
                "DEV-261",
                "DEV-558",
                "DEV-594",
                "DEV-599",
                "DEV-618",
                "DEV-626",
                "DEV-627",
                "DEV-638",
                "DEV-644",
                "DEV-651",
                "DEV-655",
                "DEV-658",
                "DEV-671",
                "DEV-676",
                "DEV-677",
                "DEV-681",
                "DEV-684",
                "DEV-685",
                "DEV-685",
                "DEV-691",
                "DEV-692",
                "DEV-698",
                "DEV-708",
                "DEV-711",
                "DEV-714",
                "DEV-719",
                "DEV-720",
                "DEV-723",
                "DEV-724",
                "DEV-724",
                "DEV-725",
                "DEV-726",
                "DEV-732",
                "DEV-733",
                "DEV-736",
                "DEV-743",
                "DEV-749",
                "DEV-757",
                "DEV-760",
                "DEV-770",
                "DEV-772",
                "DEV-772",
                "DEV-774",
            ],
        },
        "lines_overall": {"github.com/src-d/go-git": 258545},
        "lines_prs": {"github.com/src-d/go-git": 136819},
        "prs": {"github.com/src-d/go-git": 513},
    }


@pytest.mark.parametrize(
    "account, date_from, date_to, repos, status",
    [
        (1, "2018-01-12", "2020-01-12", ["github.com/athenianco/athenian-api"], 403),
        (3, "2018-01-12", "2020-01-12", ["github.com/src-d/go-git"], 404),
        (1, "2020-01-12", "2018-01-12", ["github.com/src-d/go-git"], 400),
        (1, "2018-01-12", "2020-01-12", None, 400),
        (2, "2018-01-12", "2020-01-12", [], 422),
        (None, "2018-01-12", "2020-01-12", ["github.com/src-d/go-git"], 400),
    ],
)
async def test_filter_deployments_nasty_input(
    client,
    headers,
    account,
    date_from,
    date_to,
    repos,
    status,
):
    body = {
        "account": account,
        "date_from": date_from,
        "date_to": date_to,
        "in": repos,
    }
    response = await client.request(
        method="POST", path="/v1/filter/deployments", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == status, response_text


@pytest.mark.parametrize("repos", [None, ["github.com/src-d/go-git"]])
async def test_filter_environments_smoke(client, headers, repos, sample_deployments, rdb):
    body = {
        "account": 1,
        "date_from": "2017-01-01",
        "date_to": "2018-06-01",
        **({"repositories": repos} if repos else {}),
    }
    await rdb.execute(
        delete(DBDeployedComponent).where(
            DBDeployedComponent.deployment_name == "canary_2018_01_12",
        ),
    )
    await rdb.execute(
        delete(DBDeploymentNotification).where(
            DBDeploymentNotification.name == "canary_2018_01_12",
        ),
    )
    response = await client.request(
        method="POST", path="/v1/filter/environments", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 200, response_text
    envs = [FilteredEnvironment.from_dict(x) for x in json.loads(response_text)]
    assert envs == [
        FilteredEnvironment(
            name="canary",
            deployments_count=2,
            last_conclusion="SUCCESS",
            repositories=["github.com/src-d/go-git"],
        ),
        FilteredEnvironment(
            name="production",
            deployments_count=3,
            last_conclusion="FAILURE",
            repositories=["github.com/src-d/go-git"],
        ),
        FilteredEnvironment(
            name="staging",
            deployments_count=3,
            last_conclusion="FAILURE",
            repositories=["github.com/src-d/go-git"],
        ),
    ]


@pytest.mark.parametrize(
    "account, date_from, date_to, repos, status",
    [
        (1, "2018-01-12", "2020-01-12", ["github.com/athenianco/athenian-api"], 403),
        (3, "2018-01-12", "2020-01-12", ["github.com/src-d/go-git"], 404),
        (1, "2020-01-12", "2018-01-12", ["github.com/src-d/go-git"], 400),
        (1, "2018-01-12", "2020-01-12", None, 400),
        (2, "2018-01-12", "2020-01-12", [], 422),
        (None, "2018-01-12", "2020-01-12", ["github.com/src-d/go-git"], 400),
    ],
)
async def test_filter_environments_nasty_input(
    client,
    headers,
    sample_deployments,
    account,
    date_from,
    date_to,
    repos,
    status,
):
    body = {
        "account": account,
        "date_from": date_from,
        "date_to": date_to,
        "repositories": repos,
    }
    response = await client.request(
        method="POST", path="/v1/filter/environments", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == status, response_text


async def test_filter_environments_422(client, headers, rdb):
    body = {
        "account": 1,
        "date_from": "2016-01-01",
        "date_to": "2020-01-01",
    }
    await rdb.execute(delete(DBDeployedComponent))
    await rdb.execute(delete(DBDeploymentNotification))
    response = await client.request(
        method="POST", path="/v1/filter/environments", headers=headers, json=body,
    )
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 422, response_text
