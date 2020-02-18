from collections import defaultdict
import json
from typing import Set

from aiohttp import ClientResponse
import pytest

from athenian.api.controllers.miners.pull_request_list_item import Stage
from athenian.api.models.web.pull_request_participant import PullRequestParticipant
from athenian.api.models.web.pull_request_pipeline_stage import PullRequestPipelineStage
from tests.conftest import FakeCache


async def test_filter_repositories_no_repos(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == ["github.com/src-d/go-git"]
    body["date_from"] = body["date_to"]
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == []


async def test_filter_repositories(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
        "in": ["github.com/src-d/go-git"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == ["github.com/src-d/go-git"]
    body["in"] = ["github.com/athenianco/athenian-api"]
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == []


async def test_filter_repositories_bad_account(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 3,
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    assert response.status == 403
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 2,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    assert response.status == 403
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    assert response.status == 200


async def test_filter_contributors_no_repos(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    contribs = json.loads((await response.read()).decode("utf-8"))
    assert len(contribs) == 179
    assert len(set(contribs)) == len(contribs)
    assert all(c.startswith("github.com/") for c in contribs)
    assert "github.com/mcuadros" in contribs
    body["date_from"] = body["date_to"]
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    contribs = json.loads((await response.read()).decode("utf-8"))
    assert contribs == []


async def test_filter_contributors(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
        "in": ["github.com/src-d/go-git"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    contribs = json.loads((await response.read()).decode("utf-8"))
    assert len(contribs) == 179
    assert len(set(contribs)) == len(contribs)
    assert all(c.startswith("github.com/") for c in contribs)
    assert "github.com/mcuadros" in contribs
    body["in"] = ["github.com/athenianco/athenian-api"]
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    contribs = json.loads((await response.read()).decode("utf-8"))
    assert contribs == []


async def test_filter_contributors_bad_account(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 3,
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    assert response.status == 403
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 2,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    assert response.status == 403
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    assert response.status == 200


@pytest.fixture(scope="module")
def filter_prs_single_stage_cache():
    return FakeCache()


@pytest.mark.parametrize("stage", [k.name.lower() for k in Stage])
async def test_filter_prs_single_stage(client, headers, stage, app, filter_prs_single_stage_cache):
    if stage == "gold":
        pytest.skip("no releases data")
    app._cache = filter_prs_single_stage_cache
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
        "stages": [stage],
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body)
    await validate_prs_response(response, {stage})


async def test_filter_prs_all_stages(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
        "stages": [],
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body)
    await validate_prs_response(response, PullRequestPipelineStage.ALL)
    del body["stages"]
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body)
    await validate_prs_response(response, PullRequestPipelineStage.ALL)


async def validate_prs_response(response: ClientResponse, stages: Set[str]):
    assert response.status == 200
    prs = json.loads((await response.read()).decode("utf-8"))
    assert len(prs) > 0
    statuses = defaultdict(int)
    for pr in prs:
        assert pr["repository"].startswith("github.com/"), str(pr)
        assert pr["number"] > 0
        assert pr["title"]
        assert pr["size_added"] + pr["size_removed"] >= 0, str(pr)
        assert pr["files_changed"] >= 0, str(pr)
        assert pr["created"], str(pr)
        assert pr["updated"], str(pr)
        assert pr["stage"] in stages
        participants = pr["participants"]
        assert len(participants) > 0
        authors = 0
        for p in participants:
            assert p["id"].startswith("github.com/")
            for s in p["status"]:
                statuses[s] += 1
                authors += s == PullRequestParticipant.STATUS_AUTHOR
                assert s in PullRequestParticipant.STATUSES
        assert authors == 1
    if "wip" in stages:
        assert statuses[PullRequestParticipant.STATUS_COMMIT_COMMITTER] > 0
        assert statuses[PullRequestParticipant.STATUS_COMMIT_AUTHOR] > 0
    elif "review" in stages:
        assert statuses[PullRequestParticipant.STATUS_REVIEWER] > 0
    elif "merge" in stages or "release" in stages:
        assert statuses[PullRequestParticipant.STATUS_MERGER] > 0
