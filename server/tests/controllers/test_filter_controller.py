from collections import defaultdict
import json
from typing import Set

from aiohttp import ClientResponse
from prometheus_client import CollectorRegistry
import pytest

from athenian.api import setup_cache_metrics
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
    assert response.status == 200
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
    body["in"] = ["github.com/src-d/gitbase"]
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


@pytest.mark.parametrize("in_", [{}, {"in": []}])
async def test_filter_contributors_no_repos(client, headers, in_):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
        **in_,
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
    assert response.status == 200
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
    body["in"] = ["github.com/src-d/gitbase"]
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
    fc = FakeCache()
    setup_cache_metrics(fc, CollectorRegistry(auto_describe=True))
    return fc


@pytest.mark.parametrize("stage", [k.name.lower() for k in Stage])
async def test_filter_prs_single_stage(client, headers, stage, app, filter_prs_single_stage_cache):
    if stage == PullRequestPipelineStage.DONE:
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
    obj = json.loads((await response.read()).decode("utf-8"))
    users = obj["include"]["users"]
    assert len(users) > 0
    assert len(obj["data"]) > 0
    statuses = defaultdict(int)
    mentioned_users = set()
    comments = 0
    commits = 0
    review_requested = False
    review_comments = 0
    merged = False
    for pr in obj["data"]:
        if pr["number"] == 639:
            assert pr["stage"] == "done"
        assert pr["repository"].startswith("github.com/"), str(pr)
        assert pr["number"] > 0
        assert pr["title"]
        assert pr["size_added"] + pr["size_removed"] >= 0, str(pr)
        assert pr["files_changed"] >= 0, str(pr)
        assert pr["created"], str(pr)
        assert pr["updated"], str(pr)
        if pr["merged"]:
            assert pr["closed"], str(pr)
        assert pr["stage"] in stages
        comments += pr["comments"]
        commits += pr["commits"]
        review_requested |= pr["review_requested"]
        review_comments += pr["review_comments"]
        merged |= pr["merged"]
        participants = pr["participants"]
        assert len(participants) > 0
        authors = 0
        for p in participants:
            assert p["id"].startswith("github.com/")
            mentioned_users.add(p["id"])
            for s in p["status"]:
                statuses[s] += 1
                authors += s == PullRequestParticipant.STATUS_AUTHOR
                assert s in PullRequestParticipant.STATUSES
        assert authors == 1
    assert not (set(users) - mentioned_users)
    assert commits > 0
    if PullRequestPipelineStage.WIP in stages:
        assert statuses[PullRequestParticipant.STATUS_COMMIT_COMMITTER] > 0
        assert statuses[PullRequestParticipant.STATUS_COMMIT_AUTHOR] > 0
    elif PullRequestPipelineStage.REVIEW in stages:
        assert review_requested
        assert comments > 0
        assert review_comments > 0
        assert statuses[PullRequestParticipant.STATUS_REVIEWER] > 0
    elif PullRequestPipelineStage.MERGE in stages or PullRequestPipelineStage.RELEASE in stages:
        assert review_requested
        assert comments > 0
        assert review_comments >= 0
        assert statuses[PullRequestParticipant.STATUS_MERGER] > 0
    elif PullRequestPipelineStage.DONE in stages:
        assert review_requested
        assert comments > 0
        assert review_comments > 0
        assert merged


async def test_filter_prs_bad_account(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 3,
        "stages": [],
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body)
    assert response.status == 403


async def test_filter_prs_access_denied(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
        "in": ["github.com/athenianco/athenian-api"],
        "stages": [],
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body)
    assert response.status == 403


async def test_filter_prs_david_bug(client, headers):
    body = {
        "account": 1,
        "date_from": "2019-02-22",
        "date_to": "2020-02-22",
        "in": ["github.com/src-d/go-git"],
        "stages": ["wip", "review", "merge", "release"],
        "with": {
            "author": ["github.com/Junnplus"],
            "reviewer": ["github.com/Junnplus"],
            "commit_author": ["github.com/Junnplus"],
            "commit_committer": ["github.com/Junnplus"],
            "commenter": ["github.com/Junnplus"],
            "merger": ["github.com/Junnplus"],
        },
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body)
    assert response.status == 200
