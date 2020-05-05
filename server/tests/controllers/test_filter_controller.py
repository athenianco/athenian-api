from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from typing import Set

from aiohttp import ClientResponse
import dateutil
import numpy as np
from prometheus_client import CollectorRegistry
import pytest

from athenian.api import setup_cache_metrics
from athenian.api.controllers.miners.pull_request_list_item import Property
from athenian.api.models.web import CommitsList, PullRequestSet
from athenian.api.models.web.filtered_releases import FilteredReleases
from athenian.api.models.web.pull_request_participant import PullRequestParticipant
from athenian.api.models.web.pull_request_property import PullRequestProperty
from tests.conftest import FakeCache


async def test_filter_repositories_no_repos(client, headers):
    body = {
        "date_from": "2015-01-12",
        "date_to": "2015-01-12",
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    assert response.status == 200
    repos = json.loads((await response.read()).decode("utf-8"))
    assert repos == []


async def test_filter_repositories_smoke(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "timezone": 60,
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


@pytest.mark.parametrize("account, date_to, code",
                         [(3, "2020-01-23", 403), (10, "2020-01-23", 403), (1, "2015-10-13", 200),
                          (1, "2010-01-11", 400), (1, "2020-01-32", 400)])
async def test_filter_repositories_nasty_input(client, headers, account, date_to, code):
    body = {
        "date_from": "2015-10-13",
        "date_to": date_to,
        "account": account,
    }
    response = await client.request(
        method="POST", path="/v1/filter/repositories", headers=headers, json=body)
    assert response.status == code


@pytest.mark.parametrize("in_", [{}, {"in": []}])
async def test_filter_contributors_no_repos(client, headers, in_):
    body = {
        "date_from": "2015-01-12",
        "date_to": "2020-01-23",
        "account": 1,
        **in_,
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    contribs = json.loads((await response.read()).decode("utf-8"))
    assert len(contribs) == 202
    assert len(set(c["login"] for c in contribs)) == len(contribs)
    assert all(c["login"].startswith("github.com/") for c in contribs)
    contribs = {c["login"]: c for c in contribs}
    assert "github.com/mcuadros" in contribs
    body["date_to"] = body["date_from"]
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    assert response.status == 200
    contribs = json.loads((await response.read()).decode("utf-8"))
    assert contribs == []


async def test_filter_contributors(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "timezone": 60,
        "account": 1,
        "in": ["github.com/src-d/go-git"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    contribs = json.loads((await response.read()).decode("utf-8"))
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
    assert topics == {"prs", "commenter", "commit_author", "commit_committer", "reviewer",
                      "releaser"}
    body["in"] = ["github.com/src-d/gitbase"]
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    contribs = json.loads((await response.read()).decode("utf-8"))
    assert contribs == []


@pytest.mark.parametrize("account, date_to, code",
                         [(3, "2020-01-23", 403), (10, "2020-01-23", 403), (1, "2015-10-13", 200),
                          (1, "2010-01-11", 400), (1, "2020-01-32", 400)])
async def test_filter_contributors_nasty_input(client, headers, account, date_to, code):
    body = {
        "date_from": "2015-10-13",
        "date_to": date_to,
        "account": account,
    }
    response = await client.request(
        method="POST", path="/v1/filter/contributors", headers=headers, json=body)
    assert response.status == code


@pytest.fixture(scope="module")
def filter_prs_single_prop_cache():
    fc = FakeCache()
    setup_cache_metrics(fc, CollectorRegistry(auto_describe=True))
    return fc


@pytest.mark.parametrize("prop", [k.name.lower() for k in Property])
async def test_filter_prs_single_prop(client, headers, prop, app, filter_prs_single_prop_cache):
    app._cache = filter_prs_single_prop_cache
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "account": 1,
        "in": [],
        "properties": [prop],
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body)
    await validate_prs_response(response, {prop})


async def test_filter_prs_all_properties(client, headers):
    body = {
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "timezone": 60,
        "account": 1,
        "in": [],
        "properties": [],
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body)
    await validate_prs_response(response, set(PullRequestProperty))
    del body["properties"]
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body)
    await validate_prs_response(response, set(PullRequestProperty))


@pytest.mark.parametrize("timezone, must_match", [(120, True), (60, True), (0, False)])
async def test_filter_prs_merged_timezone(client, headers, timezone, must_match):
    body = {
        "date_from": "2017-07-08",
        "date_to": "2017-07-10",
        "timezone": timezone,
        "account": 1,
        "in": [],
        "properties": [PullRequestProperty.MERGE_HAPPENED],
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body)
    assert response.status == 200
    obj = json.loads((await response.read()).decode("utf-8"))
    prs = PullRequestSet.from_dict(obj)  # type: PullRequestSet
    matched = False
    for pr in prs.data:
        if pr.number == 467:  # merged 2017-07-08 01:37 GMT+2 = 2017-07-07 23:37 UTC
            matched = True
    assert matched == must_match


@pytest.mark.parametrize("timezone, must_match", [(-7 * 60, False), (-8 * 60, True)])
async def test_filter_prs_created_timezone(client, headers, timezone, must_match):
    body = {
        "date_from": "2017-07-15",
        "date_to": "2017-07-16",
        "timezone": timezone,
        "account": 1,
        "in": [],
        "properties": [],
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body)
    assert response.status == 200
    obj = json.loads((await response.read()).decode("utf-8"))
    prs = PullRequestSet.from_dict(obj)  # type: PullRequestSet
    matched = False
    for pr in prs.data:
        if pr.number == 485:  # created 2017-07-17 09:02 GMT+2 = 2017-07-17 07:02 UTC
            matched = True
    assert matched == must_match


async def validate_prs_response(response: ClientResponse, props: Set[str]):
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    obj = json.loads(text)
    users = obj["include"]["users"]
    assert len(users) > 0
    assert len(obj["data"]) > 0
    statuses = defaultdict(int)
    mentioned_users = set()
    comments = 0
    commits = 0
    review_comments = 0
    release_urls = 0
    timestamps = defaultdict(bool)
    response_props = defaultdict(bool)
    stage_timings = defaultdict(int)
    stages = {"wip": 0, "review": 1, "merge": 2, "release": 3}
    for pr in obj["data"]:
        assert pr["repository"].startswith("github.com/"), str(pr)
        assert pr["number"] > 0
        assert pr["title"]
        assert pr["size_added"] + pr["size_removed"] >= 0, str(pr)
        assert pr["files_changed"] >= 0, str(pr)
        assert pr["created"], str(pr)
        for k in ("closed", "updated", "merged", "released", "review_requested", "approved"):
            timestamps[k] |= bool(pr.get(k))
        if pr.get("merged"):
            assert pr["closed"], str(pr)
        if pr.get("released"):
            assert pr["merged"], str(pr)
        assert props.intersection(set(pr["properties"]))
        comments += pr["comments"]
        commits += pr["commits"]
        review_comments += pr["review_comments"]
        release_urls += bool(pr.get("release_url"))
        for prop in pr["properties"]:
            response_props[prop] = True
        reported_timings = np.zeros(4, dtype=int)
        for k, v in pr["stage_timings"].items():
            reported_timings[stages[k]] = 1
            stage_timings[k] += int(v[:-1])
        diff = np.diff(reported_timings)
        assert (diff == -1).sum() <= 1 or (reported_timings[:3] == [1, 0, 1]).all(), \
            str(pr["stage_timings"])
        participants = pr["participants"]
        assert len(participants) > 0
        authors = 0
        for p in participants:
            assert p["id"].startswith("github.com/")
            mentioned_users.add(p["id"])
            is_author = PullRequestParticipant.STATUS_AUTHOR in p["status"]
            authors += is_author
            if is_author:
                assert PullRequestParticipant.STATUS_REVIEWER not in p["status"], pr["number"]
            for s in p["status"]:
                statuses[s] += 1
                assert s in PullRequestParticipant.STATUSES
        if pr["number"] != 749:
            # the author of 749 is deleted on GitHub
            assert authors == 1
    assert not (set(users) - mentioned_users)
    assert commits > 0
    assert timestamps["updated"]
    if PullRequestProperty.WIP in props:
        assert statuses[PullRequestParticipant.STATUS_COMMIT_COMMITTER] > 0
        assert statuses[PullRequestParticipant.STATUS_COMMIT_AUTHOR] > 0
        assert response_props.get(PullRequestProperty.WIP)
        assert response_props.get(PullRequestProperty.CREATED)
        assert response_props.get(PullRequestProperty.COMMIT_HAPPENED)
        assert "wip" in stage_timings
    if PullRequestProperty.REVIEWING in props:
        assert comments > 0
        assert review_comments > 0
        assert statuses[PullRequestParticipant.STATUS_REVIEWER] > 0
        assert response_props.get(PullRequestProperty.REVIEWING)
        assert response_props.get(PullRequestProperty.REVIEW_HAPPENED)
        assert response_props.get(PullRequestProperty.COMMIT_HAPPENED)
        assert response_props.get(PullRequestProperty.REVIEW_REQUEST_HAPPENED)
        assert response_props.get(PullRequestProperty.CHANGES_REQUEST_HAPPENED)
        assert "wip" in stage_timings
        assert "review" in stage_timings
    if PullRequestProperty.MERGING in props:
        assert timestamps["review_requested"]
        assert timestamps["approved"]
        assert comments > 0
        assert review_comments >= 0
        assert statuses[PullRequestParticipant.STATUS_REVIEWER] > 0
        assert response_props.get(PullRequestProperty.MERGING)
        assert response_props.get(PullRequestProperty.COMMIT_HAPPENED)
        assert response_props.get(PullRequestProperty.REVIEW_HAPPENED)
        assert response_props.get(PullRequestProperty.APPROVE_HAPPENED)
        assert "wip" in stage_timings
        assert "review" in stage_timings
        assert "merge" in stage_timings
    if PullRequestProperty.RELEASING in props:
        assert timestamps["review_requested"]
        assert timestamps["approved"]
        assert comments > 0
        assert review_comments >= 0
        assert statuses[PullRequestParticipant.STATUS_REVIEWER] > 0
        assert statuses[PullRequestParticipant.STATUS_MERGER] > 0
        assert response_props.get(PullRequestProperty.RELEASING)
        assert response_props.get(PullRequestProperty.MERGE_HAPPENED)
        assert timestamps["merged"]
        assert "wip" in stage_timings
        assert "review" in stage_timings
        assert "merge" in stage_timings
        assert "release" in stage_timings
    if PullRequestProperty.DONE in props:
        assert timestamps["review_requested"]
        assert timestamps["approved"]
        assert comments > 0
        assert review_comments > 0
        assert statuses[PullRequestParticipant.STATUS_REVIEWER] > 0
        assert statuses[PullRequestParticipant.STATUS_MERGER] > 0
        assert statuses[PullRequestParticipant.STATUS_RELEASER] > 0
        assert response_props.get(PullRequestProperty.DONE)
        assert response_props.get(PullRequestProperty.RELEASE_HAPPENED)
        assert timestamps["released"]
        assert timestamps["closed"]
        assert "wip" in stage_timings
        assert "review" in stage_timings
        assert "merge" in stage_timings
        assert "release" in stage_timings


@pytest.mark.parametrize("account, date_to, code",
                         [(3, "2020-01-23", 403), (10, "2020-01-23", 403), (1, "2015-10-13", 200),
                          (1, "2010-01-11", 400), (1, "2020-01-32", 400)])
async def test_filter_prs_nasty_input(client, headers, account, date_to, code):
    body = {
        "date_from": "2015-10-13",
        "date_to": date_to,
        "account": account,
        "in": [],
        "properties": [],
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body)
    assert response.status == code


async def test_filter_prs_david_bug(client, headers):
    body = {
        "account": 1,
        "date_from": "2019-02-22",
        "date_to": "2020-02-22",
        "in": ["github.com/src-d/go-git"],
        "properties": ["wip", "reviewing", "merging", "releasing"],
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


async def test_filter_prs_developer_filter(client, headers):
    body = {
        "date_from": "2017-07-15",
        "date_to": "2017-12-16",
        "account": 1,
        "in": [],
        "properties": [],
        "with": {
            "author": ["github.com/mcuadros"],
        },
    }
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=body)
    assert response.status == 200
    obj = json.loads((await response.read()).decode("utf-8"))
    prs = PullRequestSet.from_dict(obj)  # type: PullRequestSet
    assert len(prs.data) == 27
    for pr in prs.data:
        passed = False
        for part in pr.participants:
            if part.id == "github.com/mcuadros":
                assert PullRequestParticipant.STATUS_AUTHOR in part.status
                passed = True
        assert passed


async def test_filter_commits_bypassing_prs_mcuadros(client, headers):
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
        method="POST", path="/v1/filter/commits", headers=headers, json=body)
    assert response.status == 200
    commits = CommitsList.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert commits.to_dict() == {
        "data": [{"author": {"email": "mcuadros@gmail.com",
                             "login": "github.com/mcuadros",
                             "name": "Máximo Cuadros",
                             "timestamp": datetime(2019, 4, 24, 13, 20, 51, tzinfo=timezone.utc),
                             "timezone": 2.0},
                  "committer": {"email": "mcuadros@gmail.com",
                                "login": "github.com/mcuadros",
                                "name": "Máximo Cuadros",
                                "timestamp": datetime(2019, 4, 24, 13, 20, 51,
                                                      tzinfo=timezone.utc),
                                "timezone": 2.0},
                  "files_changed": 1,
                  "hash": "5c6d199dc675465f5e103ea36c0bfcb9d3ebc565",
                  "message": "plumbing: commit.Stats, fix panic on empty chucks\n\n"
                             "Signed-off-by: Máximo Cuadros <mcuadros@gmail.com>",
                  "repository": "src-d/go-git",
                  "size_added": 4,
                  "size_removed": 0}],
        "include": {"users": {
            "github.com/mcuadros": {
                "avatar": "https://avatars0.githubusercontent.com/u/1573114?s=600&v=4"}}}}


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
        method="POST", path="/v1/filter/commits", headers=headers, json=body)
    assert response.status == 200
    commits = CommitsList.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert len(commits.data) == 6
    assert len(commits.include.users) == 1
    for c in commits.data:
        assert c.author.login == "github.com/mcuadros"
        assert c.committer.login == "github.com/mcuadros"


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
        method="POST", path="/v1/filter/commits", headers=headers, json=body)
    assert response.status == 200
    commits = CommitsList.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert len(commits.data) == 25
    for c in commits.data:
        assert c.committer.email != "noreply@github.com"


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
        method="POST", path="/v1/filter/commits", headers=headers, json=body)
    assert response.status == 200
    commits = CommitsList.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert len(commits.data) == 0
    assert len(commits.include.users) == 0


async def test_filter_commits_bypassing_prs_no_with(client, headers):
    body = {
        "account": 1,
        "date_from": "2020-01-12",
        "date_to": "2020-02-21",
        "in": ["{1}"],
        "property": "bypassing_prs",
    }
    response = await client.request(
        method="POST", path="/v1/filter/commits", headers=headers, json=body)
    assert response.status == 200
    commits = CommitsList.from_dict(
        json.loads((await response.read()).decode("utf-8")))  # type: CommitsList
    assert len(commits.data) == 0
    assert len(commits.include.users) == 0
    body["date_to"] = "2020-02-22"
    response = await client.request(
        method="POST", path="/v1/filter/commits", headers=headers, json=body)
    assert response.status == 200
    commits = CommitsList.from_dict(json.loads((await response.read()).decode("utf-8")))
    assert len(commits.data) == 1
    assert commits.data[0].committer.timestamp == datetime(2020, 2, 22, 18, 58, 50,
                                                           tzinfo=dateutil.tz.tzutc())


@pytest.mark.parametrize("account, date_to, code",
                         [(3, "2020-02-22", 403), (10, "2020-02-22", 403), (1, "2020-01-12", 200),
                          (1, "2010-01-11", 400), (1, "2020-02-32", 400)])
async def test_filter_commits_bypassing_prs_nasty_input(client, headers, account, date_to, code):
    body = {
        "account": account,
        "date_from": "2020-01-12",
        "date_to": date_to,
        "in": ["{1}"],
        "property": "bypassing_prs",
    }
    response = await client.request(
        method="POST", path="/v1/filter/commits", headers=headers, json=body)
    assert response.status == code


async def test_filter_releases_by_tag(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-01-12",
        "timezone": 60,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/releases", headers=headers, json=body)
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 200, response_text
    releases = FilteredReleases.from_dict(json.loads(response_text))  # type: FilteredReleases
    releases.include.users = set(releases.include.users)
    assert len(releases.include.users) == 71
    assert "github.com/mcuadros" in releases.include.users
    assert len(releases.data) == 21
    for release in releases.data:
        assert release.publisher.startswith("github.com/"), str(release)
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
        assert release.published >= datetime(year=2018, month=1, day=12, tzinfo=timezone.utc), \
            str(release)
        assert release.repository.startswith("github.com/"), str(release)


async def test_filter_releases_by_branch(client, headers, cache, app):
    app._cache = cache
    body = {
        "account": 1,
        "date_from": "2015-01-01",
        "date_to": "2020-10-22",
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/releases", headers=headers, json=body)
    response_text = (await response.read()).decode("utf-8")
    assert response.status == 200, response_text


@pytest.mark.parametrize("account, date_to, code",
                         [(3, "2020-02-22", 403), (10, "2020-02-22", 403), (1, "2020-01-12", 200),
                          (1, "2010-01-11", 400), (1, "2020-02-32", 400)])
async def test_filter_releases_nasty_input(client, headers, account, date_to, code):
    body = {
        "account": account,
        "date_from": "2020-01-12",
        "date_to": date_to,
        "in": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/releases", headers=headers, json=body)
    assert response.status == code
