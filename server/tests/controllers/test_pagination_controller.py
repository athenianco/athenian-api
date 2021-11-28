from datetime import date
import json

import pytest
from sqlalchemy import insert

from athenian.api.controllers.settings import ReleaseMatch
from athenian.api.models.state.models import ReleaseSetting
from athenian.api.models.web import PullRequestPaginationPlan, PullRequestStage


@pytest.mark.filter_pull_requests
@pytest.mark.parametrize("batch, count",
                         [(100, 8),
                          (500, 3)])
async def test_paginate_prs_smoke(client, headers, batch, count):
    print("filter", flush=True)
    main_request = {
        "date_from": "2015-10-13",
        "date_to": "2020-04-01",
        "account": 1,
        "in": [],
        "stages": list(PullRequestStage),
        "exclude_inactive": True,
    }
    # populate pdb
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=main_request)
    assert response.status == 200
    await response.read()
    print("paginate", flush=True)
    body = {
        "request": main_request,
        "batch": batch,
    }
    response = await client.request(
        method="POST", path="/v1/paginate/pull_requests", headers=headers, json=body)
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    obj = json.loads(text)
    try:
        model = PullRequestPaginationPlan.from_dict(obj)
    except Exception as e:
        raise ValueError(text) from e
    assert len(model.updated) == count, model.updated
    for i in range(count - 1):
        assert isinstance(model.updated[i], date)
        assert isinstance(model.updated[i + 1], date)
        assert model.updated[i] > model.updated[i + 1]
    assert model.updated[0] < date(2020, 4, 1)
    assert model.updated[-1] > date(2015, 10, 12)


@pytest.mark.filter_pull_requests
@pytest.mark.parametrize("account, batch, stages, repos, code",
                         [(1, 0, list(PullRequestStage), [], 400),
                          (1, -10, list(PullRequestStage), [], 400),
                          (1, None, list(PullRequestStage), [], 400),
                          (3, 1, list(PullRequestStage), [], 404),
                          (1, 1, list(PullRequestStage), ["github.com/a/b"], 403),
                          (3, 1, None, [], 400)])
async def test_paginate_prs_nasty_input(client, headers, account, batch, stages, repos, code):
    body = {
        "request": {
            "date_from": "2015-10-13",
            "date_to": "2020-04-01",
            "account": account,
            "in": [*repos],
            "stages": stages,
            "exclude_inactive": True,
        },
        "batch": batch,
    }
    response = await client.request(
        method="POST", path="/v1/paginate/pull_requests", headers=headers, json=body)
    text = (await response.read()).decode("utf-8")
    assert response.status == code, text


async def test_paginate_prs_jira(client, headers):
    print("filter", flush=True)
    main_request = {
        "date_from": "2017-10-13",
        "date_to": "2018-04-01",
        "account": 1,
        "in": [],
        "jira": {
            "labels_include": ["bug", "enhancement"],
        },
        "stages": list(PullRequestStage),
        "exclude_inactive": True,
    }
    # populate pdb
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=main_request)
    assert response.status == 200
    await response.read()
    print("paginate", flush=True)
    body = {
        "request": main_request,
        "batch": 1,
    }
    response = await client.request(
        method="POST", path="/v1/paginate/pull_requests", headers=headers, json=body)
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    obj = json.loads(text)
    try:
        model = PullRequestPaginationPlan.from_dict(obj)
    except Exception as e:
        raise ValueError(text) from e
    assert model.updated == [date(2018, 4, 3), date(2018, 1, 16)]
    main_request["jira"]["labels_include"] = ["nope"]
    response = await client.request(
        method="POST", path="/v1/paginate/pull_requests", headers=headers, json=body)
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    obj = json.loads(text)
    try:
        model = PullRequestPaginationPlan.from_dict(obj)
    except Exception as e:
        raise ValueError(text) from e
    assert model.updated == [date(2018, 4, 1), date(2017, 10, 13)]


async def test_paginate_prs_no_done(client, headers):
    print("filter", flush=True)
    main_request = {
        "date_from": "2020-01-12",
        "date_to": "2020-03-01",
        "account": 1,
        "in": [],
        "stages": list(PullRequestStage),
        "exclude_inactive": True,
    }
    # populate pdb
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=main_request)
    assert response.status == 200
    await response.read()
    print("paginate", flush=True)
    body = {
        "request": main_request,
        "batch": 5,
    }
    response = await client.request(
        method="POST", path="/v1/paginate/pull_requests", headers=headers, json=body)
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    obj = json.loads(text)
    try:
        model = PullRequestPaginationPlan.from_dict(obj)
    except Exception as e:
        raise ValueError(text) from e
    assert model.updated == [date(2020, 3, 10), date(2020, 2, 28), date(2020, 1, 13)]
    main_request["stages"] = ["done"]
    response = await client.request(
        method="POST", path="/v1/paginate/pull_requests", headers=headers, json=body)
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    obj = json.loads(text)
    try:
        model = PullRequestPaginationPlan.from_dict(obj)
    except Exception as e:
        raise ValueError(text) from e
    # exactly the same even though there should be 0 PRs
    # we ignore the stages
    assert model.updated == [date(2020, 3, 10), date(2020, 2, 28), date(2020, 1, 13)]


async def test_paginate_prs_empty(client, headers):
    print("filter", flush=True)
    main_request = {
        "date_from": "2015-01-01",
        "date_to": "2015-01-01",
        "account": 1,
        "in": [],
        "stages": list(PullRequestStage),
        "exclude_inactive": True,
    }
    # populate pdb
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=main_request)
    assert response.status == 200
    await response.read()
    print("paginate", flush=True)
    body = {
        "request": main_request,
        "batch": 5,
    }
    response = await client.request(
        method="POST", path="/v1/paginate/pull_requests", headers=headers, json=body)
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    obj = json.loads(text)
    try:
        model = PullRequestPaginationPlan.from_dict(obj)
    except Exception as e:
        raise ValueError(text) from e
    assert model.updated == [date(2015, 1, 1), date(2015, 1, 1)]


async def test_paginate_prs_same_day(client, headers, sdb):
    await sdb.execute(insert(ReleaseSetting).values(ReleaseSetting(
        repository="github.com/src-d/go-git",
        account_id=1,
        branches="",
        tags=".*",
        events=".*",
        match=ReleaseMatch.tag,
    ).create_defaults().explode(with_primary_keys=True)))
    print("filter", flush=True)
    main_request = {
        "date_from": "2015-10-23",
        "date_to": "2015-10-23",
        "account": 1,
        "in": [],
        "stages": list(PullRequestStage),
        "exclude_inactive": True,
    }
    # populate pdb
    response = await client.request(
        method="POST", path="/v1/filter/pull_requests", headers=headers, json=main_request)
    assert response.status == 200
    await response.read()
    print("paginate", flush=True)
    body = {
        "request": main_request,
        "batch": 5,
    }
    response = await client.request(
        method="POST", path="/v1/paginate/pull_requests", headers=headers, json=body)
    text = (await response.read()).decode("utf-8")
    assert response.status == 200, text
    obj = json.loads(text)
    try:
        model = PullRequestPaginationPlan.from_dict(obj)
    except Exception as e:
        raise ValueError(text) from e
    assert model.updated == [date(2015, 12, 13), date(2015, 12, 13)]
