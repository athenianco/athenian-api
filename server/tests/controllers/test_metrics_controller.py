from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import delete, insert

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github import developer
from athenian.api.internal.miners.github.release_mine import mine_releases, \
    override_first_releases
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseMatch
from athenian.api.models.persistentdata.models import DeployedComponent, DeployedLabel
from athenian.api.models.state.models import ReleaseSetting, Team
from athenian.api.models.web import CalculatedCodeCheckMetrics, CalculatedDeploymentMetric, \
    CalculatedDeveloperMetrics, CalculatedLinearMetricValues, CalculatedPullRequestMetrics, \
    CalculatedReleaseMetric, CodeBypassingPRsMeasurement, CodeCheckMetricID, DeploymentMetricID, \
    DeveloperMetricID, PullRequestMetricID, PullRequestWith, ReleaseMetricID
from athenian.api.serialization import FriendlyJson


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize(
    "metric, count", [
        (PullRequestMetricID.PR_WIP_TIME, 51),
        (PullRequestMetricID.PR_WIP_PENDING_COUNT, 0),
        (PullRequestMetricID.PR_WIP_COUNT, 51),
        (PullRequestMetricID.PR_WIP_COUNT_Q, 51),
        (PullRequestMetricID.PR_REVIEW_TIME, 46),
        (PullRequestMetricID.PR_REVIEW_PENDING_COUNT, 0),
        (PullRequestMetricID.PR_REVIEW_COUNT, 46),
        (PullRequestMetricID.PR_REVIEW_COUNT_Q, 46),
        (PullRequestMetricID.PR_MERGING_TIME, 51),
        (PullRequestMetricID.PR_MERGING_PENDING_COUNT, 0),
        (PullRequestMetricID.PR_MERGING_COUNT, 51),
        (PullRequestMetricID.PR_MERGING_COUNT_Q, 51),
        (PullRequestMetricID.PR_RELEASE_TIME, 19),
        (PullRequestMetricID.PR_RELEASE_PENDING_COUNT, 189),
        (PullRequestMetricID.PR_RELEASE_COUNT, 19),
        (PullRequestMetricID.PR_RELEASE_COUNT_Q, 19),
        (PullRequestMetricID.PR_LEAD_TIME, 19),
        (PullRequestMetricID.PR_LEAD_COUNT, 19),
        (PullRequestMetricID.PR_LEAD_COUNT_Q, 19),
        (PullRequestMetricID.PR_CYCLE_TIME, 71),
        (PullRequestMetricID.PR_CYCLE_COUNT, 71),
        (PullRequestMetricID.PR_CYCLE_COUNT_Q, 71),
        (PullRequestMetricID.PR_ALL_COUNT, 200),
        (PullRequestMetricID.PR_FLOW_RATIO, 224),
        (PullRequestMetricID.PR_OPENED, 51),
        (PullRequestMetricID.PR_REVIEWED, 45),
        (PullRequestMetricID.PR_NOT_REVIEWED, 19),
        (PullRequestMetricID.PR_MERGED, 50),
        (PullRequestMetricID.PR_REJECTED, 3),
        (PullRequestMetricID.PR_CLOSED, 51),
        (PullRequestMetricID.PR_DONE, 22),
        (PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME, 51),
        (PullRequestMetricID.PR_WAIT_FIRST_REVIEW_COUNT, 51),
        (PullRequestMetricID.PR_WAIT_FIRST_REVIEW_COUNT_Q, 51),
        (PullRequestMetricID.PR_SIZE, 51),
        (PullRequestMetricID.PR_DEPLOYMENT_TIME, 0),  # because pdb is empty
    ],
)
async def test_calc_metrics_prs_smoke(client, metric, count, headers, app, client_cache):
    """Trivial test to prove that at least something is working."""
    req_body = {
        "for": [
            {
                "with": {
                    "author": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                },
                "repositories": [
                    "github.com/src-d/go-git",
                ],
                **({"environments": ["production"]}
                   if metric == PullRequestMetricID.PR_DEPLOYMENT_TIME
                   else {}),
            },
        ],
        "metrics": [metric],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularities": ["week"],
        "exclude_inactive": False,
        "account": 1,
    }

    for _ in range(2):
        response = await client.request(
            method="POST", path="/v1/metrics/pull_requests", headers=headers, json=req_body,
        )
        body = (await response.read()).decode("utf-8")
        assert response.status == 200, "Response body is : " + body
        cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
        assert len(cm.calculated) == 1
        assert len(cm.calculated[0].values) > 0
        s = 0
        is_int = "TIME" not in metric
        for val in cm.calculated[0].values:
            assert len(val.values) == 1
            m = val.values[0]
            if is_int:
                s += m != 0 and m is not None
            else:
                s += m is not None
        assert s == count


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_all_time(client, headers):
    """https://athenianco.atlassian.net/browse/ENG-116"""
    devs = ["github.com/vmarkovtsev", "github.com/mcuadros"]
    for_block = {
        "with": {
            "author": devs,
            "merger": devs,
            "releaser": devs,
            "commenter": devs,
            "reviewer": devs,
            "commit_author": devs,
            "commit_committer": devs,
        },
        "repositories": [
            "github.com/src-d/go-git",
        ],
    }
    body = {
        "for": [for_block, for_block],
        "metrics": [PullRequestMetricID.PR_WIP_TIME,
                    PullRequestMetricID.PR_REVIEW_TIME,
                    PullRequestMetricID.PR_MERGING_TIME,
                    PullRequestMetricID.PR_RELEASE_TIME,
                    PullRequestMetricID.PR_LEAD_TIME,
                    PullRequestMetricID.PR_CYCLE_TIME,
                    PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME],
        "date_from": "2015-10-13",
        "date_to": "2019-03-15",
        "timezone": 60,
        "granularities": ["day", "week", "month"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert cm.calculated[0].values[0].date == date(year=2015, month=10, day=13)
    assert cm.calculated[0].values[-1].date <= date(year=2019, month=3, day=15)
    for i in range(len(cm.calculated[0].values) - 1):
        assert cm.calculated[0].values[i].date < cm.calculated[0].values[i + 1].date
    cmins = cmaxs = cscores = 0
    gcounts = defaultdict(int)
    assert len(cm.calculated) == 6
    for calc in cm.calculated:
        assert calc.for_.with_ == PullRequestWith(**for_block["with"])
        assert calc.for_.repositories == ["github.com/src-d/go-git"]
        gcounts[calc.granularity] += 1
        nonzero = defaultdict(int)
        for val in calc.values:
            for m, t in zip(cm.metrics, val.values):
                if t is None:
                    continue
                assert pd.to_timedelta(t) >= timedelta(0), \
                    "Metric: %s\nValues: %s" % (m, val.values)
                nonzero[m] += pd.to_timedelta(t) > timedelta(0)
            if val.confidence_mins is not None:
                cmins += 1
                for t, v in zip(val.confidence_mins, val.values):
                    if t is None:
                        assert v is None
                        continue
                    assert pd.to_timedelta(t) >= timedelta(0), \
                        "Metric: %s\nConfidence mins: %s" % (m, val.confidence_mins)
            if val.confidence_maxs is not None:
                cmaxs += 1
                for t, v in zip(val.confidence_maxs, val.values):
                    if t is None:
                        assert v is None
                        continue
                    assert pd.to_timedelta(t) >= timedelta(0), \
                        "Metric: %s\nConfidence maxs: %s" % (m, val.confidence_maxs)
            if val.confidence_scores is not None:
                cscores += 1
                for s, v in zip(val.confidence_scores, val.values):
                    if s is None:
                        assert v is None
                        continue
                    assert 0 <= s <= 100, \
                        "Metric: %s\nConfidence scores: %s" % (m, val.confidence_scores)
        for k, v in nonzero.items():
            assert v > 0, k
    assert cmins > 0
    assert cmaxs > 0
    assert cscores > 0
    assert all((v == 2) for v in gcounts.values())


async def test_calc_metrics_prs_access_denied(client, headers):
    """https://athenianco.atlassian.net/browse/ENG-116"""
    body = {
        "for": [
            {
                "with": {"commit_committer": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                "repositories": [
                    "github.com/src-d/go-git",
                    "github.com/athenianco/athenian-api",
                ],
            },
        ],
        "metrics": list(PullRequestMetricID),
        "date_from": "2015-10-13",
        "date_to": "2019-03-15",
        "granularities": ["month"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize(("devs", "date_from"),
                         ([{"with": {}}, "2019-11-28"], [{}, "2018-09-28"]))
async def test_calc_metrics_prs_empty_devs_tight_date(client, devs, date_from, headers):
    """https://athenianco.atlassian.net/browse/ENG-126"""
    body = {
        "date_from": date_from,
        "date_to": "2020-01-16",
        "for": [{
            **devs,
            "repositories": [
                "github.com/src-d/go-git",
            ],
            "environments": ["production"],
        }],
        "granularities": ["month"],
        "exclude_inactive": False,
        "account": 1,
        "metrics": list(PullRequestMetricID),
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert len(cm.calculated[0].values) > 0


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize("account, date_to, quantiles, lines, in_, code",
                         [(3, "2020-02-22", [0, 1], None, "{1}", 404),
                          (2, "2020-02-22", [0, 1], None, "{1}", 422),
                          (10, "2020-02-22", [0, 1], None, "{1}", 404),
                          (1, "2015-10-13", [0, 1], None, "{1}", 200),
                          (1, "2010-01-11", [0, 1], None, "{1}", 400),
                          (1, "2020-01-32", [0, 1], None, "{1}", 400),
                          (1, "2020-01-01", [-1, 0.5], None, "{1}", 400),
                          (1, "2020-01-01", [0, -1], None, "{1}", 400),
                          (1, "2020-01-01", [10, 20], None, "{1}", 400),
                          (1, "2020-01-01", [0.5, 0.25], None, "{1}", 400),
                          (1, "2020-01-01", [0.5, 0.5], None, "{1}", 400),
                          (1, "2015-10-13", [0, 1], [], "{1}", 400),
                          (1, "2015-10-13", [0, 1], [1], "{1}", 400),
                          (1, "2015-10-13", [0, 1], [1, 1], "{1}", 400),
                          (1, "2015-10-13", [0, 1], [-1, 1], "{1}", 400),
                          (1, "2015-10-13", [0, 1], [1, 0], "{1}", 400),
                          (1, "2015-10-13", [0, 1], None, "github.com/athenianco/api", 403),
                          ("1", "2020-02-22", [0, 1], None, "{1}", 400),
                          (1, "0015-10-13", [0, 1], None, "{1}", 400),
                          ])
async def test_calc_metrics_prs_nasty_input(
        client, headers, account, date_to, quantiles, lines, in_, code, mdb):
    """What if we specify a date that does not exist?"""
    body = {
        "for": [
            {
                "with": {"merger": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                "repositories": [in_],
                **({"lines": lines} if lines is not None else {}),
            },
            {
                "with": {"releaser": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
        ],
        "metrics": [PullRequestMetricID.PR_LEAD_TIME],
        "date_from": "2015-10-13" if date_to != "0015-10-13" else "0015-10-13",
        "date_to": date_to if date_to != "0015-10-13" else "2015-10-13",
        "granularities": ["week"],
        "quantiles": quantiles,
        "exclude_inactive": False,
        "account": account,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == code, "Response body is : " + body


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_reposet(client, headers):
    """Substitute {id} with the real repos."""
    body = {
        "for": [
            {
                "with": {"author": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                "repositories": ["{1}"],
            },
        ],
        "metrics": [PullRequestMetricID.PR_LEAD_TIME],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularities": ["all"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert len(cm.calculated[0].values) > 0
    assert cm.calculated[0].for_.repositories == ["{1}"]


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize("metric, count", [
    (PullRequestMetricID.PR_WIP_COUNT, 596),
    (PullRequestMetricID.PR_REVIEW_COUNT, 433),
    (PullRequestMetricID.PR_MERGING_COUNT, 589),
    (PullRequestMetricID.PR_RELEASE_COUNT, 408),
    (PullRequestMetricID.PR_LEAD_COUNT, 408),
    (PullRequestMetricID.PR_CYCLE_COUNT, 932),
    (PullRequestMetricID.PR_OPENED, 596),
    (PullRequestMetricID.PR_REVIEWED, 373),
    (PullRequestMetricID.PR_NOT_REVIEWED, 276),
    (PullRequestMetricID.PR_CLOSED, 589),
    (PullRequestMetricID.PR_MERGED, 538),
    (PullRequestMetricID.PR_REJECTED, 51),
    (PullRequestMetricID.PR_DONE, 468),
    (PullRequestMetricID.PR_WIP_PENDING_COUNT, 0),
    (PullRequestMetricID.PR_REVIEW_PENDING_COUNT, 86),
    (PullRequestMetricID.PR_MERGING_PENDING_COUNT, 21),
    (PullRequestMetricID.PR_RELEASE_PENDING_COUNT, 4395),
])
async def test_calc_metrics_prs_counts_sums(client, headers, metric, count):
    body = {
        "for": [
            {
                "with": {k: ["github.com/vmarkovtsev", "github.com/mcuadros"]
                         for k in PullRequestWith().attribute_types},
                "repositories": ["{1}"],
            },
        ],
        "metrics": [metric],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularities": ["month"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200, response.text()
    body = FriendlyJson.loads((await response.read()).decode("utf-8"))
    s = 0
    for item in body["calculated"][0]["values"]:
        assert "confidence_mins" not in item
        assert "confidence_maxs" not in item
        assert "confidence_scores" not in item
        val = item["values"][0]
        if val is not None:
            s += val
    assert s == count


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize("with_bots, values", [
    (False, [[2.461538553237915, None, None, None],
             [2.8328359127044678, 2.452173948287964, 7.439024448394775, 8.300812721252441],
             [3.003115177154541, 2.5336787700653076, 5.616822242736816, 6.504673004150391],
             [2.9247312545776367, 2.4838709831237793, 6.034883499145508, 6.813953399658203],
             [2.660493850708008, 2.5050504207611084, 7.74545431137085, 8.163636207580566]]),
    (True, [[1.4807692766189575, None, None, None],
            [1.9402985572814941, 2.2058823108673096, 7.699999809265137, 8.050000190734863],
            [2.121495246887207, 2.3231706619262695, 5.641975402832031, 6.111111164093018],
            [2.0465950965881348, 2.2727272510528564, 6.234375, 6.75],
            [1.814814805984497, 2.4146342277526855, 7.767441749572754, 7.860465049743652]]),
])
async def test_calc_metrics_prs_averages(client, headers, with_bots, values, sdb):
    if with_bots:
        await sdb.execute(insert(Team).values(Team(
            owner_id=1,
            name=Team.BOTS,
            members=[39789],
        ).create_defaults().explode()))
    body = {
        "for": [
            {
                "with": {},
                "repositories": ["{1}"],
            },
        ],
        "metrics": [PullRequestMetricID.PR_PARTICIPANTS_PER,
                    PullRequestMetricID.PR_REVIEWS_PER,
                    PullRequestMetricID.PR_REVIEW_COMMENTS_PER,
                    PullRequestMetricID.PR_COMMENTS_PER],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularities": ["year"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200, response.text()
    body = FriendlyJson.loads((await response.read()).decode("utf-8"))
    assert [v["values"] for v in body["calculated"][0]["values"]] == values


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_sizes(client, headers):
    body = {
        "for": [
            {
                "with": {},
                "repositories": ["{1}"],
            },
        ],
        "metrics": [PullRequestMetricID.PR_SIZE,
                    PullRequestMetricID.PR_MEDIAN_SIZE],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularities": ["all"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200, response.text()
    rbody = FriendlyJson.loads((await response.read()).decode("utf-8"))
    values = [v["values"] for v in rbody["calculated"][0]["values"]]
    assert values == [[296, 54]]
    for ts in rbody["calculated"][0]["values"]:
        for v, cmin, cmax in zip(ts["values"], ts["confidence_mins"], ts["confidence_maxs"]):
            assert cmin < v < cmax

    body["quantiles"] = [0, 0.9]
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200, response.text()
    rbody = FriendlyJson.loads((await response.read()).decode("utf-8"))
    values = [v["values"] for v in rbody["calculated"][0]["values"]]
    assert values == [[177, 54]]

    body["granularities"].append("month")
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200, response.text()
    rbody = FriendlyJson.loads((await response.read()).decode("utf-8"))
    values = [v["values"] for v in rbody["calculated"][0]["values"]]
    assert values == [[177, 54]]


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_index_error(client, headers):
    body = {
        "for": [
            {
                "with": {},
                "repositories": ["github.com/src-d/go-git"],
            },
        ],
        "metrics": [PullRequestMetricID.PR_WIP_TIME,
                    PullRequestMetricID.PR_REVIEW_TIME,
                    PullRequestMetricID.PR_MERGING_TIME,
                    PullRequestMetricID.PR_RELEASE_TIME,
                    PullRequestMetricID.PR_LEAD_TIME],
        "date_from": "2019-02-25",
        "date_to": "2019-02-28",
        "granularities": ["week"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_ratio_flow(client, headers):
    """https://athenianco.atlassian.net/browse/ENG-411"""
    body = {
        "date_from": "2016-01-01",
        "date_to": "2020-01-16",
        "for": [{
            "repositories": [
                "github.com/src-d/go-git",
            ],
        }],
        "granularities": ["month"],
        "exclude_inactive": False,
        "account": 1,
        "metrics": [PullRequestMetricID.PR_FLOW_RATIO, PullRequestMetricID.PR_OPENED,
                    PullRequestMetricID.PR_CLOSED],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert len(cm.calculated[0].values) > 0
    for v in cm.calculated[0].values:
        flow, opened, closed = v.values
        if opened is not None:
            assert flow is not None
        else:
            opened = 0
        if flow is None:
            assert closed is None
            continue
        assert flow == np.float32((opened + 1) / (closed + 1)), \
            "%.3f != %d / %d" % (flow, opened, closed)


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_exclude_inactive_full_span(client, headers):
    body = {
        "date_from": "2017-01-01",
        "date_to": "2017-01-11",
        "for": [{
            "repositories": [
                "github.com/src-d/go-git",
            ],
        }],
        "granularities": ["all"],
        "account": 1,
        "metrics": [PullRequestMetricID.PR_ALL_COUNT],
        "exclude_inactive": True,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(rbody))
    assert cm.calculated[0].values[0].values[0] == 6


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_exclude_inactive_split(client, headers):
    body = {
        "date_from": "2016-12-21",
        "date_to": "2017-01-11",
        "for": [{
            "repositories": [
                "github.com/src-d/go-git",
            ],
        }],
        "granularities": ["11 day"],
        "account": 1,
        "metrics": [PullRequestMetricID.PR_ALL_COUNT],
        "exclude_inactive": True,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(rbody))
    assert cm.calculated[0].values[0].values[0] == 1
    assert cm.calculated[0].values[1].date == date(2017, 1, 1)
    assert cm.calculated[0].values[1].values[0] == 6


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_filter_authors(client, headers):
    body = {
        "date_from": "2017-01-01",
        "date_to": "2017-01-11",
        "for": [{
            "repositories": [
                "github.com/src-d/go-git",
            ],
            "with": {
                "author": ["github.com/mcuadros"],
            },
        }],
        "granularities": ["all"],
        "account": 1,
        "metrics": [PullRequestMetricID.PR_ALL_COUNT],
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert cm.calculated[0].values[0].values[0] == 1


@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_filter_team(client, headers, sample_team):
    body = {
        "date_from": "2017-01-01",
        "date_to": "2017-01-11",
        "for": [{
            "repositories": [
                "github.com/src-d/go-git",
            ],
            "with": {
                "author": ["{%d}" % sample_team],
            },
        }],
        "granularities": ["all"],
        "account": 1,
        "metrics": [PullRequestMetricID.PR_ALL_COUNT],
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert cm.calculated[0].values[0].values[0] == 4


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_group_authors(client, headers):
    body = {
        "date_from": "2017-01-01",
        "date_to": "2017-04-11",
        "for": [{
            "repositories": [
                "github.com/src-d/go-git",
            ],
            "withgroups": [
                {
                    "author": ["github.com/mcuadros"],
                },
                {
                    "merger": ["github.com/mcuadros"],
                },
            ],
        }],
        "granularities": ["all"],
        "account": 1,
        "metrics": [PullRequestMetricID.PR_ALL_COUNT],
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert cm.calculated[0].values[0].values[0] == 13
    assert cm.calculated[0].for_.with_.author == ["github.com/mcuadros"]
    assert not cm.calculated[0].for_.with_.merger
    assert cm.calculated[1].values[0].values[0] == 49
    assert cm.calculated[1].for_.with_.merger == ["github.com/mcuadros"]
    assert not cm.calculated[1].for_.with_.author


@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_group_team(client, headers, sample_team):
    team_str = "{%d}" % sample_team
    body = {
        "date_from": "2017-01-01",
        "date_to": "2017-04-11",
        "for": [{
            "repositories": [
                "github.com/src-d/go-git",
            ],
            "withgroups": [
                {
                    "author": [team_str],
                },
                {
                    "merger": [team_str],
                },
            ],
        }],
        "granularities": ["all"],
        "account": 1,
        "metrics": [PullRequestMetricID.PR_ALL_COUNT],
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert cm.calculated[0].values[0].values[0] == 21
    assert cm.calculated[0].for_.with_.author == [team_str]
    assert not cm.calculated[0].for_.with_.merger
    assert cm.calculated[1].values[0].values[0] == 61
    assert cm.calculated[1].for_.with_.merger == [team_str]
    assert not cm.calculated[1].for_.with_.author


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_labels_include(client, headers):
    body = {
        "date_from": "2018-09-01",
        "date_to": "2018-11-18",
        "for": [{
            "repositories": [
                "github.com/src-d/go-git",
            ],
            "labels_include": [
                "bug", "enhancement",
            ],
        }],
        "granularities": ["all"],
        "account": 1,
        "metrics": [PullRequestMetricID.PR_ALL_COUNT],
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert cm.calculated[0].values[0].values[0] == 6


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_quantiles(client, headers):
    body = {
        "date_from": "2018-06-01",
        "date_to": "2018-11-18",
        "for": [{
            "repositories": [
                "github.com/src-d/go-git",
            ],
            "labels_include": [
                "bug", "enhancement",
            ],
        }],
        "granularities": ["all"],
        "account": 1,
        "metrics": [PullRequestMetricID.PR_WIP_TIME],
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(rbody))
    wip1 = cm.calculated[0].values[0].values[0]

    body["quantiles"] = [0, 0.5]
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(rbody))
    wip2 = cm.calculated[0].values[0].values[0]
    assert int(wip1[:-1]) < int(wip2[:-1])  # yes, not >, here is why:
    # array([[['NaT', 'NaT', 'NaT'],
    #         ['NaT', 'NaT', 'NaT'],
    #         ['NaT', 'NaT', 'NaT'],
    #         [496338, 'NaT', 'NaT'],
    #         ['NaT', 'NaT', 'NaT'],
    #         [250, 'NaT', 'NaT'],
    #         ['NaT', 'NaT', 'NaT'],
    #         [1191, 0, 293],
    #         [3955, 'NaT', 'NaT'],
    #         ['NaT', 'NaT', 'NaT'],
    #         ['NaT', 'NaT', 'NaT'],
    #         ['NaT', 'NaT', 'NaT'],
    #         ['NaT', 'NaT', 'NaT']]], dtype='timedelta64[s]')
    # We discard 1191 and the overall average becomes bigger.

    body["granularities"] = ["week", "month"]
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_jira(client, headers):
    """Metrics over PRs filtered by JIRA properties."""
    body = {
        "for": [{
            "repositories": ["{1}"],
            "jira": {
                "epics": ["DEV-149", "DEV-776", "DEV-737", "DEV-667", "DEV-140"],
                "labels_include": ["performance", "enhancement"],
                "labels_exclude": ["security"],
                "issue_types": ["Task"],
            },
        }],
        "metrics": [PullRequestMetricID.PR_LEAD_TIME],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularities": ["all"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert len(cm.calculated[0].values) > 0
    assert cm.calculated[0].values[0].values[0] == "478544s"


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_jira_disabled_projects(client, headers, disabled_dev):
    body = {
        "for": [{
            "repositories": ["{1}"],
            "jira": {
                "epics": ["DEV-149", "DEV-776", "DEV-737", "DEV-667", "DEV-140"],
            },
        }],
        "metrics": [PullRequestMetricID.PR_LEAD_TIME],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularities": ["all"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert len(cm.calculated[0].values) > 0
    assert cm.calculated[0].values[0].values[0] is None


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_jira_custom_projects(client, headers):
    body = {
        "for": [{
            "repositories": ["{1}"],
            "jira": {
                "epics": ["DEV-149", "DEV-776", "DEV-737", "DEV-667", "DEV-140"],
                "projects": ["ENG"],
            },
        }],
        "metrics": [PullRequestMetricID.PR_LEAD_TIME],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularities": ["all"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert len(cm.calculated[0].values) > 0
    assert cm.calculated[0].values[0].values[0] is None


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_jira_only_custom_projects(client, headers):
    body = {
        "for": [{
            "repositories": ["{1}"],
            "jira": {
                "projects": ["DEV"],
            },
        }],
        "metrics": [PullRequestMetricID.PR_MERGED],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularities": ["all"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert len(cm.calculated[0].values) > 0
    assert cm.calculated[0].values[0].values[0] == 45  # > 400 without JIRA projects


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_groups_smoke(client, headers):
    """Two repository groups."""
    body = {
        "for": [
            {
                "with": {"author": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                "repositories": ["{1}", "github.com/src-d/go-git"],
                "repogroups": [[0], [0]],
            },
        ],
        "metrics": [PullRequestMetricID.PR_LEAD_TIME],
        "date_from": "2017-10-13",
        "date_to": "2018-01-23",
        "granularities": ["all"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert len(cm.calculated) == 2
    assert cm.calculated[0] == cm.calculated[1]
    assert cm.calculated[0].values[0].values[0] == "3667053s"
    assert cm.calculated[0].for_.repositories == ["{1}"]


@pytest.mark.parametrize("repogroups", [[[0, 0]], [[0, -1]], [[0, 1]]])
async def test_calc_metrics_prs_groups_nasty(client, headers, repogroups):
    """Two repository groups."""
    body = {
        "for": [
            {
                "with": {"author": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                "repositories": ["{1}"],
                "repogroups": repogroups,
            },
        ],
        "metrics": [PullRequestMetricID.PR_LEAD_TIME],
        "date_from": "2017-10-13",
        "date_to": "2018-01-23",
        "granularities": ["all"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 400, "Response body is : " + body


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_lines_smoke(client, headers):
    """Two repository groups."""
    body = {
        "for": [
            {
                "with": {"author": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                "repositories": ["{1}", "github.com/src-d/go-git"],
                "lines": [50, 200, 100000, 100500],
            },
        ],
        "metrics": [PullRequestMetricID.PR_OPENED],
        "date_from": "2017-10-13",
        "date_to": "2018-03-23",
        "granularities": ["all"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(rbody))
    assert len(cm.calculated) == 3
    assert cm.calculated[0].values[0].values[0] == 3
    assert cm.calculated[0].for_.lines == [50, 200]
    assert cm.calculated[1].values[0].values[0] == 3
    assert cm.calculated[1].for_.lines == [200, 100000]
    assert cm.calculated[2].values[0].values[0] == 0
    assert cm.calculated[2].for_.lines == [100000, 100500]

    body["for"][0]["lines"] = [50, 100500]
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(rbody))
    assert len(cm.calculated) == 1
    assert cm.calculated[0].values[0].values[0] == 6


@pytest.mark.parametrize("metric", [
    PullRequestMetricID.PR_DEPLOYMENT_TIME,
    PullRequestMetricID.PR_DEPLOYMENT_COUNT,
    PullRequestMetricID.PR_DEPLOYMENT_COUNT_Q,
    PullRequestMetricID.PR_CYCLE_DEPLOYMENT_TIME,
    PullRequestMetricID.PR_CYCLE_DEPLOYMENT_COUNT,
    PullRequestMetricID.PR_CYCLE_DEPLOYMENT_COUNT_Q,
    PullRequestMetricID.PR_LEAD_DEPLOYMENT_TIME,
    PullRequestMetricID.PR_LEAD_DEPLOYMENT_COUNT,
    PullRequestMetricID.PR_LEAD_DEPLOYMENT_COUNT_Q,
])
async def test_calc_metrics_prs_deployments_no_env(client, headers, metric):
    body = {
        "for": [
            {
                "with": {},
                "repositories": ["{1}"],
                **({"environments": []} if "time" in metric else {}),
            },
        ],
        "metrics": [metric],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularities": ["year"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    assert response.status == 400, response.text()


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_deployments_smoke(client, headers, precomputed_deployments):
    body = {
        "for": [
            {
                "repositories": ["{1}"],
                "environments": ["staging", "production"],
            },
        ],
        "metrics": [PullRequestMetricID.PR_DEPLOYMENT_TIME,
                    PullRequestMetricID.PR_LEAD_DEPLOYMENT_TIME,
                    PullRequestMetricID.PR_CYCLE_DEPLOYMENT_TIME,
                    PullRequestMetricID.PR_DEPLOYMENT_COUNT],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularities": ["all"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200, response.text()
    body = FriendlyJson.loads((await response.read()).decode("utf-8"))
    values = [v["values"] for v in body["calculated"][0]["values"]]
    assert values == [[[None, "57352991s"], [None, "60558816s"], [None, "61066621s"], [0, 418]]]


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize("second_repo, counts", [
    ("/beta", [266, 52, 204, 197]),
    ("", [463, 81, 372, 347]),
])
async def test_calc_metrics_prs_logical_smoke(
        client, headers, logical_settings_db, release_match_setting_tag_logical_db,
        second_repo, counts):
    body = {
        "for": [
            {
                "repositories": ["github.com/src-d/go-git/alpha",
                                 "github.com/src-d/go-git" + second_repo],
            },
        ],
        "metrics": [
            PullRequestMetricID.PR_MERGED,
            PullRequestMetricID.PR_REJECTED,
            PullRequestMetricID.PR_REVIEW_COUNT,
            PullRequestMetricID.PR_RELEASE_COUNT,
        ],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularities": ["all"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200, response.text()
    body = FriendlyJson.loads((await response.read()).decode("utf-8"))
    values = [v["values"] for v in body["calculated"][0]["values"]]
    assert values == [counts]


@pytest.mark.app_validate_responses(False)
async def test_calc_metrics_prs_logical_dupes(client, headers, logical_settings_db, sdb):
    await sdb.execute(insert(ReleaseSetting).values(
        ReleaseSetting(repository="github.com/src-d/go-git/alpha",
                       account_id=1,
                       branches="master",
                       tags=".*",
                       events=".*",
                       match=ReleaseMatch.tag).create_defaults().explode(with_primary_keys=True)))
    await sdb.execute(insert(ReleaseSetting).values(
        ReleaseSetting(repository="github.com/src-d/go-git/beta",
                       account_id=1,
                       branches="master",
                       tags=".*",
                       events=".*",
                       match=ReleaseMatch.tag).create_defaults().explode(with_primary_keys=True)))
    body = {
        "for": [
            {
                "repositories": ["github.com/src-d/go-git/alpha",
                                 "github.com/src-d/go-git/beta"],
            },
        ],
        "metrics": [
            PullRequestMetricID.PR_MERGED,
            PullRequestMetricID.PR_REJECTED,
            PullRequestMetricID.PR_REVIEW_COUNT,
            PullRequestMetricID.PR_RELEASE_COUNT,
        ],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularities": ["all"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200, response.text()
    body = FriendlyJson.loads((await response.read()).decode("utf-8"))
    values = [v["values"] for v in body["calculated"][0]["values"]]
    assert values == [[250, 49, 194, 186]]


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@with_defer
async def test_calc_metrics_prs_release_ignored(
        client, headers, mdb, pdb, rdb, release_match_setting_tag, pr_miner, prefixer,
        branches, default_branches):
    body = {
        "for": [{"repositories": ["{1}"]}],
        "metrics": [PullRequestMetricID.PR_RELEASE_TIME,
                    PullRequestMetricID.PR_RELEASE_COUNT,
                    PullRequestMetricID.PR_RELEASE_PENDING_COUNT,
                    PullRequestMetricID.PR_REJECTED,
                    PullRequestMetricID.PR_DONE],
        "date_from": "2017-06-01",
        "date_to": "2018-01-01",
        "granularities": ["all"],
        "exclude_inactive": True,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200, response.text()
    result = FriendlyJson.loads((await response.read()).decode("utf-8"))
    assert result["calculated"][0]["values"][0]["values"] == ["763080s", 79, 61, 21, 102]
    time_from = datetime(year=2017, month=6, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=12, day=31, tzinfo=timezone.utc)
    releases, _, _, _ = await mine_releases(
        ["src-d/go-git"], {}, None, default_branches, time_from, time_to, LabelFilter.empty(),
        JIRAFilter.empty(), release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None, with_deployments=False)
    await wait_deferred()
    ignored = await override_first_releases(
        releases, {}, release_match_setting_tag, 1, pdb, threshold_factor=0)
    assert ignored == 1
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    assert response.status == 200, response.text()
    result = FriendlyJson.loads((await response.read()).decode("utf-8"))
    assert result["calculated"][0]["values"][0]["values"] == ["779385s", 65, 61, 21, 102]


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_bypassing_prs_smoke(client, headers):
    body = {
        "account": 1,
        "date_from": "2015-01-12",
        "date_to": "2020-02-22",
        "timezone": 60,
        "in": ["{1}"],
        "granularity": "month",
    }
    response = await client.request(
        method="POST", path="/v1/metrics/code_bypassing_prs", headers=headers, json=body,
    )
    assert response.status == 200
    body = FriendlyJson.loads((await response.read()).decode("utf-8"))
    ms = [CodeBypassingPRsMeasurement.from_dict(x) for x in body]
    assert len(ms) == 62
    total_commits = total_lines = 0
    for s in ms:
        assert date(year=2015, month=1, day=12) <= s.date <= date(year=2020, month=2, day=22)
        assert s.total_commits >= 0
        assert s.total_lines >= 0
        assert 0 <= s.bypassed_commits <= s.total_commits
        assert 0 <= s.bypassed_lines <= s.total_lines
        total_commits += s.bypassed_commits
        total_lines += s.total_lines
    assert total_commits == 492
    assert total_lines == 261325
    for i in range(len(ms) - 1):
        assert ms[i].date < ms[i + 1].date


async def test_code_bypassing_prs_only_default_branch(client, headers):
    body = {
        "account": 1,
        "date_from": "2015-01-12",
        "date_to": "2020-02-22",
        "timezone": 60,
        "in": ["{1}"],
        "granularity": "month",
        "only_default_branch": True,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/code_bypassing_prs", headers=headers, json=body,
    )
    assert response.status == 200
    body = FriendlyJson.loads((await response.read()).decode("utf-8"))
    ms = [CodeBypassingPRsMeasurement.from_dict(x) for x in body]
    total_commits = total_lines = 0
    for s in ms:
        total_commits += s.bypassed_commits
        total_lines += s.total_lines
    assert total_commits == 417
    assert total_lines == 175297


@pytest.mark.parametrize("account, date_to, in_, code",
                         [(3, "2020-02-22", "{1}", 404),
                          (2, "2020-02-22", "github.com/src-d/go-git", 422),
                          (10, "2020-02-22", "{1}", 404),
                          (1, "2019-01-12", "{1}", 200),
                          (1, "2019-01-11", "{1}", 400),
                          (1, "2019-01-32", "{1}", 400),
                          (1, "2019-01-12", "github.com/athenianco/athenian-api", 403),
                          ])
async def test_code_bypassing_prs_nasty_input(client, headers, account, date_to, in_, code):
    body = {
        "account": account,
        "date_from": "2019-01-12",
        "date_to": date_to,
        "in": [in_],
        "granularity": "month",
    }
    response = await client.request(
        method="POST", path="/v1/metrics/code_bypassing_prs", headers=headers, json=body,
    )
    assert response.status == code


developer_metric_mcuadros_stats = {
    "dev-commits-pushed": 207,
    "dev-lines-changed": 34494,
    "dev-prs-created": 14,
    "dev-prs-reviewed": 35,
    "dev-prs-merged": 175,
    "dev-releases": 21,
    "dev-reviews": 68,
    "dev-review-approvals": 14,
    "dev-review-rejections": 13,
    "dev-review-neutrals": 41,
    "dev-pr-comments": 166,
    "dev-regular-pr-comments": 92,
    "dev-review-pr-comments": 74,
    "dev-active": 0,
    "dev-active0": 1,
    "dev-worked": 1,
}

developer_metric_be_stats = {
    "dev-commits-pushed": [[0], [0], [0], [0]],
    "dev-lines-changed": [[0], [0], [0], [0]],
    "dev-prs-created": [[0], [0], [1], [0]],
    "dev-prs-reviewed": [[2], [4], [3], [4]],
    "dev-prs-merged": [[6], [0], [0], [0]],
    "dev-releases": [[0], [0], [0], [0]],
    "dev-reviews": [[8], [6], [7], [4]],
    "dev-review-approvals": [[1], [3], [2], [3]],
    "dev-review-rejections": [[1], [1], [0], [0]],
    "dev-review-neutrals": [[6], [2], [5], [1]],
    "dev-pr-comments": [[10], [6], [5], [2]],
    "dev-regular-pr-comments": [[3], [1], [0], [1]],
    "dev-review-pr-comments": [[7], [5], [5], [1]],
    "dev-active": [[0], [0], [0], [0]],
    "dev-active0": [[0], [0], [0], [0]],
    "dev-worked": [[1], [1], [1], [1]],
}


@pytest.mark.parametrize("metric, value", sorted((m, developer_metric_mcuadros_stats[m])
                                                 for m in DeveloperMetricID))
async def test_developer_metrics_single(client, headers, metric, value):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "granularities": ["all"],
        "for": [
            {"repositories": ["{1}"], "developers": ["github.com/mcuadros"]},
        ],
        "metrics": [metric],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert result.metrics == [metric]
    assert result.date_from == date(year=2018, month=1, day=12)
    assert result.date_to == date(year=2020, month=3, day=1)
    assert len(result.calculated) == 1
    assert result.calculated[0].for_.repositories == ["{1}"]
    assert result.calculated[0].for_.developers == ["github.com/mcuadros"]
    assert result.calculated[0].values[0][0].values == [value]


developer_metric_mcuadros_jira_stats = {
    "dev-prs-created": 3,
    "dev-prs-reviewed": 11,
    "dev-prs-merged": 42,
    "dev-reviews": 23,
    "dev-review-approvals": 7,
    "dev-review-rejections": 3,
    "dev-review-neutrals": 13,
    "dev-pr-comments": 43,
    "dev-regular-pr-comments": 24,
    "dev-review-pr-comments": 19,
    "dev-active": 0,
    "dev-active0": 0,
    "dev-worked": 1,
    "dev-releases": 0,
    "dev-commits-pushed": 0,  # this will be non-zero when the metric becomes PR-aware
    "dev-lines-changed": 0,  # this will be non-zero when the metric becomes PR-aware
}


@pytest.mark.parametrize("metric, value", list(developer_metric_mcuadros_jira_stats.items()))
async def test_developer_metrics_jira_single(client, headers, metric, value):
    body = {
        "account": 1,
        "date_from": "2016-01-12",
        "date_to": "2020-03-01",
        "granularities": ["all"],
        "for": [
            {"repositories": ["{1}"], "developers": ["github.com/mcuadros"],
             "jira": {
                 "labels_include": [
                     "API", "Webapp", "accounts", "bug", "code-quality", "discarded",
                     "discussion", "feature", "functionality", "internal-story", "needs-specs",
                     "onboarding", "performance", "user-story", "webapp",
                 ],
            }},
        ],
        "metrics": [metric],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert result.metrics == [metric]
    assert result.date_from == date(year=2016, month=1, day=12)
    assert result.date_to == date(year=2020, month=3, day=1)
    assert len(result.calculated) == 1
    assert result.calculated[0].for_.repositories == ["{1}"]
    assert result.calculated[0].for_.developers == ["github.com/mcuadros"]
    assert result.calculated[0].values[0][0].values == [value]


developer_metric_mcuadros_jira_labels_stats = {
    "dev-prs-created": 0,
    "dev-prs-reviewed": 0,
    "dev-prs-merged": 1,
    "dev-reviews": 0,
    "dev-review-approvals": 0,
    "dev-review-rejections": 0,
    "dev-review-neutrals": 0,
    "dev-pr-comments": 0,
    "dev-regular-pr-comments": 0,
    "dev-review-pr-comments": 0,
    "dev-active": 0,
    "dev-active0": 0,
    "dev-worked": 1,
}


@pytest.mark.parametrize("metric, value",
                         list(developer_metric_mcuadros_jira_labels_stats.items()))
async def test_developer_metrics_jira_labels_single(client, headers, metric, value):
    body = {
        "account": 1,
        "date_from": "2016-01-12",
        "date_to": "2020-03-01",
        "granularities": ["all"],
        "for": [
            {"repositories": ["{1}"], "developers": ["github.com/mcuadros"],
             "labels_include": ["enhancement", "bug", "plumbing", "performance", "ssh",
                                "documentation", "windows"],
             "jira": {
                 "labels_include": [
                     "API", "Webapp", "accounts", "bug", "code-quality", "discarded",
                     "discussion", "feature", "functionality", "internal-story", "needs-specs",
                     "onboarding", "performance", "user-story", "webapp",
                 ],
            }},
        ],
        "metrics": [metric],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert result.metrics == [metric]
    assert result.date_from == date(year=2016, month=1, day=12)
    assert result.date_to == date(year=2020, month=3, day=1)
    assert len(result.calculated) == 1
    assert result.calculated[0].for_.repositories == ["{1}"]
    assert result.calculated[0].for_.developers == ["github.com/mcuadros"]
    assert result.calculated[0].values[0][0].values == [value]


@pytest.mark.parametrize("dev", ["mcuadros", "vmarkovtsev", "xxx", "EmrysMyrddin"])
async def test_developer_metrics_all(client, headers, dev):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "timezone": 60,
        "granularities": ["all"],
        "for": [
            {"repositories": ["{1}"], "developers": ["github.com/" + dev]},
        ],
        "metrics": sorted(DeveloperMetricID),
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200, (await response.read()).decode("utf-8")
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert set(result.metrics) == set(DeveloperMetricID)
    assert result.date_from == date(year=2018, month=1, day=12)
    assert result.date_to == date(year=2020, month=3, day=1)
    assert len(result.calculated) == 1
    assert result.calculated[0].for_.repositories == ["{1}"]
    assert result.calculated[0].for_.developers == ["github.com/" + dev]
    assert len(result.calculated[0].values) == 1
    assert len(result.calculated[0].values[0][0].values) == len(DeveloperMetricID)
    if dev == "mcuadros":
        for v, m in zip(result.calculated[0].values[0][0].values, sorted(DeveloperMetricID)):
            assert v == developer_metric_mcuadros_stats[m], m
    elif dev == "xxx":
        assert all(v == 0 for v in result.calculated[0].values[0][0].values), \
            "%s\n%s" % (str(result.calculated[0].values[0]), sorted(DeveloperMetricID))
    else:
        assert all(isinstance(v, int) for v in result.calculated[0].values[0][0].values), \
            "%s\n%s" % (str(result.calculated[0].values[0]), sorted(DeveloperMetricID))


async def test_developer_metrics_repogroups(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "timezone": 60,
        "granularities": ["all"],
        "for": [
            {"repositories": ["github.com/src-d/go-git", "github.com/src-d/gitbase"],
             "repogroups": [[0], [1]],
             "developers": ["github.com/mcuadros"]},
        ],
        "metrics": sorted(DeveloperMetricID),
    }
    developer.ACTIVITY_DAYS_THRESHOLD_DENSITY = 0.1
    try:
        response = await client.request(
            method="POST", path="/v1/metrics/developers", headers=headers, json=body,
        )
    finally:
        developer.ACTIVITY_DAYS_THRESHOLD_DENSITY = 0.2
    assert response.status == 200, (await response.read()).decode("utf-8")
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert set(result.metrics) == set(DeveloperMetricID)
    assert len(result.calculated) == 2
    assert all((v > 0 or m == DeveloperMetricID.ACTIVE)
               for m, v in zip(sorted(sorted(DeveloperMetricID)),
                               result.calculated[0].values[0][0].values))
    assert all(v == 0 for v in result.calculated[1].values[0][0].values)


@pytest.mark.parametrize("metric, value", sorted((m, developer_metric_be_stats[m])
                                                 for m in DeveloperMetricID))
async def test_developer_metrics_labels_include(client, headers, metric, value):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "granularities": ["all"],
        "for": [{
            "repositories": ["{1}"],
            "developers": ["github.com/mcuadros", "github.com/smola",
                           "github.com/jfontan", "github.com/ajnavarro"],
            "labels_include": ["bug", "enhancement"],
        }],
        "metrics": [metric],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert result.calculated[0].for_.labels_include == ["bug", "enhancement"]
    assert [m[0].values for m in result.calculated[0].values] == value


async def test_developer_metrics_aggregate(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-07-12",
        "date_to": "2018-09-15",
        "granularities": ["all"],
        "for": [{
            "repositories": ["{1}"],
            "developers": ["github.com/mcuadros", "github.com/erizocosmico",
                           "github.com/jfontan", "github.com/vancluever"],
            "aggregate_devgroups": [[0, 1, 2, 3]],
        }],
        "metrics": [DeveloperMetricID.ACTIVE, DeveloperMetricID.PRS_CREATED],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert [m[0].values for m in result.calculated[0].values] == [[1, 17]]
    body["for"][0]["aggregate_devgroups"][0].append(-1)
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 400
    body["for"][0]["aggregate_devgroups"][0][-1] = 4
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 400


async def test_developer_metrics_granularities(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-07-12",
        "date_to": "2018-09-15",
        "granularities": ["all", "2 week"],
        "for": [{
            "repositories": ["{1}"],
            "developers": ["github.com/mcuadros", "github.com/erizocosmico",
                           "github.com/jfontan"],
        }],
        "metrics": [DeveloperMetricID.ACTIVE, DeveloperMetricID.PRS_CREATED],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert len(result.calculated) == 2
    assert result.calculated[0].values == [
        [CalculatedLinearMetricValues(date=date(2018, 7, 12), values=[1, 0])],
        [CalculatedLinearMetricValues(date=date(2018, 7, 12), values=[0, 4])],
        [CalculatedLinearMetricValues(date=date(2018, 7, 12), values=[1, 11])],
    ]
    assert result.calculated[0].granularity == "all"
    assert result.calculated[1].values == [
        [
            CalculatedLinearMetricValues(date=date(2018, 7, 12), values=[0, 0]),
            CalculatedLinearMetricValues(date=date(2018, 7, 26), values=[0, 0]),
            CalculatedLinearMetricValues(date=date(2018, 8, 9), values=[1, 0]),
            CalculatedLinearMetricValues(date=date(2018, 8, 23), values=[1, 0]),
            CalculatedLinearMetricValues(date=date(2018, 9, 6), values=[1, 0]),
        ], [
            CalculatedLinearMetricValues(date=date(2018, 7, 12), values=[0, 2]),
            CalculatedLinearMetricValues(date=date(2018, 7, 26), values=[1, 2]),
            CalculatedLinearMetricValues(date=date(2018, 8, 9), values=[0, 0]),
            CalculatedLinearMetricValues(date=date(2018, 8, 23), values=[0, 0]),
            CalculatedLinearMetricValues(date=date(2018, 9, 6), values=[0, 0]),
        ], [
            CalculatedLinearMetricValues(date=date(2018, 7, 12), values=[1, 2]),
            CalculatedLinearMetricValues(date=date(2018, 7, 26), values=[0, 2]),
            CalculatedLinearMetricValues(date=date(2018, 8, 9), values=[1, 4]),
            CalculatedLinearMetricValues(date=date(2018, 8, 23), values=[1, 3]),
            CalculatedLinearMetricValues(date=date(2018, 9, 6), values=[0, 0]),
        ],
    ]
    assert result.calculated[1].granularity == "2 week"


async def test_developer_metrics_labels_exclude(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "granularities": ["all"],
        "for": [{
            "repositories": ["{1}"],
            "developers": ["github.com/mcuadros", "github.com/smola",
                           "github.com/jfontan", "github.com/ajnavarro"],
            "labels_exclude": ["bug", "enhancement"],
        }],
        "metrics": [DeveloperMetricID.PRS_CREATED],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert not result.calculated[0].for_.labels_include
    assert result.calculated[0].for_.labels_exclude == ["bug", "enhancement"]
    assert [m[0].values for m in result.calculated[0].values] == [[14], [8], [26], [7]]


async def test_developer_metrics_labels_include_exclude(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "granularities": ["all"],
        "for": [{
            "repositories": ["{1}"],
            "developers": ["github.com/mcuadros", "github.com/smola",
                           "github.com/jfontan", "github.com/ajnavarro"],
            "labels_include": ["bug"],
            "labels_exclude": ["enhancement"],
        }],
        "metrics": [DeveloperMetricID.PRS_CREATED],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert result.calculated[0].for_.labels_include == ["bug"]
    assert result.calculated[0].for_.labels_exclude == ["enhancement"]
    assert [m[0].values for m in result.calculated[0].values] == [[0], [0], [1], [0]]


async def test_developer_metrics_labels_contradiction(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "granularities": ["all"],
        "for": [{
            "repositories": ["{1}"],
            "developers": ["github.com/mcuadros", "github.com/smola",
                           "github.com/jfontan", "github.com/ajnavarro"],
            "labels_include": ["bug"],
            "labels_exclude": ["bug"],
        }],
        "metrics": [DeveloperMetricID.PRS_CREATED],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert result.calculated[0].for_.labels_include == ["bug"]
    assert result.calculated[0].for_.labels_exclude == ["bug"]
    assert [m[0].values for m in result.calculated[0].values] == [[0], [0], [0], [0]]


@pytest.mark.parametrize("account, date_to, in_, code",
                         [(3, "2020-02-22", "{1}", 404),
                          (2, "2020-02-22", "github.com/src-d/go-git", 422),
                          (10, "2020-02-22", "{1}", 404),
                          (1, "2018-01-12", "{1}", 200),
                          (1, "2018-01-11", "{1}", 400),
                          (1, "2019-01-32", "{1}", 400),
                          (1, "2018-01-12", "github.com/athenianco/athenian-api", 403),
                          ])
async def test_developer_metrics_nasty_input(client, headers, account, date_to, in_, code):
    body = {
        "account": account,
        "date_from": "2018-01-12",
        "date_to": date_to,
        "granularities": ["all"],
        "for": [
            {"repositories": [in_], "developers": ["github.com/mcuadros"]},
        ],
        "metrics": sorted(DeveloperMetricID),
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == code


@pytest.mark.parametrize("repos, devs", [
    ([], ["github.com/mcuadros"]),
    (["github.com/src-d/go-git"], []),
])
async def test_developer_metrics_empty(client, headers, repos, devs):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-01-01",
        "granularities": ["all"],
        "for": [
            {"repositories": repos, "developers": devs},
        ],
        "metrics": sorted(DeveloperMetricID),
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 400, rbody


async def test_developer_metrics_order(client, headers):
    """https://athenianco.atlassian.net/browse/DEV-247"""
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "granularities": ["all"],
        "for": [
            {"repositories": ["{1}"], "developers": [
                "github.com/mcuadros", "github.com/smola"]},
        ],
        "metrics": [DeveloperMetricID.PRS_CREATED],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert result.calculated[0].for_.developers == body["for"][0]["developers"]
    assert [m[0].values for m in result.calculated[0].values] == [[14], [8]]
    body["for"][0]["developers"] = list(reversed(body["for"][0]["developers"]))
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert result.calculated[0].for_.developers == body["for"][0]["developers"]
    assert [m[0].values for m in result.calculated[0].values] == [[8], [14]]


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_release_metrics_smoke(client, headers, no_jira):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [["{1}"]],
        "jira": {
            "epics": [],
        },
        "metrics": list(ReleaseMetricID),
        "granularities": ["all", "3 month"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/releases", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
    assert len(models) == 2
    for model in models:
        assert model.for_ == ["{1}"]
        assert model.metrics == body["metrics"]
        assert model.matches == {
            "github.com/src-d/go-git": "tag",
            "github.com/src-d/gitbase": "branch",
        }
        assert model.granularity in body["granularities"]
        for mv in model.values:
            exist = mv.values[model.metrics.index(ReleaseMetricID.TAG_RELEASE_AGE)] is not None
            for metric, value in zip(model.metrics, mv.values):
                if "branch" in metric:
                    if "avg" not in metric and metric != ReleaseMetricID.BRANCH_RELEASE_AGE:
                        assert value == 0, metric
                    else:
                        assert value is None, metric
                elif exist:
                    assert value is not None, metric
        if model.granularity == "all":
            assert len(model.values) == 1
            assert any(v is not None for v in model.values[0].values)
        else:
            assert any(v is not None for values in model.values for v in values.values)
            assert len(model.values) == 9


@pytest.mark.parametrize("role, n", [("releaser", 21), ("pr_author", 10), ("commit_author", 21)])
async def test_release_metrics_participants_single(client, headers, role, n):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [["github.com/src-d/go-git"]],
        "with": [{role: ["github.com/mcuadros"]}],
        "metrics": [ReleaseMetricID.RELEASE_COUNT],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/releases", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
    assert len(models) == 1
    assert models[0].values[0].values[0] == n


async def test_release_metrics_participants_multiple(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [["github.com/src-d/go-git"]],
        "with": [{"releaser": ["github.com/smola"],
                  "pr_author": ["github.com/mcuadros"],
                  "commit_author": ["github.com/smola"]}],
        "metrics": [ReleaseMetricID.RELEASE_COUNT],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/releases", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
    assert len(models) == 1
    assert models[0].values[0].values[0] == 12


async def test_release_metrics_participants_team(client, headers, sample_team):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [["github.com/src-d/go-git"]],
        "with": [{"releaser": ["{%d}" % sample_team],
                  "pr_author": ["{%d}" % sample_team],
                  "commit_author": ["{%d}" % sample_team]}],
        "metrics": [ReleaseMetricID.RELEASE_COUNT],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/releases", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
    assert len(models) == 1
    assert models[0].values[0].values[0] == 21


async def test_release_metrics_participants_groups(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [["github.com/src-d/go-git"]],
        "with": [{"releaser": ["github.com/mcuadros"]},
                 {"pr_author": ["github.com/smola"]}],
        "metrics": [ReleaseMetricID.RELEASE_COUNT],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/releases", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
    assert len(models) == 2
    assert models[0].values[0].values[0] == 21
    assert models[1].values[0].values[0] == 4


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize("account, date_to, quantiles, extra_metrics, in_, code",
                         [(3, "2020-02-22", [0, 1], [], "{1}", 404),
                          (2, "2020-02-22", [0, 1], [], "github.com/src-d/go-git", 422),
                          (10, "2020-02-22", [0, 1], [], "{1}", 404),
                          (1, "2015-10-13", [0, 1], [], "{1}", 200),
                          (1, "2015-10-13", [0, 1], ["whatever"], "{1}", 400),
                          (1, "2010-01-11", [0, 1], [], "{1}", 400),
                          (1, "2020-01-32", [0, 1], [], "{1}", 400),
                          (1, "2020-01-01", [-1, 0.5], [], "{1}", 400),
                          (1, "2020-01-01", [0, -1], [], "{1}", 400),
                          (1, "2020-01-01", [10, 20], [], "{1}", 400),
                          (1, "2020-01-01", [0.5, 0.25], [], "{1}", 400),
                          (1, "2020-01-01", [0.5, 0.5], [], "{1}", 400),
                          (1, "2015-10-13", [0, 1], [], "github.com/athenianco/athenian-api", 403),
                          ])
async def test_release_metrics_nasty_input(
        client, headers, account, date_to, quantiles, extra_metrics, in_, code):
    body = {
        "for": [[in_], [in_]],
        "metrics": [ReleaseMetricID.TAG_RELEASE_AGE] + extra_metrics,
        "date_from": "2015-10-13",
        "date_to": date_to,
        "granularities": ["4 month"],
        "quantiles": quantiles,
        "account": account,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/releases", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == code, "Response body is : " + body


@pytest.mark.parametrize("devid", ["whatever", ""])
async def test_release_metrics_participants_invalid(client, headers, devid):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [["github.com/src-d/go-git"]],
        "with": [{"releaser": [devid]}],
        "metrics": [ReleaseMetricID.RELEASE_COUNT],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/releases", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 400, rbody


@pytest.mark.parametrize("q, value", ((0.95, "2687847s"), (1, "2687847s")))
async def test_release_metrics_quantiles(client, headers, q, value):
    body = {
        "account": 1,
        "date_from": "2015-01-12",
        "date_to": "2020-03-01",
        "for": [["{1}"], ["github.com/src-d/go-git"]],
        "metrics": [ReleaseMetricID.TAG_RELEASE_AGE],
        "granularities": ["all"],
        "quantiles": [0, q],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/releases", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
    assert len(models) == 2
    assert models[0].values == models[1].values
    model = models[0]
    assert model.values[0].values == [value]


async def test_release_metrics_jira(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-01",
        "date_to": "2020-03-01",
        "for": [["{1}"]],
        "metrics": [ReleaseMetricID.RELEASE_COUNT, ReleaseMetricID.RELEASE_PRS],
        "jira": {
            "labels_include": ["bug", "onboarding", "performance"],
        },
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/releases", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
    assert len(models) == 1
    assert models[0].values[0].values == [8, 43]
    del body["jira"]

    response = await client.request(
        method="POST", path="/v1/metrics/releases", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
    assert len(models) == 1
    assert models[0].values[0].values == [22, 234]


async def test_release_metrics_labels(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-01",
        "date_to": "2020-03-01",
        "for": [["{1}"]],
        "metrics": [ReleaseMetricID.RELEASE_COUNT, ReleaseMetricID.RELEASE_PRS],
        "labels_include": ["bug", "plumbing", "Enhancement"],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/releases", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
    assert len(models) == 1
    assert models[0].values[0].values == [3, 36]
    del body["labels_include"]

    response = await client.request(
        method="POST", path="/v1/metrics/releases", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
    assert len(models) == 1
    assert models[0].values[0].values == [22, 234]


@pytest.mark.parametrize("second_repo, counts", [
    ("/beta", [44, 118]),
    ("", [44, 191]),
])
async def test_release_metrics_logical(
        client, headers, logical_settings_db, release_match_setting_tag_logical_db,
        second_repo, counts):
    body = {
        "account": 1,
        "date_from": "2018-01-01",
        "date_to": "2020-03-01",
        "for": [["github.com/src-d/go-git/alpha", "github.com/src-d/go-git" + second_repo]],
        "metrics": [ReleaseMetricID.RELEASE_COUNT, ReleaseMetricID.RELEASE_PRS],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/releases", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
    assert len(models) == 1
    assert models[0].values[0].values == counts


async def test_release_metrics_participants_many_participants(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [["github.com/src-d/go-git"]],
        "with": [{"releaser": ["github.com/smola"],
                  "pr_author": ["github.com/mcuadros"],
                  "commit_author": ["github.com/smola"]},
                 {"releaser": ["github.com/mcuadros"]}],
        "metrics": [ReleaseMetricID.RELEASE_COUNT],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/releases", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
    assert len(models) == 2
    assert models[0].values[0].values[0] == 12
    assert models[1].values[0].values[0] == 21


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_smoke(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [{
            "repositories": ["github.com/src-d/go-git"],
            "pushers": ["github.com/mcuadros"],
        }],
        "metrics": [CodeCheckMetricID.SUITES_COUNT],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/code_checks", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    model = CalculatedCodeCheckMetrics.from_dict(rbody)
    assert model.to_dict() == {
        "calculated": [{"for": {"pushers": ["github.com/mcuadros"],
                                "repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [245]}]}],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
        "split_by_check_runs": None,
        "timezone": None,
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_jira(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [{
            "repositories": ["github.com/src-d/go-git"],
            "jira": {
                "labels_include": ["bug", "onboarding", "performance"],
            },
        }],
        "metrics": [CodeCheckMetricID.SUITES_COUNT],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/code_checks", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    model = CalculatedCodeCheckMetrics.from_dict(rbody)
    assert model.to_dict() == {
        "calculated": [{"for": {"jira": {"labels_include": ["bug",
                                                            "onboarding",
                                                            "performance"]},
                                "repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [60]}]}],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
        "split_by_check_runs": None,
        "timezone": None,
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_labels(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [{
            "repositories": ["github.com/src-d/go-git"],
            "labels_include": ["bug", "plumbing", "enhancement"],
        }, {
            "repositories": ["github.com/src-d/go-git"],
            "labels_include": ["bug", "plumbing", "enhancement"],
            "labels_exclude": ["xxx"],
        }, {
            "repositories": ["github.com/src-d/go-git"],
            "labels_exclude": ["bug"],
        }],
        "metrics": [CodeCheckMetricID.SUITES_COUNT],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/code_checks", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    model = CalculatedCodeCheckMetrics.from_dict(rbody)
    md = model.to_dict()
    md["calculated"].sort(key=lambda x: (x["for"].get("labels_include", []),
                                         x["for"].get("labels_exclude", [])))
    assert md == {
        "calculated": [{"for": {"labels_exclude": ["bug"],
                                "repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [734]}]},
                       {"for": {"labels_include": ["bug", "plumbing", "enhancement"],
                                "repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [23]}]},
                       {"for": {"labels_include": ["bug", "plumbing", "enhancement"],
                                "labels_exclude": ["xxx"],
                                "repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [18]}]},
                       ],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
        "split_by_check_runs": None,
        "timezone": None,
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_split_by_check_runs(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [{
            "repositories": ["github.com/src-d/go-git"],
        }],
        "metrics": [CodeCheckMetricID.SUITES_COUNT],
        "split_by_check_runs": True,
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/code_checks", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    model = CalculatedCodeCheckMetrics.from_dict(rbody)
    assert model.to_dict() == {
        "calculated": [{"check_runs": 1,
                        "for": {"repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "suites_ratio": 0.35198372329603256,
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [344]}]},
                       {"check_runs": 2,
                        "for": {"repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "suites_ratio": 0.1861648016276704,
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [183]}]},
                       {"check_runs": 3,
                        "for": {"repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "suites_ratio": 0.35096642929806715,
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [345]}]},
                       {"check_runs": 4,
                        "for": {"repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "suites_ratio": 0.09766022380467955,
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [96]}]},
                       {"check_runs": 5,
                        "for": {"repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "suites_ratio": 0.004069175991861648,
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [4]}]},
                       {"check_runs": 6,
                        "for": {"repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "suites_ratio": 0.009155645981688708,
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [9]}]}],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
        "split_by_check_runs": True,
        "timezone": None,
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_repogroups(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [{
            "repositories": ["github.com/src-d/go-git"],
            "repogroups": [[0], [0]],
        }],
        "metrics": [CodeCheckMetricID.SUITES_COUNT],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/code_checks", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    model = CalculatedCodeCheckMetrics.from_dict(rbody)
    assert model.to_dict() == {
        "calculated": [{"for": {"repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [981]}]},
                       {"for": {"repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [981]}]}],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
        "split_by_check_runs": None,
        "timezone": None,
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_authorgroups(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [{
            "repositories": ["github.com/src-d/go-git"],
            "pusher_groups": [["github.com/mcuadros"], ["github.com/erizocosmico"]],
        }],
        "metrics": [CodeCheckMetricID.SUITES_COUNT],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/code_checks", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    model = CalculatedCodeCheckMetrics.from_dict(rbody)
    assert model.to_dict() == {
        "calculated": [{"for": {"pushers": ["github.com/mcuadros"],
                                "repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [245]}]},
                       {"for": {"pushers": ["github.com/erizocosmico"],
                                "repositories": ["github.com/src-d/go-git"]},
                        "granularity": "all",
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [22]}]}],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
        "split_by_check_runs": None,
        "timezone": None,
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_lines(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [{
            "repositories": ["github.com/src-d/go-git"],
            "lines": [0, 10, 1000],
        }],
        "metrics": [CodeCheckMetricID.SUITES_COUNT],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/code_checks", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    model = CalculatedCodeCheckMetrics.from_dict(rbody)
    assert model.to_dict() == {
        "calculated": [
            {"for": {"lines": [0, 10],
                     "repositories": ["github.com/src-d/go-git"]},
             "granularity": "all",
             "values": [{"date": date(2018, 1, 12),
                         "values": [299]}]},
            {"for": {"lines": [10, 1000],
                     "repositories": ["github.com/src-d/go-git"]},
             "granularity": "all",
             "values": [{"date": date(2018, 1, 12),
                         "values": [666]}]},
        ],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
        "split_by_check_runs": None,
        "timezone": None,
    }


@pytest.mark.parametrize("account, repos, metrics, code", [
    (3, ["{1}"], [CodeCheckMetricID.SUITES_COUNT], 404),
    (2, ["github.com/src-d/go-git"], [CodeCheckMetricID.SUITES_COUNT], 422),
    (10, ["{1}"], [CodeCheckMetricID.SUITES_COUNT], 404),
    (1, None, [CodeCheckMetricID.SUITES_COUNT], 400),
    (1, ["{1}"], None, 400),
    (1, ["{1}"], ["xxx"], 400),
    (1, ["github.com/athenianco/athenian-api"], [CodeCheckMetricID.SUITES_COUNT], 403),
])
async def test_code_check_metrics_nasty_input(client, headers, account, repos, metrics, code):
    body = {
        "account": account,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [{
            "repositories": repos,
        }],
        "metrics": metrics,
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/code_checks", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == code, "Response body is : " + body


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_logical_repos(client, headers, logical_settings_db):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [{
            "repositories": ["github.com/src-d/go-git/alpha"],
            "pushers": ["github.com/mcuadros"],
        }],
        "metrics": [CodeCheckMetricID.SUITES_COUNT],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/code_checks", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, rbody
    rbody = json.loads(rbody)
    model = CalculatedCodeCheckMetrics.from_dict(rbody)
    assert model.to_dict() == {
        "calculated": [{"for": {"pushers": ["github.com/mcuadros"],
                                "repositories": ["github.com/src-d/go-git/alpha"]},
                        "granularity": "all",
                        "values": [{"date": date(2018, 1, 12),
                                    "values": [91]}]}],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
        "split_by_check_runs": None,
        "timezone": None,
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_deployment_metrics_smoke(client, headers, sample_deployments):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [{
            "repositories": ["{1}"],
            "withgroups": [{"releaser": ["github.com/mcuadros"]},
                           {"pr_author": ["github.com/mcuadros"]}],
            "environments": ["staging", "production", "mirror"],
        }],
        "metrics": [DeploymentMetricID.DEP_SUCCESS_COUNT,
                    DeploymentMetricID.DEP_DURATION_SUCCESSFUL],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/deployments", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = [CalculatedDeploymentMetric.from_dict(obj) for obj in json.loads(body)]
    assert [m.to_dict() for m in model] == [{
        "for": {
            "repositories": ["{1}"],
            "with": {"releaser": ["github.com/mcuadros"]},
            "environments": ["staging"],
        },
        "metrics": ["dep-success-count", "dep-duration-successful"],
        "granularity": "all",
        "values": [{
            "date": date(2018, 1, 12),
            "values": [3, "600s"],
            "confidence_maxs": [None, "600s"],
            "confidence_mins": [None, "600s"],
            "confidence_scores": [None, 100],
        }]}, {
        "for": {
            "repositories": ["{1}"],
            "with": {"releaser": ["github.com/mcuadros"]},
            "environments": ["production"],
        },
        "metrics": ["dep-success-count", "dep-duration-successful"],
        "granularity": "all",
        "values": [{
            "date": date(2018, 1, 12),
            "values": [3, "600s"],
            "confidence_maxs": [None, "600s"],
            "confidence_mins": [None, "600s"],
            "confidence_scores": [None, 100],
        }]}, {
        "for": {
            "repositories": ["{1}"],
            "with": {"releaser": ["github.com/mcuadros"]},
            "environments": ["mirror"],
        },
        "metrics": ["dep-success-count", "dep-duration-successful"],
        "granularity": "all",
        "values": [{
            "date": date(2018, 1, 12),
            "values": [0, None],
        }]}, {
        "for": {
            "repositories": ["{1}"],
            "with": {"pr_author": ["github.com/mcuadros"]},
            "environments": ["staging"],
        },
        "metrics": ["dep-success-count", "dep-duration-successful"],
        "granularity": "all",
        "values": [{
            "date": date(2018, 1, 12),
            "values": [3, "600s"],
            "confidence_maxs": [None, "600s"],
            "confidence_mins": [None, "600s"],
            "confidence_scores": [None, 100],
        }]}, {
        "for": {
            "repositories": ["{1}"],
            "with": {"pr_author": ["github.com/mcuadros"]},
            "environments": ["production"],
        },
        "metrics": ["dep-success-count", "dep-duration-successful"],
        "granularity": "all",
        "values": [{
            "date": date(2018, 1, 12),
            "values": [3, "600s"],
            "confidence_maxs": [None, "600s"],
            "confidence_mins": [None, "600s"],
            "confidence_scores": [None, 100],
        }]}, {
        "for": {
            "repositories": ["{1}"],
            "with": {"pr_author": ["github.com/mcuadros"]},
            "environments": ["mirror"],
        },
        "metrics": ["dep-success-count", "dep-duration-successful"],
        "granularity": "all",
        "values": [{
            "date": date(2018, 1, 12),
            "values": [0, None],
        }]}]


async def test_deployment_metrics_empty_for(client, headers, sample_deployments):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [{}],
        "metrics": [DeploymentMetricID.DEP_SUCCESS_COUNT],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/deployments", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = [CalculatedDeploymentMetric.from_dict(obj) for obj in json.loads(body)]
    assert [m.to_dict() for m in model] == [{
        "for": {},
        "granularity": "all",
        "metrics": [DeploymentMetricID.DEP_SUCCESS_COUNT],
        "values": [{
            "date": date(2018, 1, 12),
            "values": [9]}]}]


@pytest.mark.parametrize("account, date_from, date_to, repos, withgroups, metrics, code", [
    (1, "2018-01-12", "2020-01-12", ["{1}"], [], [DeploymentMetricID.DEP_PRS_COUNT], 200),
    (1, "2020-01-12", "2018-01-12", ["{1}"], [], [DeploymentMetricID.DEP_PRS_COUNT], 400),
    (2, "2018-01-12", "2020-01-12", ["github.com/src-d/go-git"], [],
     [DeploymentMetricID.DEP_PRS_COUNT], 422),
    (3, "2018-01-12", "2020-01-12", ["github.com/src-d/go-git"], [],
     [DeploymentMetricID.DEP_PRS_COUNT], 404),
    (1, "2018-01-12", "2020-01-12", ["{1}"], [], ["whatever"], 400),
    (1, "2018-01-12", "2020-01-12", ["github.com/athenianco/athenian-api"], [],
     [DeploymentMetricID.DEP_PRS_COUNT], 403),
    (1, "2018-01-12", "2020-01-12", ["{1}"], [{"pr_author": ["github.com/akbarik"]}],
     [DeploymentMetricID.DEP_PRS_COUNT], 400),
])
async def test_deployment_metrics_nasty_input(
        client, headers, account, date_from, date_to, repos, withgroups, metrics, code):
    body = {
        "account": account,
        "date_from": date_from,
        "date_to": date_to,
        "for": [{
            "repositories": [*repos],
            "withgroups": [*withgroups],
        }],
        "metrics": [*metrics],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/deployments", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == code, "Response body is : " + body


async def test_deployment_metrics_filter_labels(
        client, headers, precomputed_deployments, rdb, client_cache):
    body = {
        "account": 1,
        "date_from": "2015-01-12",
        "date_to": "2020-03-01",
        "for": [{
            "pr_labels_include": ["bug", "plumbing", "enhancement"],
        }],
        "metrics": [DeploymentMetricID.DEP_COUNT],
        "granularities": ["all"],
    }

    async def request():
        response = await client.request(
            method="POST", path="/v1/metrics/deployments", headers=headers, json=body,
        )
        rbody = (await response.read()).decode("utf-8")
        assert response.status == 200, "Response body is : " + rbody
        return [CalculatedDeploymentMetric.from_dict(obj) for obj in json.loads(rbody)]

    model = await request()
    assert model[0].values[0].values[0] == 1

    del body["for"][0]["pr_labels_include"]
    body["for"][0]["pr_labels_exclude"] = ["bug", "plumbing", "enhancement"]
    model = await request()
    assert model[0].values[0].values[0] == 0

    del body["for"][0]["pr_labels_exclude"]
    await rdb.execute(insert(DeployedLabel).values({
        DeployedLabel.account_id: 1,
        DeployedLabel.deployment_name: "Dummy deployment",
        DeployedLabel.key: "nginx",
        DeployedLabel.value: 504,
    }))
    body["for"][0]["with_labels"] = {"nginx": 503}
    model = await request()
    assert model[0].values[0].values[0] == 0

    body["for"][0]["with_labels"] = {"nginx": 504}
    model = await request()
    assert model[0].values[0].values[0] == 1

    del body["for"][0]["with_labels"]
    body["for"][0]["without_labels"] = {"nginx": 503}
    model = await request()
    assert model[0].values[0].values[0] == 1

    body["for"][0]["without_labels"] = {"nginx": 504}
    model = await request()
    assert model[0].values[0].values[0] == 0


async def test_deployment_metrics_environments(
        client, headers, sample_deployments, rdb, client_cache):
    await rdb.execute(delete(DeployedComponent)
                      .where(DeployedComponent.deployment_name == "staging_2018_12_02"))
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [{
            "envgroups": [["production"]],
        }],
        "metrics": [DeploymentMetricID.DEP_SUCCESS_COUNT,
                    DeploymentMetricID.DEP_DURATION_SUCCESSFUL],
        "granularities": ["all"],
    }

    async def request():
        response = await client.request(
            method="POST", path="/v1/metrics/deployments", headers=headers, json=body,
        )
        rbody = (await response.read()).decode("utf-8")
        assert response.status == 200, "Response body is : " + rbody
        model = [CalculatedDeploymentMetric.from_dict(obj) for obj in json.loads(rbody)]
        return model

    model = await request()
    assert len(model) == 1
    assert model[0].values[0].values[0] == 3
    body["for"][0]["envgroups"] = [["staging"], ["production"]]
    model = await request()
    assert len(model) == 2
    assert model[0].for_.environments == ["staging"]
    assert model[0].values[0].values[0] == 3
    assert model[1].for_.environments == ["production"]
    assert model[1].values[0].values[0] == 3


async def test_deployment_metrics_with(client, headers, sample_deployments):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [{
            "with": {"pr_author": ["github.com/mcuadros"]},
            "environments": ["production"],
        }],
        "metrics": [DeploymentMetricID.DEP_SUCCESS_COUNT,
                    DeploymentMetricID.DEP_DURATION_SUCCESSFUL],
        "granularities": ["all"],
    }

    response = await client.request(
        method="POST", path="/v1/metrics/deployments", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    model = [CalculatedDeploymentMetric.from_dict(obj) for obj in json.loads(rbody)]

    assert len(model) == 1
    assert model[0].values[0].values[0] == 3


@pytest.mark.app_validate_responses(False)
async def test_deployment_metrics_team(client, headers, sample_deployments, sample_team):
    team_str = "{%d}" % sample_team
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [{
            "repositories": ["{1}"],
            "withgroups": [{"releaser": [team_str]},
                           {"pr_author": [team_str]}],
            "environments": ["production"],
        }],
        "metrics": [DeploymentMetricID.DEP_SUCCESS_COUNT,
                    DeploymentMetricID.DEP_DURATION_SUCCESSFUL],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/deployments", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = [CalculatedDeploymentMetric.from_dict(obj) for obj in json.loads(body)]
    assert [m.to_dict() for m in model] == [{
        "for": {
            "repositories": ["{1}"],
            "with": {"releaser": [team_str]},
            "environments": ["production"],
        },
        "metrics": ["dep-success-count", "dep-duration-successful"],
        "granularity": "all",
        "values": [{
            "date": date(2018, 1, 12),
            "values": [3, "600s"],
            "confidence_maxs": [None, "600s"],
            "confidence_mins": [None, "600s"],
            "confidence_scores": [None, 100],
        }]}, {
        "for": {
            "repositories": ["{1}"],
            "with": {"pr_author": [team_str]},
            "environments": ["production"],
        },
        "metrics": ["dep-success-count", "dep-duration-successful"],
        "granularity": "all",
        "values": [{
            "date": date(2018, 1, 12),
            "values": [3, "600s"],
            "confidence_maxs": [None, "600s"],
            "confidence_mins": [None, "600s"],
            "confidence_scores": [None, 100],
        }]}]
