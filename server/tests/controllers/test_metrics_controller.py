from collections import defaultdict
from datetime import date, timedelta
import itertools
import json

import pandas as pd
import pytest

from athenian.api import FriendlyJson
from athenian.api.models.web import CalculatedDeveloperMetrics, CalculatedPullRequestMetrics, \
    CalculatedReleaseMetric, CodeBypassingPRsMeasurement, DeveloperMetricID, PullRequestMetricID, \
    PullRequestWith, ReleaseMetricID


@pytest.mark.parametrize(
    "metric, cached",
    itertools.chain(zip(PullRequestMetricID, itertools.repeat(False)),
                    [(PullRequestMetricID.PR_WIP_TIME, True)]))
async def test_calc_metrics_prs_smoke(client, metric, headers, cached, app, client_cache):
    """Trivial test to prove that at least something is working."""
    if cached:
        app._cache = client_cache
    repeats = 1 if not cached else 2
    for _ in range(repeats):
        body = {
            "for": [
                {
                    "with": {
                        "author": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                    },
                    "repositories": [
                        "github.com/src-d/go-git",
                    ],
                },
            ],
            "metrics": [metric],
            "date_from": "2015-10-13",
            "date_to": "2020-01-23",
            "granularities": ["week"],
            "exclude_inactive": False,
            "account": 1,
        }
        response = await client.request(
            method="POST", path="/v1/metrics/prs", headers=headers, json=body,
        )
        body = (await response.read()).decode("utf-8")
        assert response.status == 200, "Response body is : " + body
        cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
        assert len(cm.calculated) == 1
        assert len(cm.calculated[0].values) > 0
        nonzero = 0
        for val in cm.calculated[0].values:
            assert len(val.values) == 1
            nonzero += val.values[0] is not None
        assert nonzero > 0


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
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
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
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


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
        }],
        "granularities": ["month"],
        "exclude_inactive": False,
        "account": 1,
        "metrics": list(PullRequestMetricID),
    }
    response = await client.request(
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert len(cm.calculated[0].values) > 0


@pytest.mark.parametrize("account, date_to, quantiles, code",
                         [(3, "2020-02-22", [0, 1], 403),
                          (10, "2020-02-22", [0, 1], 403),
                          (1, "2015-10-13", [0, 1], 200),
                          (1, "2010-01-11", [0, 1], 400),
                          (1, "2020-01-32", [0, 1], 400),
                          (1, "2020-01-01", [-1, 0.5], 400),
                          (1, "2020-01-01", [0, -1], 400),
                          (1, "2020-01-01", [10, 20], 400),
                          (1, "2020-01-01", [0.5, 0.25], 400),
                          (1, "2020-01-01", [0.5, 0.5], 400)])
async def test_calc_metrics_prs_nasty_input(client, headers, account, date_to, quantiles, code):
    """What if we specify a date that does not exist?"""
    body = {
        "for": [
            {
                "with": {"merger": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                "repositories": [
                    "{1}",
                ],
            },
            {
                "with": {"releaser": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
        ],
        "metrics": [PullRequestMetricID.PR_LEAD_TIME],
        "date_from": "2015-10-13",
        "date_to": date_to,
        "granularities": ["week"],
        "quantiles": quantiles,
        "exclude_inactive": False,
        "account": account,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == code, "Response body is : " + body


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
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert len(cm.calculated[0].values) > 0
    assert cm.calculated[0].for_.repositories == ["{1}"]


@pytest.mark.parametrize("metric", [PullRequestMetricID.PR_WIP_COUNT,
                                    PullRequestMetricID.PR_REVIEW_COUNT,
                                    PullRequestMetricID.PR_MERGING_COUNT,
                                    PullRequestMetricID.PR_RELEASE_COUNT,
                                    PullRequestMetricID.PR_LEAD_COUNT,
                                    PullRequestMetricID.PR_CYCLE_COUNT,
                                    PullRequestMetricID.PR_OPENED,
                                    PullRequestMetricID.PR_REVIEWED,
                                    PullRequestMetricID.PR_NOT_REVIEWED,
                                    PullRequestMetricID.PR_CLOSED,
                                    PullRequestMetricID.PR_MERGED,
                                    PullRequestMetricID.PR_REJECTED,
                                    PullRequestMetricID.PR_DONE,
                                    ])
async def test_calc_metrics_prs_counts_sums(client, headers, metric):
    body = {
        "for": [
            {
                "with": {k: ["github.com/vmarkovtsev", "github.com/mcuadros"]
                         for k in PullRequestWith().openapi_types},
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
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
    )
    assert response.status == 200
    body = FriendlyJson.loads((await response.read()).decode("utf-8"))
    s = 0
    for item in body["calculated"][0]["values"]:
        assert "confidence_mins" not in item
        assert "confidence_maxs" not in item
        assert "confidence_scores" not in item
        val = item["values"][0]
        if val is not None:
            s += val
    assert s > 0


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
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
    )
    assert response.status == 200


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
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
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
        assert flow == (opened + 1) / (closed + 1), "%.3f != %d / %d" % (flow, opened, closed)


async def test_calc_metrics_prs_exclude_inactive(client, headers):
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
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert cm.calculated[0].values[0].values[0] == 6


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
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert cm.calculated[0].values[0].values[0] == 1


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
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert cm.calculated[0].values[0].values[0] == 6


async def test_calc_metrics_quantiles(client, headers):
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
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(rbody))
    wip1 = cm.calculated[0].values[0].values[0]
    body["quantiles"] = [0.05, 0.95]
    response = await client.request(
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(rbody))
    wip2 = cm.calculated[0].values[0].values[0]
    assert int(wip1[:-1]) > int(wip2[:-1])


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
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedPullRequestMetrics.from_dict(FriendlyJson.loads(body))
    assert len(cm.calculated[0].values) > 0
    assert cm.calculated[0].values[0].values[0] == "478544s"


async def test_code_bypassing_prs_smoke(client, headers):
    body = {
        "account": 1,
        "date_from": "2019-01-12",
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
    assert len(ms) == 14
    for s in ms:
        assert date(year=2019, month=1, day=12) <= s.date <= date(year=2020, month=2, day=22)
        assert s.total_commits >= 0
        assert s.total_lines >= 0
        assert 0 <= s.bypassed_commits <= s.total_commits
        assert 0 <= s.bypassed_lines <= s.total_lines
    for i in range(len(ms) - 1):
        assert ms[i].date < ms[i + 1].date


@pytest.mark.parametrize("account, date_to, code",
                         [(3, "2020-02-22", 403), (10, "2020-02-22", 403), (1, "2019-01-12", 200),
                          (1, "2019-01-11", 400), (1, "2019-01-32", 400)])
async def test_code_bypassing_prs_nasty_input(client, headers, account, date_to, code):
    body = {
        "account": account,
        "date_from": "2019-01-12",
        "date_to": date_to,
        "in": ["{1}"],
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
}

developer_metric_be_stats = {
    "dev-commits-pushed": [[207], [8], [86], [11]],
    "dev-lines-changed": [[34494], [482], [6159], [592]],
    "dev-prs-created": [[0], [0], [1], [0]],
    "dev-prs-reviewed": [[2], [4], [3], [4]],
    "dev-prs-merged": [[6], [0], [0], [0]],
    "dev-releases": [[21], [0], [0], [0]],
    "dev-reviews": [[8], [6], [7], [4]],
    "dev-review-approvals": [[1], [3], [2], [3]],
    "dev-review-rejections": [[1], [1], [0], [0]],
    "dev-review-neutrals": [[6], [2], [5], [1]],
    "dev-pr-comments": [[10], [6], [0], [2]],
    "dev-regular-pr-comments": [[3], [1], [0], [1]],
    "dev-review-pr-comments": [[7], [5], [5], [1]],
}


@pytest.mark.parametrize("metric, value", [(m, developer_metric_mcuadros_stats[m])
                                           for m in DeveloperMetricID])
async def test_developer_metrics_single(client, headers, metric, value):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [
            {"repositories": ["{1}"], "developers": ["github.com/mcuadros"]},
        ],
        "metrics": [metric],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result: CalculatedDeveloperMetrics
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert result.metrics == [metric]
    assert result.date_from == date(year=2018, month=1, day=12)
    assert result.date_to == date(year=2020, month=3, day=1)
    assert len(result.calculated) == 1
    assert result.calculated[0].for_.repositories == ["{1}"]
    assert result.calculated[0].for_.developers == ["github.com/mcuadros"]
    assert result.calculated[0].values == [[value]]


@pytest.mark.parametrize("dev", ["mcuadros", "vmarkovtsev", "xxx", "EmrysMyrddin"])
async def test_developer_metrics_all(client, headers, dev):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "timezone": 60,
        "for": [
            {"repositories": ["{1}"], "developers": ["github.com/" + dev]},
        ],
        "metrics": sorted(DeveloperMetricID),
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200, (await response.read()).decode("utf-8")
    result: CalculatedDeveloperMetrics
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert set(result.metrics) == set(DeveloperMetricID)
    assert result.date_from == date(year=2018, month=1, day=12)
    assert result.date_to == date(year=2020, month=3, day=1)
    assert len(result.calculated) == 1
    assert result.calculated[0].for_.repositories == ["{1}"]
    assert result.calculated[0].for_.developers == ["github.com/" + dev]
    assert len(result.calculated[0].values) == 1
    assert len(result.calculated[0].values[0]) == len(DeveloperMetricID)
    if dev == "mcuadros":
        for v, m in zip(result.calculated[0].values[0], sorted(DeveloperMetricID)):
            assert v == developer_metric_mcuadros_stats[m], m
    elif dev == "xxx":
        assert all(v == 0 for v in result.calculated[0].values[0]), \
            "%s\n%s" % (str(result.calculated[0].values[0]), sorted(DeveloperMetricID))
    else:
        assert all(isinstance(v, int) for v in result.calculated[0].values[0]), \
            "%s\n%s" % (str(result.calculated[0].values[0]), sorted(DeveloperMetricID))


@pytest.mark.parametrize("metric, value", [(m, developer_metric_be_stats[m])
                                           for m in DeveloperMetricID])
async def test_developer_metrics_labels_include(client, headers, metric, value):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
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
    result: CalculatedDeveloperMetrics
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert result.calculated[0].for_.labels_include == ["bug", "enhancement"]
    assert result.calculated[0].values == value


async def test_developer_metrics_labels_exclude(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
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
    result: CalculatedDeveloperMetrics
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert not result.calculated[0].for_.labels_include
    assert result.calculated[0].for_.labels_exclude == ["bug", "enhancement"]
    assert result.calculated[0].values == [[14], [8], [26], [7]]


async def test_developer_metrics_labels_include_exclude(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
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
    result: CalculatedDeveloperMetrics
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert result.calculated[0].for_.labels_include == ["bug"]
    assert result.calculated[0].for_.labels_exclude == ["enhancement"]
    assert result.calculated[0].values == [[0], [0], [1], [0]]


async def test_developer_metrics_labels_contradiction(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
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
    result: CalculatedDeveloperMetrics
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert result.calculated[0].for_.labels_include == ["bug"]
    assert result.calculated[0].for_.labels_exclude == ["bug"]
    assert result.calculated[0].values == [[0], [0], [0], [0]]


@pytest.mark.parametrize("account, date_to, code",
                         [(3, "2020-02-22", 403), (10, "2020-02-22", 403), (1, "2018-01-12", 200),
                          (1, "2018-01-11", 400), (1, "2019-01-32", 400)])
async def test_developer_metrics_nasty_input(client, headers, account, date_to, code):
    body = {
        "account": account,
        "date_from": "2018-01-12",
        "date_to": date_to,
        "for": [
            {"repositories": ["{1}"], "developers": ["github.com/mcuadros"]},
        ],
        "metrics": sorted(DeveloperMetricID),
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == code


async def test_developer_metrics_order(client, headers):
    """https://athenianco.atlassian.net/browse/DEV-247"""
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
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
    result: CalculatedDeveloperMetrics
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert result.calculated[0].for_.developers == body["for"][0]["developers"]
    assert result.calculated[0].values == [[14], [8]]
    body["for"][0]["developers"] = list(reversed(body["for"][0]["developers"]))
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")))
    assert result.calculated[0].for_.developers == body["for"][0]["developers"]
    assert result.calculated[0].values == [[8], [14]]


async def test_release_metrics_smoke(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [["{1}"]],
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
            exist = any(v is not None for v in mv.values)
            for metric, value in zip(model.metrics, mv.values):
                if "branch" in metric:
                    assert value is None, metric
                else:
                    if exist:
                        assert value is not None, metric
        if model.granularity == "all":
            assert len(model.values) == 1
            assert any(v is not None for v in model.values[0].values)
        else:
            assert any(v is not None for values in model.values for v in values.values)
            assert len(model.values) == 9


@pytest.mark.parametrize("account, date_to, quantiles, extra_metrics, code",
                         [(3, "2020-02-22", [0, 1], [], 403),
                          (10, "2020-02-22", [0, 1], [], 403),
                          (1, "2015-10-13", [0, 1], [], 200),
                          (1, "2015-10-13", [0, 1], ["whatever"], 400),
                          (1, "2010-01-11", [0, 1], [], 400),
                          (1, "2020-01-32", [0, 1], [], 400),
                          (1, "2020-01-01", [-1, 0.5], [], 400),
                          (1, "2020-01-01", [0, -1], [], 400),
                          (1, "2020-01-01", [10, 20], [], 400),
                          (1, "2020-01-01", [0.5, 0.25], [], 400),
                          (1, "2020-01-01", [0.5, 0.5], [], 400)])
async def test_release_metrics_nasty_input(
        client, headers, account, date_to, quantiles, extra_metrics, code):
    body = {
        "for": [["{1}"], ["{1}"]],
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


async def test_release_metrics_quantiles(client, headers):
    for q, gt in ((0.95, "2176285s"), (1, "2779256s")):
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
        assert model.values[0].values == [gt]
