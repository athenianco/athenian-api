from collections import defaultdict
from datetime import date, timedelta
import itertools

import pandas as pd
import pytest

from athenian.api import FriendlyJson
from athenian.api.models.web import CalculatedDeveloperMetrics, CalculatedPullRequestMetrics, \
    CodeBypassingPRsMeasurement, DeveloperMetricID, PullRequestMetricID


@pytest.mark.parametrize(
    "metric, cached",
    itertools.chain(zip(PullRequestMetricID, itertools.repeat(False)),
                    [(PullRequestMetricID.PR_WIP_TIME, True)]))
async def test_calc_metrics_prs_smoke(client, metric, headers, cached, app, cache):
    """Trivial test to prove that at least something is working."""
    if cached:
        app._cache = cache
    repeats = 1 if not cached else 2
    for _ in range(repeats):
        body = {
            "for": [
                {
                    "developers": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                    "repositories": [
                        "github.com/src-d/go-git",
                    ],
                },
            ],
            "metrics": [metric],
            "date_from": "2015-10-13",
            "date_to": "2020-01-23",
            "granularities": ["week"],
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
    body = {
        "for": [
            {
                "developers": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
            {
                "developers": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
        ],
        "metrics": [PullRequestMetricID.PR_WIP_TIME,
                    PullRequestMetricID.PR_REVIEW_TIME,
                    PullRequestMetricID.PR_MERGING_TIME,
                    PullRequestMetricID.PR_RELEASE_TIME,
                    PullRequestMetricID.PR_LEAD_TIME,
                    PullRequestMetricID.PR_CYCLE_TIME,
                    PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME],
        "date_from": "2015-10-13",
        "date_to": "2019-03-15",
        "granularities": ["day", "week", "month"],
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
        assert calc.for_.developers == ["github.com/vmarkovtsev", "github.com/mcuadros"]
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
                "developers": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                "repositories": [
                    "github.com/src-d/go-git",
                    "github.com/athenianco/athenian-api",
                ],
            },
        ],
        "metrics": list(PullRequestMetricID),
        "date_from": "2015-10-13",
        "date_to": "2019-03-15",
        "granularity": "month",
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/prs", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


@pytest.mark.parametrize(("devs", "date_from"),
                         ([{"developers": []}, "2019-11-28"], [{}, "2018-09-28"]))
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
        "granularity": "month",
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


@pytest.mark.parametrize("account, date_to, code",
                         [(3, "2020-02-22", 403), (10, "2020-02-22", 403), (1, "2015-10-13", 200),
                          (1, "2010-01-11", 400), (1, "2020-01-32", 400)])
async def test_calc_metrics_prs_nasty_input(client, headers, account, date_to, code):
    """What if we specify a date that does not exist?"""
    body = {
        "for": [
            {
                "developers": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                "repositories": [
                    "{1}",
                ],
            },
            {
                "developers": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
        ],
        "metrics": [PullRequestMetricID.PR_LEAD_TIME],
        "date_from": "2015-10-13",
        "date_to": date_to,
        "granularity": "week",
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
                "developers": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                "repositories": ["{1}"],
            },
        ],
        "metrics": [PullRequestMetricID.PR_LEAD_TIME],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularity": "all",
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
                                    PullRequestMetricID.PR_CLOSED,
                                    PullRequestMetricID.PR_MERGED,
                                    PullRequestMetricID.PR_RELEASED,
                                    ])
async def test_calc_metrics_prs_counts_sums(client, headers, metric):
    body = {
        "for": [
            {
                "developers": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                "repositories": ["{1}"],
            },
        ],
        "metrics": [metric],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularity": "month",
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
                "developers": [],
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
        "granularity": "week",
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
        "granularity": "month",
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
        if flow is None:
            assert closed is None
            continue
        assert flow == opened / closed, "%.3f != %d / %d" % (flow, opened, closed)


async def test_code_bypassing_prs_smoke(client, headers):
    body = {
        "account": 1,
        "date_from": "2019-01-12",
        "date_to": "2020-02-22",
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
