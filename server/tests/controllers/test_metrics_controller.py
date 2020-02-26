from datetime import timedelta

import pandas as pd
import pytest

from athenian.api import FriendlyJson
from athenian.api.models.web import CalculatedMetrics, MetricID


@pytest.mark.parametrize("metric", MetricID.ALL)
async def test_calc_metrics_prs_smoke(client, metric, headers):
    """Trivial test to prove that at least something is working."""
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
        "metrics": [metric],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularity": "week",
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics_line", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedMetrics.from_dict(FriendlyJson.loads(body))
    assert len(cm.calculated[0].values) > 0


@pytest.mark.parametrize("granularity", ["day", "week", "month"])
async def test_calc_metrics_prs_all_time(client, granularity, headers):
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
        "metrics": [MetricID.PR_WIP_TIME,
                    MetricID.PR_REVIEW_TIME,
                    MetricID.PR_MERGING_TIME,
                    MetricID.PR_RELEASE_TIME,
                    MetricID.PR_LEAD_TIME,
                    MetricID.PR_WAIT_FIRST_REVIEW_TIME],
        "date_from": "2015-10-13",
        "date_to": "2019-03-15",
        "granularity": granularity,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics_line", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedMetrics.from_dict(FriendlyJson.loads(body))
    cmins = cmaxs = cscores = 0
    for m, calc in zip(cm.metrics, cm.calculated):
        nonzero = 0
        for val in calc.values:
            for t in val.values:
                if t is None:
                    continue
                assert pd.to_timedelta(t) >= timedelta(0), \
                    "Metric: %s\nValues: %s" % (m, val.values)
                nonzero += pd.to_timedelta(t) > timedelta(0)
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
        if m != "pr-release-time":
            assert nonzero > 0, str(m)
    assert cmins > 0
    assert cmaxs > 0
    assert cscores > 0


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
        "metrics": list(MetricID.ALL),
        "date_from": "2015-10-13",
        "date_to": "2019-03-15",
        "granularity": "month",
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics_line", headers=headers, json=body,
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
        "metrics": list(MetricID.ALL),
    }
    response = await client.request(
        method="POST", path="/v1/metrics_line", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedMetrics.from_dict(FriendlyJson.loads(body))
    assert len(cm.calculated[0].values) > 0


async def test_calc_metrics_prs_bad_date(client, headers):
    """What if we specify a date that does not exist?"""
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
        "metrics": [MetricID.PR_LEAD_TIME],
        "date_from": "2015-10-13",
        "date_to": "2020-02-30",  # 30th of February does not exist
        "granularity": "week",
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics_line", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 400, "Response body is : " + body


@pytest.mark.parametrize("account", [3, 10])
async def test_calc_metrics_prs_reposet_bad_account(client, account, headers):
    """What if we specify a account that the user does not belong to or does not exist?"""
    body = {
        "for": [
            {
                "developers": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                "repositories": ["{1}"],
            },
        ],
        "metrics": [MetricID.PR_LEAD_TIME],
        "date_from": "2015-10-13",
        "date_to": "2020-02-20",
        "granularity": "week",
        "account": account,
    }
    response = await client.request(
        method="POST", path="/v1/metrics_line", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 403, "Response body is : " + body


async def test_calc_metrics_prs_reposet(client, headers):
    """Substitute {id} with the real repos."""
    body = {
        "for": [
            {
                "developers": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                "repositories": ["{1}"],
            },
        ],
        "metrics": [MetricID.PR_LEAD_TIME],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularity": "week",
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics_line", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedMetrics.from_dict(FriendlyJson.loads(body))
    assert len(cm.calculated[0].values) > 0
    assert cm.calculated[0]._for.repositories == ["{1}"]


@pytest.mark.parametrize("metric", [MetricID.PR_WIP_COUNT,
                                    MetricID.PR_REVIEW_COUNT,
                                    MetricID.PR_MERGING_COUNT,
                                    # FIXME(vmarkovtsev): no releases
                                    # MetricID.PR_RELEASE_COUNT,
                                    # MetricID.PR_LEAD_COUNT,
                                    MetricID.PR_OPENED,
                                    MetricID.PR_CLOSED,
                                    MetricID.PR_MERGED,
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
        method="POST", path="/v1/metrics_line", headers=headers, json=body,
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
        "metrics": [MetricID.PR_WIP_TIME,
                    MetricID.PR_REVIEW_TIME,
                    MetricID.PR_MERGING_TIME,
                    MetricID.PR_RELEASE_TIME,
                    MetricID.PR_LEAD_TIME],
        "date_from": "2019-02-25",
        "date_to": "2019-02-28",
        "granularity": "week",
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics_line", headers=headers, json=body,
    )
    assert response.status == 200
