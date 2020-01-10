from datetime import timedelta

import pandas as pd
import pytest

from athenian.api import FriendlyJson
from athenian.api.models.web import CalculatedMetrics, Granularity, MetricID


@pytest.mark.parametrize("metric", MetricID.ALL)
async def test_calc_metrics_line_smoke(client, metric):
    """Trivial test to prove that at least something is working."""
    body = {
        "for": [
            {
                "developers": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                "repositories": [
                    "github.com/src-d/go-git",
                    "github.com/athenianco/athenian-api",
                ],
            },
            {
                "developers": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                "repositories": [
                    "github.com/src-d/go-git",
                    "github.com/athenianco/athenian-api",
                ],
            },
        ],
        "metrics": [metric],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularity": "week",
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="POST", path="/v1/metrics_line", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body


@pytest.mark.parametrize("granularity", Granularity.ALL)
async def test_calc_metrics_line_all(client, granularity):
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
        "date_to": "2020-01-23",
        "granularity": granularity,
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    response = await client.request(
        method="POST", path="/v1/metrics_line", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    cm = CalculatedMetrics.from_dict(FriendlyJson.loads(body))
    for m, calc in zip(cm.metrics, cm.calculated):
        for val in calc.values:
            for t in val.values:
                assert pd.to_timedelta(t) >= timedelta(0), \
                    "Metric: %s\nValues: %s" % (m, val.values)
            for t in val.confidence_mins:
                assert pd.to_timedelta(t) >= timedelta(0), \
                    "Metric: %s\nConfidence mins: %s" % (m, val.confidence_mins)
            for t in val.confidence_maxs:
                assert pd.to_timedelta(t) >= timedelta(0), \
                    "Metric: %s\nConfidence maxs: %s" % (m, val.confidence_maxs)
            for s in val.confidence_scores:
                assert 0 <= s <= 100, \
                    "Metric: %s\nConfidence scores: %s" % (m, val.confidence_scores)
