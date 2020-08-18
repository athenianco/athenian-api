import itertools

import pytest

from athenian.api import FriendlyJson
from athenian.api.models.web import CalculatedPullRequestHistogram, PullRequestMetricID


@pytest.mark.parametrize(
    "metric, cached, result",
    itertools.chain(zip((PullRequestMetricID.PR_CYCLE_TIME,
                         PullRequestMetricID.PR_LEAD_TIME), itertools.repeat(False),
                        [(["60s", "122s", "249s", "507s", "1033s", "2105s", "4288s", "8737s",
                           "17799s", "36261s", "73870s", "150489s", "306576s", "624554s",
                           "1272338s", "2591999s"],
                          [4, 4, 10, 18, 15, 22, 20, 20, 17, 27, 34, 53, 63, 73, 298]),
                         (["687s", "1038s", "1567s", "2365s", "3571s", "5390s", "8135s", "12280s",
                           "18535s", "27977s", "42229s", "63740s", "96209s", "145217s", "219190s",
                           "330844s", "499373s", "753751s", "1137706s", "1717246s", "2591999s"],
                          [4, 1, 5, 6, 2, 3, 5, 3, 2, 10, 1, 9, 5, 8, 23, 10, 32, 29, 35, 231])]),
                    [(PullRequestMetricID.PR_WIP_TIME,
                      True,
                      (["60s", "128s", "275s", "590s", "1266s", "2714s", "5818s", "12470s",
                        "26730s", "57293s", "122803s", "263218s", "564186s", "1209285s",
                        "2591999s"],
                       [174, 65, 69, 42, 37, 35, 42, 27, 41, 40, 34, 28, 20, 24]))]))
async def test_calc_histogram_prs_smoke(
        client, headers, metric, cached, app, client_cache, result):
    if cached:
        app._cache = client_cache
    repeats = 1 if not cached else 2
    for _ in range(repeats):
        body = {
            "for": [
                {
                    "with": {},
                    "repositories": [
                        "github.com/src-d/go-git",
                    ],
                },
            ],
            "metrics": [metric],
            "date_from": "2015-10-13",
            "date_to": "2020-01-23",
            "scale": "log",
            "exclude_inactive": False,
            "account": 1,
        }
        response = await client.request(
            method="POST", path="/v1/histograms/prs", headers=headers, json=body,
        )
        body = (await response.read()).decode("utf-8")
        assert response.status == 200, "Response body is : " + body
        body = FriendlyJson.loads(body)
        [CalculatedPullRequestHistogram.from_dict(item) for item in body]
        assert body == [{
            "for": {"repositories": ["github.com/src-d/go-git"],
                    "with": {"author": None, "reviewer": None, "commit_author": None,
                             "commit_committer": None, "commenter": None, "merger": None,
                             "releaser": None},
                    "labels_include": None,
                    "jira": None},
            "metric": metric,
            "scale": "log",
            "ticks": result[0],
            "frequencies": result[1],
        }]


@pytest.mark.parametrize(
    "metric, date_to, bins, scale, quantiles, account, status",
    [
        (PullRequestMetricID.PR_OPENED, "2020-01-23", 10, "log", [0, 1], 1, 400),
        (PullRequestMetricID.PR_CYCLE_TIME, "2020-01-23", -1, "log", [0, 1], 1, 400),
        (PullRequestMetricID.PR_CYCLE_TIME, "2020-01-23", 10, "xxx", [0, 1], 1, 400),
        (PullRequestMetricID.PR_CYCLE_TIME, "2015-01-23", 10, "linear", [0, 1], 1, 400),
        (PullRequestMetricID.PR_CYCLE_TIME, "2020-01-23", 10, "linear", [0, 1], 2, 422),
        (PullRequestMetricID.PR_CYCLE_TIME, "2020-01-23", 10, "linear", [0, 1], 4, 404),
        (PullRequestMetricID.PR_CYCLE_TIME, "2015-11-23", 10, "linear", [-1, 1], 1, 400),
    ],
)
async def test_calc_histogram_prs_nasty_input(
        client, headers, metric, date_to, bins, scale, quantiles, account, status):
    body = {
        "for": [
            {
                "with": {},
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
        ],
        "metrics": [metric],
        "date_from": "2015-10-13",
        "date_to": date_to,
        "scale": scale,
        "bins": bins,
        "quantiles": quantiles,
        "exclude_inactive": False,
        "account": account,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/prs", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == status, "Response body is : " + body


async def test_calc_histogram_prs_multiple(client, headers):
    body = {
        "for": [
            {
                "with": {},
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
            {
                "with": {"merger": ["github.com/mcuadros"]},
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
        ],
        "metrics": [PullRequestMetricID.PR_RELEASE_TIME, PullRequestMetricID.PR_REVIEW_TIME],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "scale": "linear",
        "bins": 10,
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/prs", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = FriendlyJson.loads(body)
    assert len(body) == 4
    for i in range(4):
        for j in range(i + 1, 4):
            assert body[i] != body[j], "%d == %d" % (i, j)


async def test_calc_histogram_prs_size(client, headers):
    body = {
        "for": [
            {
                "with": {"merger": ["github.com/mcuadros"]},
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
        ],
        "metrics": [PullRequestMetricID.PR_SIZE],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "scale": "linear",
        "bins": 10,
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/prs", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = FriendlyJson.loads(body)
    assert body == [{"for": {"repositories": ["github.com/src-d/go-git"],
                             "with": {"author": None, "reviewer": None, "commit_author": None,
                                      "commit_committer": None, "commenter": None,
                                      "merger": ["github.com/mcuadros"], "releaser": None},
                             "labels_include": None, "jira": None},
                     "metric": "pr-size", "scale": "linear",
                     "ticks": [0.0, 1109.9, 2219.8, 3329.7000000000003, 4439.6, 5549.5,
                               6659.400000000001, 7769.300000000001, 8879.2, 9989.1, 11099.0],
                     "frequencies": [465, 17, 2, 2, 1, 0, 1, 0, 0, 1]}]
