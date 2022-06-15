import itertools

import pytest

from athenian.api.cache import CACHE_VAR_NAME
from athenian.api.models.web import (
    CalculatedPullRequestHistogram,
    CodeCheckMetricID,
    PullRequestMetricID,
)
from athenian.api.serialization import FriendlyJson


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize(
    "metric, cached, result",
    itertools.chain(
        zip(
            (PullRequestMetricID.PR_CYCLE_TIME, PullRequestMetricID.PR_LEAD_TIME),
            itertools.repeat(False),
            [
                (
                    [
                        "60s",
                        "116s",
                        "227s",
                        "443s",
                        "865s",
                        "1685s",
                        "3284s",
                        "6399s",
                        "12470s",
                        "24300s",
                        "47350s",
                        "92266s",
                        "179789s",
                        "350333s",
                        "682652s",
                        "1330200s",
                        "2591999s",
                    ],
                    [4, 4, 7, 14, 12, 15, 21, 19, 18, 17, 29, 25, 57, 63, 69, 304],
                    {"left": "123003s", "right": "2592000s"},
                ),
                (
                    [
                        "1368s",
                        "1929s",
                        "2718s",
                        "3830s",
                        "5398s",
                        "7607s",
                        "10719s",
                        "15106s",
                        "21287s",
                        "29997s",
                        "42272s",
                        "59568s",
                        "83942s",
                        "118290s",
                        "166691s",
                        "234898s",
                        "331012s",
                        "466454s",
                        "657316s",
                        "926274s",
                        "1305283s",
                        "1839373s",
                        "2591999s",
                    ],
                    [1, 0, 7, 1, 2, 3, 0, 1, 4, 5, 0, 4, 5, 5, 8, 17, 10, 27, 30, 20, 31, 233],
                    {"left": "691126s", "right": "2592000s"},
                ),
            ],
        ),
        [
            (
                PullRequestMetricID.PR_WIP_TIME,
                True,
                (
                    [
                        "60s",
                        "128s",
                        "275s",
                        "590s",
                        "1266s",
                        "2714s",
                        "5818s",
                        "12470s",
                        "26730s",
                        "57293s",
                        "122803s",
                        "263218s",
                        "564186s",
                        "1209285s",
                        "2591999s",
                    ],
                    [175, 65, 67, 42, 37, 35, 41, 26, 40, 42, 35, 29, 20, 24],
                    {"left": "119s", "right": "43877s"},
                ),
            ),
        ],
    ),
)
async def test_calc_histogram_prs_smoke(
    client,
    headers,
    metric,
    cached,
    app,
    client_cache,
    result,
):
    if cached:
        app.app[CACHE_VAR_NAME] = client_cache
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
            "histograms": [
                {
                    "metric": metric,
                    "scale": "log",
                },
            ],
            "date_from": "2015-10-13",
            "date_to": "2020-01-23",
            "exclude_inactive": False,
            "account": 1,
        }
        response = await client.request(
            method="POST", path="/v1/histograms/pull_requests", headers=headers, json=body,
        )
        body = (await response.read()).decode("utf-8")
        assert response.status == 200, "Response body is : " + body
        body = FriendlyJson.loads(body)
        for item in body:
            CalculatedPullRequestHistogram.from_dict(item)

        assert body == [
            {
                "for": {"repositories": ["github.com/src-d/go-git"], "with": {}},
                "metric": metric,
                "scale": "log",
                "ticks": result[0],
                "frequencies": result[1],
                "interquartile": result[2],
            },
        ]


_gg = "github.com/src-d/go-git"


@pytest.mark.parametrize(
    "metric, date_to, bins, scale, ticks, quantiles, account, repo, status",
    [
        (PullRequestMetricID.PR_OPENED, "2020-01-23", 10, "log", None, [0, 1], 1, _gg, 400),
        (PullRequestMetricID.PR_CYCLE_TIME, "2020-01-23", -1, "log", None, [0, 1], 1, _gg, 400),
        (PullRequestMetricID.PR_CYCLE_TIME, "2020-01-23", 10, "xxx", None, [0, 1], 1, _gg, 400),
        (PullRequestMetricID.PR_CYCLE_TIME, "2015-01-23", 10, "linear", None, [0, 1], 1, _gg, 400),
        (PullRequestMetricID.PR_CYCLE_TIME, "2020-01-23", 10, "linear", None, [0, 1], 2, _gg, 422),
        (PullRequestMetricID.PR_CYCLE_TIME, "2020-01-23", 10, "linear", None, [0, 1], 4, _gg, 404),
        (
            PullRequestMetricID.PR_CYCLE_TIME,
            "2015-11-23",
            10,
            "linear",
            None,
            [-1, 1],
            1,
            _gg,
            400,
        ),
        (PullRequestMetricID.PR_CYCLE_TIME, "2015-11-23", None, None, None, [0, 1], 1, _gg, 200),
        (PullRequestMetricID.PR_CYCLE_TIME, "2015-11-23", None, None, [], [0, 1], 1, _gg, 400),
        ("xxx", "2015-11-23", None, None, None, [0, 1], 1, _gg, 400),
        (
            PullRequestMetricID.PR_CYCLE_TIME,
            "2015-11-23",
            None,
            None,
            None,
            [0, 1],
            1,
            "github.com/athenianco/athenian-api",
            403,
        ),
    ],
)
async def test_calc_histogram_prs_nasty_input(
    client,
    headers,
    metric,
    date_to,
    bins,
    scale,
    ticks,
    quantiles,
    account,
    repo,
    status,
):
    body = {
        "for": [
            {
                "with": {},
                "repositories": [repo],
            },
        ],
        "histograms": [
            {
                "metric": metric,
                **({"scale": scale} if scale is not None else {}),
                **({"bins": bins} if bins is not None else {}),
                **({"ticks": ticks} if ticks is not None else {}),
            },
        ],
        "date_from": "2015-10-13",
        "date_to": date_to,
        "quantiles": quantiles,
        "exclude_inactive": False,
        "account": account,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == status, "Response body is : " + body


async def test_calc_histogram_prs_no_histograms(client, headers):
    body = {
        "for": [
            {
                "with": {},
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
        ],
        "date_from": "2015-10-13",
        "date_to": "2015-11-23",
        "quantiles": [0, 1],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 400, "Response body is : " + body


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
        "histograms": [
            {
                "metric": PullRequestMetricID.PR_RELEASE_TIME,
                "scale": "linear",
                "bins": 10,
            },
            {
                "metric": PullRequestMetricID.PR_REVIEW_TIME,
                "scale": "linear",
                "bins": 10,
            },
        ],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = FriendlyJson.loads(body)
    assert len(body) == 4
    for i in range(4):
        for j in range(i + 1, 4):
            assert body[i] != body[j], "%d == %d" % (i, j)


async def test_calc_histogram_prs_team(client, headers, sample_team):
    team_str = "{%d}" % sample_team
    body = {
        "for": [
            {
                "with": {"merger": [team_str]},
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
        ],
        "histograms": [
            {
                "metric": PullRequestMetricID.PR_RELEASE_TIME,
                "scale": "linear",
                "bins": 10,
            },
        ],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = FriendlyJson.loads(body)
    assert body == [
        {
            "for": {
                "with": {"merger": [team_str]},
                "repositories": ["github.com/src-d/go-git"],
            },
            "metric": PullRequestMetricID.PR_RELEASE_TIME,
            "scale": "linear",
            "ticks": [
                "60s",
                "259254s",
                "518448s",
                "777642s",
                "1036836s",
                "1296030s",
                "1555224s",
                "1814418s",
                "2073612s",
                "2332806s",
                "2592000s",
            ],
            "frequencies": [91, 33, 25, 21, 8, 14, 21, 9, 20, 146],
            "interquartile": {"left": "283986s", "right": "2592000s"},
        },
    ]


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
        "histograms": [
            {
                "metric": PullRequestMetricID.PR_SIZE,
                "scale": "linear",
                "bins": 10,
            },
        ],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = FriendlyJson.loads(body)
    assert body == [
        {
            "for": {
                "repositories": ["github.com/src-d/go-git"],
                "with": {"merger": ["github.com/mcuadros"]},
            },
            "metric": "pr-size",
            "scale": "linear",
            "ticks": [
                0.0,
                1109.9,
                2219.8,
                3329.7000000000003,
                4439.6,
                5549.5,
                6659.400000000001,
                7769.300000000001,
                8879.2,
                9989.1,
                11099.0,
            ],
            "frequencies": [465, 17, 2, 2, 1, 0, 1, 0, 0, 1],
            "interquartile": {"left": 18.0, "right": 188.0},
        },
    ]


async def test_calc_histogram_prs_ticks(client, headers):
    body = {
        "for": [
            {
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
        ],
        "histograms": [
            {
                "metric": PullRequestMetricID.PR_RELEASE_TIME,
                "ticks": ["10000s", "100000s", "10000000000000s"],
            },
        ],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/pull_requests", headers=headers, json=body,
    )
    result = (await response.read()).decode("utf-8")
    assert response.status == 400, "Response body is : " + result
    body["histograms"][0]["ticks"] = body["histograms"][0]["ticks"][:-1]
    response = await client.request(
        method="POST", path="/v1/histograms/pull_requests", headers=headers, json=body,
    )
    result = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + result
    result = FriendlyJson.loads(result)
    assert result == [
        {
            "for": {"repositories": ["github.com/src-d/go-git"]},
            "metric": "pr-release-time",
            "scale": "linear",
            "ticks": ["60s", "10000s", "100000s", "2592000s"],
            "frequencies": [39, 29, 346],
            "interquartile": {"left": "322517s", "right": "2592000s"},
        },
    ]
    body["histograms"][0]["ticks"].append(f"{2 * 365 * 24 * 3600}s")
    response = await client.request(
        method="POST", path="/v1/histograms/pull_requests", headers=headers, json=body,
    )
    result = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + result


async def test_calc_histogram_prs_groups(client, headers):
    body = {
        "for": [
            {
                "repositories": ["{1}"],
                "repogroups": [[0], [0]],
            },
        ],
        "histograms": [
            {
                "metric": PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME,
                "scale": "log",
            },
        ],
        "date_from": "2017-10-13",
        "date_to": "2018-01-23",
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = FriendlyJson.loads(body)
    cprh = []
    for item in body:
        cprh.append(CalculatedPullRequestHistogram.from_dict(item))

    assert len(cprh) == 2
    for h in body:
        assert h == {
            "for": {"repositories": ["{1}"]},
            "metric": "pr-wait-first-review-time",
            "scale": "log",
            "ticks": [
                "60s",
                "184s",
                "569s",
                "1752s",
                "5398s",
                "16624s",
                "51201s",
                "157688s",
                "485648s",
            ],
            "frequencies": [6, 5, 8, 5, 4, 14, 10, 2],
            "interquartile": {"left": "790s", "right": "45167s"},
        }


async def test_calc_histogram_prs_lines(client, headers):
    body = {
        "for": [
            {
                "repositories": ["{1}"],
                "lines": [0, 100, 100500],
            },
        ],
        "histograms": [
            {
                "metric": PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME,
                "scale": "log",
            },
        ],
        "date_from": "2017-10-13",
        "date_to": "2018-05-23",
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = FriendlyJson.loads(body)
    cprh = []
    for item in body:
        cprh.append(CalculatedPullRequestHistogram.from_dict(item))

    assert len(cprh) == 2
    assert body[0] == {
        "for": {"repositories": ["{1}"], "lines": [0, 100]},
        "metric": "pr-wait-first-review-time",
        "scale": "log",
        "ticks": [
            "60s",
            "195s",
            "637s",
            "2078s",
            "6776s",
            "22090s",
            "72014s",
            "234763s",
            "765318s",
            "2494902s",
        ],
        "frequencies": [4, 8, 7, 8, 4, 22, 8, 6, 1],
        "interquartile": {"left": "1762s", "right": "62368s"},
    }

    assert body[1] == {
        "for": {"repositories": ["{1}"], "lines": [100, 100500]},
        "metric": "pr-wait-first-review-time",
        "scale": "log",
        "ticks": ["60s", "273s", "1243s", "5661s", "25774s", "117343s", "534220s"],
        "frequencies": [8, 4, 1, 4, 7, 2],
        "interquartile": {"left": "60s", "right": "49999s"},
    }


@pytest.mark.parametrize("envs", [{}, {"environments": []}, {"environments": ["staging", "prod"]}])
async def test_calc_histogram_prs_deployments_bad_envs(client, headers, envs):
    body = {
        "for": [
            {
                "repositories": ["{1}"],
                **envs,
            },
        ],
        "histograms": [
            {
                "metric": PullRequestMetricID.PR_DEPLOYMENT_TIME,
                "scale": "log",
            },
        ],
        "date_from": "2017-10-13",
        "date_to": "2018-05-23",
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 400, "Response body is : " + body


async def test_calc_histogram_prs_deployment_time(client, headers, precomputed_deployments):
    body = {
        "for": [
            {
                "repositories": ["{1}"],
                "environments": ["production"],
            },
        ],
        "histograms": [
            {
                "metric": PullRequestMetricID.PR_DEPLOYMENT_TIME,
                "scale": "log",
            },
        ],
        "date_from": "2015-10-13",
        "date_to": "2020-05-23",
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = FriendlyJson.loads(body)
    CalculatedPullRequestHistogram.from_dict(body[0])
    assert body[0] == {
        "for": {"repositories": ["{1}"], "environments": ["production"]},
        "metric": "pr-deployment-time",
        "scale": "log",
        "ticks": ["1572127s", "4273485s"],
        "frequencies": [418],
        "interquartile": {"left": "2592000s", "right": "2592000s"},
    }


async def test_calc_histogram_prs_logical(
    client,
    headers,
    logical_settings_db,
    release_match_setting_tag_logical_db,
):
    body = {
        "for": [
            {
                "repositories": ["github.com/src-d/go-git/alpha", "github.com/src-d/go-git/beta"],
            },
        ],
        "histograms": [
            {
                "metric": PullRequestMetricID.PR_REVIEW_TIME,
                "scale": "log",
            },
        ],
        "date_from": "2015-10-13",
        "date_to": "2020-05-23",
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = FriendlyJson.loads(body)
    CalculatedPullRequestHistogram.from_dict(body[0])
    assert body[0] == {
        "for": {"repositories": ["github.com/src-d/go-git/alpha", "github.com/src-d/go-git/beta"]},
        "metric": "pr-review-time",
        "scale": "log",
        "ticks": [
            "60s",
            "146s",
            "355s",
            "865s",
            "2105s",
            "5123s",
            "12470s",
            "30351s",
            "73870s",
            "179789s",
            "437576s",
            "1064987s",
            "2591999s",
        ],
        "frequencies": [17, 4, 12, 8, 5, 18, 11, 32, 34, 28, 15, 20],
        "interquartile": {"left": "7297s", "right": "290319s"},
    }


async def test_calc_histogram_code_checks_smoke(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "histograms": [
            {
                "metric": CodeCheckMetricID.SUITES_PER_PR,
                "ticks": [0, 1, 2, 3, 4, 5, 10, 50],
            },
        ],
        "for": [
            {
                "repositories": ["github.com/src-d/go-git"],
            },
        ],
    }
    response = await client.request(
        method="POST", path="/v1/histograms/code_checks", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    rbody = FriendlyJson.loads(rbody)
    assert rbody == [
        {
            "metric": "chk-suites-per-pr",
            "scale": "linear",
            "ticks": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 50.0, 73.0],
            "frequencies": [0, 46, 149, 9, 16, 21, 5, 1],
            "interquartile": {"left": 2.0, "right": 2.0},
            "for": {"repositories": ["github.com/src-d/go-git"]},
        },
    ]


async def test_calc_histogram_code_checks_labels_jira(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "histograms": [
            {
                "metric": CodeCheckMetricID.SUITES_PER_PR,
                "ticks": [0, 1, 2, 3, 4, 5, 10, 50],
            },
        ],
        "for": [
            {
                "repositories": ["github.com/src-d/go-git"],
                "labels_include": ["bug", "plumbing", "enhancement"],
            },
            {
                "repositories": ["github.com/src-d/go-git"],
                "jira": {
                    "labels_include": ["bug", "onboarding", "performance"],
                },
            },
        ],
    }
    response = await client.request(
        method="POST", path="/v1/histograms/code_checks", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    rbody = FriendlyJson.loads(rbody)
    rbody.sort(key=lambda x: x["for"].get("labels_include", []))
    assert rbody == [
        {
            "metric": "chk-suites-per-pr",
            "scale": "linear",
            "ticks": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 50.0],
            "frequencies": [0, 10, 4, 0, 1, 4, 1],
            "interquartile": {"left": 1.0, "right": 4.25},
            "for": {
                "repositories": ["github.com/src-d/go-git"],
                "jira": {"labels_include": ["bug", "onboarding", "performance"]},
            },
        },
        {
            "metric": "chk-suites-per-pr",
            "scale": "linear",
            "ticks": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 50.0],
            "frequencies": [0, 3, 5, 0, 1, 1, 0],
            "interquartile": {"left": 1.25, "right": 2.0},
            "for": {
                "repositories": ["github.com/src-d/go-git"],
                "labels_include": ["bug", "plumbing", "enhancement"],
            },
        },
    ]


async def test_calc_histogram_code_checks_team(client, headers, sample_team):
    team_str = "{%d}" % sample_team
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "histograms": [
            {
                "metric": CodeCheckMetricID.SUITES_PER_PR,
                "ticks": [0, 1, 2, 3, 4, 5, 10, 50],
            },
        ],
        "for": [
            {
                "repositories": ["github.com/src-d/go-git"],
                "pushers": [team_str],
            },
        ],
    }
    response = await client.request(
        method="POST", path="/v1/histograms/code_checks", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    rbody = FriendlyJson.loads(rbody)
    assert rbody == [
        {
            "metric": "chk-suites-per-pr",
            "scale": "linear",
            "ticks": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 50.0, 68.0],
            "frequencies": [0, 3, 16, 0, 1, 3, 0, 1],
            "interquartile": {"left": 2.0, "right": 2.0},
            "for": {"repositories": ["github.com/src-d/go-git"], "pushers": [team_str]},
        },
    ]


@pytest.mark.parametrize(
    "metric, account, repo, status",
    [
        ("xxx", 1, _gg, 400),
        (CodeCheckMetricID.SUITES_COUNT, 1, _gg, 400),
        (CodeCheckMetricID.SUITES_PER_PR, 2, _gg, 422),
        (CodeCheckMetricID.SUITES_PER_PR, 3, _gg, 404),
        (CodeCheckMetricID.SUITES_PER_PR, 1, "github.com/athenianco/athenian-api", 403),
    ],
)
async def test_calc_histogram_code_checks_nasty_input(
    client,
    headers,
    metric,
    account,
    repo,
    status,
):
    body = {
        "account": account,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "histograms": [
            {
                "metric": metric,
            },
        ],
        "for": [
            {
                "repositories": [repo],
            },
        ],
    }
    response = await client.request(
        method="POST", path="/v1/histograms/code_checks", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == status, "Response body is : " + rbody


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_histogram_code_checks_split(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "histograms": [
            {
                "metric": CodeCheckMetricID.SUITES_PER_PR,
                "ticks": [0, 1, 2, 3, 4, 5, 10, 50],
            },
        ],
        "for": [
            {
                "repositories": ["github.com/src-d/go-git"],
            },
        ],
        "split_by_check_runs": True,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/code_checks", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    rbody = FriendlyJson.loads(rbody)
    assert rbody == [
        {
            "metric": "chk-suites-per-pr",
            "scale": "linear",
            "ticks": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 50.0],
            "frequencies": [0, 1, 122, 5, 10, 12, 2],
            "interquartile": {"left": 2.0, "right": 2.0},
            "for": {"repositories": ["github.com/src-d/go-git"]},
            "check_runs": 1,
            "suites_ratio": 0.35198372329603256,
        },
        {
            "metric": "chk-suites-per-pr",
            "scale": "linear",
            "ticks": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 50.0, 73.0],
            "frequencies": [0, 3, 10, 1, 1, 3, 1, 1],
            "interquartile": {"left": 2.0, "right": 4.25},
            "for": {"repositories": ["github.com/src-d/go-git"]},
            "check_runs": 2,
            "suites_ratio": 0.1861648016276704,
        },
        {
            "metric": "chk-suites-per-pr",
            "scale": "linear",
            "ticks": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 50.0],
            "frequencies": [0, 15, 12, 0, 4, 3, 0],
            "interquartile": {"left": 1.0, "right": 2.0},
            "for": {"repositories": ["github.com/src-d/go-git"]},
            "check_runs": 3,
            "suites_ratio": 0.35096642929806715,
        },
        {
            "metric": "chk-suites-per-pr",
            "scale": "linear",
            "ticks": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 50.0],
            "frequencies": [0, 27, 5, 3, 1, 3, 1],
            "interquartile": {"left": 1.0, "right": 2.0},
            "for": {"repositories": ["github.com/src-d/go-git"]},
            "check_runs": 4,
            "suites_ratio": 0.09766022380467955,
        },
        {
            "metric": "chk-suites-per-pr",
            "scale": "linear",
            "ticks": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 50.0],
            "frequencies": [0, 0, 0, 0, 0, 0, 0],
            "interquartile": {"left": 0, "right": 0},
            "for": {"repositories": ["github.com/src-d/go-git"]},
            "check_runs": 5,
            "suites_ratio": 0.004069175991861648,
        },
        {
            "metric": "chk-suites-per-pr",
            "scale": "linear",
            "ticks": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 50.0],
            "frequencies": [0, 0, 0, 0, 0, 0, 1],
            "interquartile": {"left": 13.0, "right": 13.0},
            "for": {"repositories": ["github.com/src-d/go-git"]},
            "check_runs": 6,
            "suites_ratio": 0.009155645981688708,
        },
    ]


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_calc_histogram_code_checks_elapsed_time_per_concurrency(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "histograms": [
            {
                "metric": CodeCheckMetricID.ELAPSED_TIME_PER_CONCURRENCY,
            },
        ],
        "for": [
            {
                "repositories": ["github.com/src-d/go-git"],
            },
        ],
    }
    response = await client.request(
        method="POST", path="/v1/histograms/code_checks", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    rbody = FriendlyJson.loads(rbody)
    assert rbody == [
        {
            "metric": "chk-elapsed-time-per-concurrency",
            "scale": "linear",
            "ticks": [0, 2],
            "frequencies": ["223s"],
            "interquartile": {"left": 1.0, "right": 1.0},
            "for": {"repositories": ["github.com/src-d/go-git"]},
        },
    ]
