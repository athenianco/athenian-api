from datetime import date
import json

import pytest
from sqlalchemy import delete, insert

from athenian.api.internal.miners.github import developer
from athenian.api.models.persistentdata.models import DeployedComponent, DeployedLabel
from athenian.api.models.web import (
    CalculatedCodeCheckMetrics,
    CalculatedDeploymentMetric,
    CalculatedDeveloperMetrics,
    CalculatedLinearMetricValues,
    CodeBypassingPRsMeasurement,
    CodeCheckMetricID,
    DeploymentMetricID,
    DeveloperMetricID,
)
from athenian.api.serialization import FriendlyJson


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


@pytest.mark.parametrize(
    "account, date_to, in_, code",
    [
        (3, "2020-02-22", "{1}", 404),
        (2, "2020-02-22", "github.com/src-d/go-git", 422),
        (10, "2020-02-22", "{1}", 404),
        (1, "2019-01-12", "{1}", 200),
        (1, "2019-01-11", "{1}", 400),
        (1, "2019-01-32", "{1}", 400),
        (1, "2019-01-12", "github.com/athenianco/athenian-api", 403),
    ],
)
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
    "dev-releases": 20,
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


@pytest.mark.parametrize(
    "metric, value",
    sorted((m, developer_metric_mcuadros_stats[m]) for m in DeveloperMetricID),
)
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
        FriendlyJson.loads((await response.read()).decode("utf-8")),
    )
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
            {
                "repositories": ["{1}"],
                "developers": ["github.com/mcuadros"],
                "jira": {
                    "labels_include": [
                        "API",
                        "Webapp",
                        "accounts",
                        "bug",
                        "code-quality",
                        "discarded",
                        "discussion",
                        "feature",
                        "functionality",
                        "internal-story",
                        "needs-specs",
                        "onboarding",
                        "performance",
                        "user-story",
                        "webapp",
                    ],
                },
            },
        ],
        "metrics": [metric],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")),
    )
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


@pytest.mark.parametrize(
    "metric, value",
    list(developer_metric_mcuadros_jira_labels_stats.items()),
)
async def test_developer_metrics_jira_labels_single(client, headers, metric, value):
    body = {
        "account": 1,
        "date_from": "2016-01-12",
        "date_to": "2020-03-01",
        "granularities": ["all"],
        "for": [
            {
                "repositories": ["{1}"],
                "developers": ["github.com/mcuadros"],
                "labels_include": [
                    "enhancement",
                    "bug",
                    "plumbing",
                    "performance",
                    "ssh",
                    "documentation",
                    "windows",
                ],
                "jira": {
                    "labels_include": [
                        "API",
                        "Webapp",
                        "accounts",
                        "bug",
                        "code-quality",
                        "discarded",
                        "discussion",
                        "feature",
                        "functionality",
                        "internal-story",
                        "needs-specs",
                        "onboarding",
                        "performance",
                        "user-story",
                        "webapp",
                    ],
                },
            },
        ],
        "metrics": [metric],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")),
    )
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
        FriendlyJson.loads((await response.read()).decode("utf-8")),
    )
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
        assert all(v == 0 for v in result.calculated[0].values[0][0].values), "%s\n%s" % (
            str(result.calculated[0].values[0]),
            sorted(DeveloperMetricID),
        )
    else:
        assert all(
            isinstance(v, int) for v in result.calculated[0].values[0][0].values
        ), "%s\n%s" % (str(result.calculated[0].values[0]), sorted(DeveloperMetricID))


async def test_developer_metrics_repogroups(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "timezone": 60,
        "granularities": ["all"],
        "for": [
            {
                "repositories": ["github.com/src-d/go-git", "github.com/src-d/gitbase"],
                "repogroups": [[0], [1]],
                "developers": ["github.com/mcuadros"],
            },
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
        FriendlyJson.loads((await response.read()).decode("utf-8")),
    )
    assert set(result.metrics) == set(DeveloperMetricID)
    assert len(result.calculated) == 2
    assert all(
        (v > 0 or m == DeveloperMetricID.ACTIVE)
        for m, v in zip(sorted(DeveloperMetricID), result.calculated[0].values[0][0].values)
    )
    assert all(v == 0 for v in result.calculated[1].values[0][0].values)


@pytest.mark.parametrize(
    "metric, value",
    sorted((m, developer_metric_be_stats[m]) for m in DeveloperMetricID),
)
async def test_developer_metrics_labels_include(client, headers, metric, value):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "granularities": ["all"],
        "for": [
            {
                "repositories": ["{1}"],
                "developers": [
                    "github.com/mcuadros",
                    "github.com/smola",
                    "github.com/jfontan",
                    "github.com/ajnavarro",
                ],
                "labels_include": ["bug", "enhancement"],
            },
        ],
        "metrics": [metric],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")),
    )
    assert result.calculated[0].for_.labels_include == ["bug", "enhancement"]
    assert [m[0].values for m in result.calculated[0].values] == value


async def test_developer_metrics_aggregate(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-07-12",
        "date_to": "2018-09-15",
        "granularities": ["all"],
        "for": [
            {
                "repositories": ["{1}"],
                "developers": [
                    "github.com/mcuadros",
                    "github.com/erizocosmico",
                    "github.com/jfontan",
                    "github.com/vancluever",
                ],
                "aggregate_devgroups": [[0, 1, 2, 3]],
            },
        ],
        "metrics": [DeveloperMetricID.ACTIVE, DeveloperMetricID.PRS_CREATED],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")),
    )
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
        "for": [
            {
                "repositories": ["{1}"],
                "developers": [
                    "github.com/mcuadros",
                    "github.com/erizocosmico",
                    "github.com/jfontan",
                ],
            },
        ],
        "metrics": [DeveloperMetricID.ACTIVE, DeveloperMetricID.PRS_CREATED],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")),
    )
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
        ],
        [
            CalculatedLinearMetricValues(date=date(2018, 7, 12), values=[0, 2]),
            CalculatedLinearMetricValues(date=date(2018, 7, 26), values=[1, 2]),
            CalculatedLinearMetricValues(date=date(2018, 8, 9), values=[0, 0]),
            CalculatedLinearMetricValues(date=date(2018, 8, 23), values=[0, 0]),
            CalculatedLinearMetricValues(date=date(2018, 9, 6), values=[0, 0]),
        ],
        [
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
        "for": [
            {
                "repositories": ["{1}"],
                "developers": [
                    "github.com/mcuadros",
                    "github.com/smola",
                    "github.com/jfontan",
                    "github.com/ajnavarro",
                ],
                "labels_exclude": ["bug", "enhancement"],
            },
        ],
        "metrics": [DeveloperMetricID.PRS_CREATED],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")),
    )
    assert not result.calculated[0].for_.labels_include
    assert result.calculated[0].for_.labels_exclude == ["bug", "enhancement"]
    assert [m[0].values for m in result.calculated[0].values] == [[14], [8], [26], [7]]


async def test_developer_metrics_labels_include_exclude(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "granularities": ["all"],
        "for": [
            {
                "repositories": ["{1}"],
                "developers": [
                    "github.com/mcuadros",
                    "github.com/smola",
                    "github.com/jfontan",
                    "github.com/ajnavarro",
                ],
                "labels_include": ["bug"],
                "labels_exclude": ["enhancement"],
            },
        ],
        "metrics": [DeveloperMetricID.PRS_CREATED],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")),
    )
    assert result.calculated[0].for_.labels_include == ["bug"]
    assert result.calculated[0].for_.labels_exclude == ["enhancement"]
    assert [m[0].values for m in result.calculated[0].values] == [[0], [0], [1], [0]]


async def test_developer_metrics_labels_contradiction(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "granularities": ["all"],
        "for": [
            {
                "repositories": ["{1}"],
                "developers": [
                    "github.com/mcuadros",
                    "github.com/smola",
                    "github.com/jfontan",
                    "github.com/ajnavarro",
                ],
                "labels_include": ["bug"],
                "labels_exclude": ["bug"],
            },
        ],
        "metrics": [DeveloperMetricID.PRS_CREATED],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")),
    )
    assert result.calculated[0].for_.labels_include == ["bug"]
    assert result.calculated[0].for_.labels_exclude == ["bug"]
    assert [m[0].values for m in result.calculated[0].values] == [[0], [0], [0], [0]]


@pytest.mark.parametrize(
    "account, date_to, in_, code",
    [
        (3, "2020-02-22", "{1}", 404),
        (2, "2020-02-22", "github.com/src-d/go-git", 422),
        (10, "2020-02-22", "{1}", 404),
        (1, "2018-01-12", "{1}", 200),
        (1, "2018-01-11", "{1}", 400),
        (1, "2019-01-32", "{1}", 400),
        (1, "2018-01-12", "github.com/athenianco/athenian-api", 403),
    ],
)
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


@pytest.mark.parametrize(
    "repos, devs",
    [
        ([], ["github.com/mcuadros"]),
        (["github.com/src-d/go-git"], []),
    ],
)
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
            {"repositories": ["{1}"], "developers": ["github.com/mcuadros", "github.com/smola"]},
        ],
        "metrics": [DeveloperMetricID.PRS_CREATED],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")),
    )
    assert result.calculated[0].for_.developers == body["for"][0]["developers"]
    assert [m[0].values for m in result.calculated[0].values] == [[14], [8]]
    body["for"][0]["developers"] = list(reversed(body["for"][0]["developers"]))
    response = await client.request(
        method="POST", path="/v1/metrics/developers", headers=headers, json=body,
    )
    assert response.status == 200
    result = CalculatedDeveloperMetrics.from_dict(
        FriendlyJson.loads((await response.read()).decode("utf-8")),
    )
    assert result.calculated[0].for_.developers == body["for"][0]["developers"]
    assert [m[0].values for m in result.calculated[0].values] == [[8], [14]]


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_smoke(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [
            {
                "repositories": ["github.com/src-d/go-git"],
                "pushers": ["github.com/mcuadros"],
            },
        ],
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
            {
                "for": {
                    "pushers": ["github.com/mcuadros"],
                    "repositories": ["github.com/src-d/go-git"],
                },
                "granularity": "all",
                "values": [{"date": date(2018, 1, 12), "values": [245]}],
            },
        ],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_jira(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [
            {
                "repositories": ["github.com/src-d/go-git"],
                "jira": {
                    "labels_include": ["bug", "onboarding", "performance"],
                },
            },
        ],
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
            {
                "for": {
                    "jira": {"labels_include": ["bug", "onboarding", "performance"]},
                    "repositories": ["github.com/src-d/go-git"],
                },
                "granularity": "all",
                "values": [{"date": date(2018, 1, 12), "values": [60]}],
            },
        ],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_labels(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [
            {
                "repositories": ["github.com/src-d/go-git"],
                "labels_include": ["bug", "plumbing", "enhancement"],
            },
            {
                "repositories": ["github.com/src-d/go-git"],
                "labels_include": ["bug", "plumbing", "enhancement"],
                "labels_exclude": ["xxx"],
            },
            {
                "repositories": ["github.com/src-d/go-git"],
                "labels_exclude": ["bug"],
            },
        ],
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
    md["calculated"].sort(
        key=lambda x: (x["for"].get("labels_include", []), x["for"].get("labels_exclude", [])),
    )
    assert md == {
        "calculated": [
            {
                "for": {"labels_exclude": ["bug"], "repositories": ["github.com/src-d/go-git"]},
                "granularity": "all",
                "values": [{"date": date(2018, 1, 12), "values": [734]}],
            },
            {
                "for": {
                    "labels_include": ["bug", "plumbing", "enhancement"],
                    "repositories": ["github.com/src-d/go-git"],
                },
                "granularity": "all",
                "values": [{"date": date(2018, 1, 12), "values": [23]}],
            },
            {
                "for": {
                    "labels_include": ["bug", "plumbing", "enhancement"],
                    "labels_exclude": ["xxx"],
                    "repositories": ["github.com/src-d/go-git"],
                },
                "granularity": "all",
                "values": [{"date": date(2018, 1, 12), "values": [18]}],
            },
        ],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_split_by_check_runs(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [
            {
                "repositories": ["github.com/src-d/go-git"],
            },
        ],
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
        "calculated": [
            {
                "check_runs": 1,
                "for": {"repositories": ["github.com/src-d/go-git"]},
                "granularity": "all",
                "suites_ratio": 0.35198372329603256,
                "values": [{"date": date(2018, 1, 12), "values": [344]}],
            },
            {
                "check_runs": 2,
                "for": {"repositories": ["github.com/src-d/go-git"]},
                "granularity": "all",
                "suites_ratio": 0.1861648016276704,
                "values": [{"date": date(2018, 1, 12), "values": [183]}],
            },
            {
                "check_runs": 3,
                "for": {"repositories": ["github.com/src-d/go-git"]},
                "granularity": "all",
                "suites_ratio": 0.35096642929806715,
                "values": [{"date": date(2018, 1, 12), "values": [345]}],
            },
            {
                "check_runs": 4,
                "for": {"repositories": ["github.com/src-d/go-git"]},
                "granularity": "all",
                "suites_ratio": 0.09766022380467955,
                "values": [{"date": date(2018, 1, 12), "values": [96]}],
            },
            {
                "check_runs": 5,
                "for": {"repositories": ["github.com/src-d/go-git"]},
                "granularity": "all",
                "suites_ratio": 0.004069175991861648,
                "values": [{"date": date(2018, 1, 12), "values": [4]}],
            },
            {
                "check_runs": 6,
                "for": {"repositories": ["github.com/src-d/go-git"]},
                "granularity": "all",
                "suites_ratio": 0.009155645981688708,
                "values": [{"date": date(2018, 1, 12), "values": [9]}],
            },
        ],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
        "split_by_check_runs": True,
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_repogroups(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [
            {
                "repositories": ["github.com/src-d/go-git"],
                "repogroups": [[0], [0]],
            },
        ],
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
            {
                "for": {"repositories": ["github.com/src-d/go-git"]},
                "granularity": "all",
                "values": [{"date": date(2018, 1, 12), "values": [981]}],
            },
            {
                "for": {"repositories": ["github.com/src-d/go-git"]},
                "granularity": "all",
                "values": [{"date": date(2018, 1, 12), "values": [981]}],
            },
        ],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_authorgroups(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [
            {
                "repositories": ["github.com/src-d/go-git"],
                "pusher_groups": [["github.com/mcuadros"], ["github.com/erizocosmico"]],
            },
        ],
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
            {
                "for": {
                    "pushers": ["github.com/mcuadros"],
                    "repositories": ["github.com/src-d/go-git"],
                },
                "granularity": "all",
                "values": [{"date": date(2018, 1, 12), "values": [245]}],
            },
            {
                "for": {
                    "pushers": ["github.com/erizocosmico"],
                    "repositories": ["github.com/src-d/go-git"],
                },
                "granularity": "all",
                "values": [{"date": date(2018, 1, 12), "values": [22]}],
            },
        ],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_code_check_metrics_lines(client, headers):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [
            {
                "repositories": ["github.com/src-d/go-git"],
                "lines": [0, 10, 1000],
            },
        ],
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
            {
                "for": {"lines": [0, 10], "repositories": ["github.com/src-d/go-git"]},
                "granularity": "all",
                "values": [{"date": date(2018, 1, 12), "values": [299]}],
            },
            {
                "for": {"lines": [10, 1000], "repositories": ["github.com/src-d/go-git"]},
                "granularity": "all",
                "values": [{"date": date(2018, 1, 12), "values": [666]}],
            },
        ],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
    }


@pytest.mark.parametrize(
    "account, repos, metrics, code",
    [
        (3, ["{1}"], [CodeCheckMetricID.SUITES_COUNT], 404),
        (2, ["github.com/src-d/go-git"], [CodeCheckMetricID.SUITES_COUNT], 422),
        (10, ["{1}"], [CodeCheckMetricID.SUITES_COUNT], 404),
        (1, None, [CodeCheckMetricID.SUITES_COUNT], 400),
        (1, ["{1}"], None, 400),
        (1, ["{1}"], ["xxx"], 400),
        (1, ["github.com/athenianco/athenian-api"], [CodeCheckMetricID.SUITES_COUNT], 403),
    ],
)
async def test_code_check_metrics_nasty_input(client, headers, account, repos, metrics, code):
    body = {
        "account": account,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [
            {
                "repositories": repos,
            },
        ],
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
        "for": [
            {
                "repositories": ["github.com/src-d/go-git/alpha"],
                "pushers": ["github.com/mcuadros"],
            },
        ],
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
            {
                "for": {
                    "pushers": ["github.com/mcuadros"],
                    "repositories": ["github.com/src-d/go-git/alpha"],
                },
                "granularity": "all",
                "values": [{"date": date(2018, 1, 12), "values": [91]}],
            },
        ],
        "date_from": date(2018, 1, 12),
        "date_to": date(2020, 3, 1),
        "granularities": ["all"],
        "metrics": ["chk-suites-count"],
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_deployment_metrics_smoke(client, headers, sample_deployments):
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [
            {
                "repositories": ["{1}"],
                "withgroups": [
                    {"releaser": ["github.com/mcuadros"]},
                    {"pr_author": ["github.com/mcuadros"]},
                ],
                "environments": ["staging", "production", "mirror"],
            },
        ],
        "metrics": [
            DeploymentMetricID.DEP_SUCCESS_COUNT,
            DeploymentMetricID.DEP_DURATION_SUCCESSFUL,
        ],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/deployments", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = [CalculatedDeploymentMetric.from_dict(obj) for obj in json.loads(body)]
    assert [m.to_dict() for m in model] == [
        {
            "for": {
                "repositories": ["{1}"],
                "with": {"releaser": ["github.com/mcuadros"]},
                "environments": ["staging"],
            },
            "metrics": ["dep-success-count", "dep-duration-successful"],
            "granularity": "all",
            "values": [
                {
                    "date": date(2018, 1, 12),
                    "values": [3, "600s"],
                    "confidence_maxs": [None, "600s"],
                    "confidence_mins": [None, "600s"],
                    "confidence_scores": [None, 100],
                },
            ],
        },
        {
            "for": {
                "repositories": ["{1}"],
                "with": {"releaser": ["github.com/mcuadros"]},
                "environments": ["production"],
            },
            "metrics": ["dep-success-count", "dep-duration-successful"],
            "granularity": "all",
            "values": [
                {
                    "date": date(2018, 1, 12),
                    "values": [3, "600s"],
                    "confidence_maxs": [None, "600s"],
                    "confidence_mins": [None, "600s"],
                    "confidence_scores": [None, 100],
                },
            ],
        },
        {
            "for": {
                "repositories": ["{1}"],
                "with": {"releaser": ["github.com/mcuadros"]},
                "environments": ["mirror"],
            },
            "metrics": ["dep-success-count", "dep-duration-successful"],
            "granularity": "all",
            "values": [
                {
                    "date": date(2018, 1, 12),
                    "values": [0, None],
                },
            ],
        },
        {
            "for": {
                "repositories": ["{1}"],
                "with": {"pr_author": ["github.com/mcuadros"]},
                "environments": ["staging"],
            },
            "metrics": ["dep-success-count", "dep-duration-successful"],
            "granularity": "all",
            "values": [
                {
                    "date": date(2018, 1, 12),
                    "values": [3, "600s"],
                    "confidence_maxs": [None, "600s"],
                    "confidence_mins": [None, "600s"],
                    "confidence_scores": [None, 100],
                },
            ],
        },
        {
            "for": {
                "repositories": ["{1}"],
                "with": {"pr_author": ["github.com/mcuadros"]},
                "environments": ["production"],
            },
            "metrics": ["dep-success-count", "dep-duration-successful"],
            "granularity": "all",
            "values": [
                {
                    "date": date(2018, 1, 12),
                    "values": [3, "600s"],
                    "confidence_maxs": [None, "600s"],
                    "confidence_mins": [None, "600s"],
                    "confidence_scores": [None, 100],
                },
            ],
        },
        {
            "for": {
                "repositories": ["{1}"],
                "with": {"pr_author": ["github.com/mcuadros"]},
                "environments": ["mirror"],
            },
            "metrics": ["dep-success-count", "dep-duration-successful"],
            "granularity": "all",
            "values": [
                {
                    "date": date(2018, 1, 12),
                    "values": [0, None],
                },
            ],
        },
    ]


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
    assert [m.to_dict() for m in model] == [
        {
            "for": {},
            "granularity": "all",
            "metrics": [DeploymentMetricID.DEP_SUCCESS_COUNT],
            "values": [{"date": date(2018, 1, 12), "values": [9]}],
        },
    ]


@pytest.mark.parametrize(
    "account, date_from, date_to, repos, withgroups, metrics, code",
    [
        (1, "2018-01-12", "2020-01-12", ["{1}"], [], [DeploymentMetricID.DEP_PRS_COUNT], 200),
        (1, "2020-01-12", "2018-01-12", ["{1}"], [], [DeploymentMetricID.DEP_PRS_COUNT], 400),
        (
            2,
            "2018-01-12",
            "2020-01-12",
            ["github.com/src-d/go-git"],
            [],
            [DeploymentMetricID.DEP_PRS_COUNT],
            422,
        ),
        (
            3,
            "2018-01-12",
            "2020-01-12",
            ["github.com/src-d/go-git"],
            [],
            [DeploymentMetricID.DEP_PRS_COUNT],
            404,
        ),
        (1, "2018-01-12", "2020-01-12", ["{1}"], [], ["whatever"], 400),
        (
            1,
            "2018-01-12",
            "2020-01-12",
            ["github.com/athenianco/athenian-api"],
            [],
            [DeploymentMetricID.DEP_PRS_COUNT],
            403,
        ),
        (
            1,
            "2018-01-12",
            "2020-01-12",
            ["{1}"],
            [{"pr_author": ["github.com/akbarik"]}],
            [DeploymentMetricID.DEP_PRS_COUNT],
            400,
        ),
    ],
)
async def test_deployment_metrics_nasty_input(
    client,
    headers,
    account,
    date_from,
    date_to,
    repos,
    withgroups,
    metrics,
    code,
):
    body = {
        "account": account,
        "date_from": date_from,
        "date_to": date_to,
        "for": [
            {
                "repositories": [*repos],
                "withgroups": [*withgroups],
            },
        ],
        "metrics": [*metrics],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/deployments", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == code, "Response body is : " + body


async def test_deployment_metrics_filter_labels(
    client,
    headers,
    precomputed_deployments,
    rdb,
    client_cache,
):
    body = {
        "account": 1,
        "date_from": "2015-01-12",
        "date_to": "2020-03-01",
        "for": [
            {
                "pr_labels_include": ["bug", "plumbing", "enhancement"],
            },
        ],
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
    await rdb.execute(
        insert(DeployedLabel).values(
            {
                DeployedLabel.account_id: 1,
                DeployedLabel.deployment_name: "Dummy deployment",
                DeployedLabel.key: "nginx",
                DeployedLabel.value: 504,
            },
        ),
    )
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
    client,
    headers,
    sample_deployments,
    rdb,
    client_cache,
):
    await rdb.execute(
        delete(DeployedComponent).where(DeployedComponent.deployment_name == "staging_2018_12_02"),
    )
    body = {
        "account": 1,
        "date_from": "2018-01-12",
        "date_to": "2020-03-01",
        "for": [
            {
                "envgroups": [["production"]],
            },
        ],
        "metrics": [
            DeploymentMetricID.DEP_SUCCESS_COUNT,
            DeploymentMetricID.DEP_DURATION_SUCCESSFUL,
        ],
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
        "for": [
            {
                "with": {"pr_author": ["github.com/mcuadros"]},
                "environments": ["production"],
            },
        ],
        "metrics": [
            DeploymentMetricID.DEP_SUCCESS_COUNT,
            DeploymentMetricID.DEP_DURATION_SUCCESSFUL,
        ],
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
        "for": [
            {
                "repositories": ["{1}"],
                "withgroups": [{"releaser": [team_str]}, {"pr_author": [team_str]}],
                "environments": ["production"],
            },
        ],
        "metrics": [
            DeploymentMetricID.DEP_SUCCESS_COUNT,
            DeploymentMetricID.DEP_DURATION_SUCCESSFUL,
        ],
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/deployments", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = [CalculatedDeploymentMetric.from_dict(obj) for obj in json.loads(body)]
    assert [m.to_dict() for m in model] == [
        {
            "for": {
                "repositories": ["{1}"],
                "with": {"releaser": [team_str]},
                "environments": ["production"],
            },
            "metrics": ["dep-success-count", "dep-duration-successful"],
            "granularity": "all",
            "values": [
                {
                    "date": date(2018, 1, 12),
                    "values": [3, "600s"],
                    "confidence_maxs": [None, "600s"],
                    "confidence_mins": [None, "600s"],
                    "confidence_scores": [None, 100],
                },
            ],
        },
        {
            "for": {
                "repositories": ["{1}"],
                "with": {"pr_author": [team_str]},
                "environments": ["production"],
            },
            "metrics": ["dep-success-count", "dep-duration-successful"],
            "granularity": "all",
            "values": [
                {
                    "date": date(2018, 1, 12),
                    "values": [3, "600s"],
                    "confidence_maxs": [None, "600s"],
                    "confidence_mins": [None, "600s"],
                    "confidence_scores": [None, 100],
                },
            ],
        },
    ]
