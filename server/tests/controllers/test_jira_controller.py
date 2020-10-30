from datetime import date, datetime
import json

from dateutil.tz import tzutc
import numpy as np
import pytest

from athenian.api.models.web import CalculatedJIRAMetricValues, CalculatedLinearMetricValues, \
    FoundJIRAStuff, JIRAEpic, JIRALabel, JIRAMetricID, JIRAPriority, JIRAUser


async def test_filter_jira_smoke(client, headers):
    body = {
        "date_from": "2019-10-13",
        "date_to": "2020-01-23",
        "timezone": 120,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/filter/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = FoundJIRAStuff.from_dict(json.loads(body))
    assert model.labels == [
        JIRALabel(title="API",
                  last_used=datetime(2020, 7, 13, 17, 45, 58, tzinfo=tzutc()),
                  issues_count=4, kind="component"),
        JIRALabel(title="Webapp",
                  last_used=datetime(2020, 7, 13, 17, 45, 58, tzinfo=tzutc()),
                  issues_count=1, kind="component"),
        JIRALabel(title="accounts", last_used=datetime(2020, 4, 3, 18, 47, 43, tzinfo=tzutc()),
                  issues_count=1, kind="regular"),
        JIRALabel(title="bug", last_used=datetime(2020, 6, 1, 7, 15, 7, tzinfo=tzutc()),
                  issues_count=16, kind="regular"),
        JIRALabel(title="code-quality", last_used=datetime(2020, 6, 4, 11, 35, 12, tzinfo=tzutc()),
                  issues_count=1, kind="regular"),
        JIRALabel(title="discarded", last_used=datetime(2020, 6, 1, 1, 27, 23, tzinfo=tzutc()),
                  issues_count=4, kind="regular"),
        JIRALabel(title="discussion", last_used=datetime(2020, 3, 31, 21, 16, 11, tzinfo=tzutc()),
                  issues_count=3, kind="regular"),
        JIRALabel(title="feature", last_used=datetime(2020, 4, 3, 18, 48, tzinfo=tzutc()),
                  issues_count=6, kind="regular"),
        JIRALabel(title="functionality",
                  last_used=datetime(2020, 6, 4, 11, 35, 15, tzinfo=tzutc()), issues_count=1,
                  kind="regular"),
        JIRALabel(title="internal-story", last_used=datetime(2020, 6, 1, 7, 15, 7, tzinfo=tzutc()),
                  issues_count=11, kind="regular"),
        JIRALabel(title="needs-specs", last_used=datetime(2020, 4, 6, 13, 25, 2, tzinfo=tzutc()),
                  issues_count=4, kind="regular"),
        JIRALabel(title="onboarding", last_used=datetime(2020, 7, 13, 17, 45, 58, tzinfo=tzutc()),
                  issues_count=1, kind="regular"),
        JIRALabel(title="performance", last_used=datetime(2020, 3, 31, 21, 16, 5, tzinfo=tzutc()),
                  issues_count=1, kind="regular"),
        JIRALabel(title="user-story", last_used=datetime(2020, 4, 3, 18, 48, tzinfo=tzutc()),
                  issues_count=5, kind="regular"),
        JIRALabel(title="webapp", last_used=datetime(2020, 4, 3, 18, 47, 6, tzinfo=tzutc()),
                  issues_count=1, kind="regular"),
    ]
    assert model.epics == [
        JIRAEpic(id="DEV-70", title="Show the installation progress in the waiting page",
                 updated=datetime(2020, 7, 13, 17, 45, 58, tzinfo=tzutc()),
                 children=["DEV-365", "DEV-183", "DEV-315", "DEV-228", "DEV-364"]),
        JIRAEpic(id="ENG-1", title="Evaluate our product and process internally",
                 updated=datetime(2020, 6, 1, 7, 19, tzinfo=tzutc()), children=[]),
    ]
    assert model.issue_types == ["Design document", "Epic", "Story", "Subtask", "Task"]
    assert model.users == [
        JIRAUser(name="David Pordomingo",
                 avatar="https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/initials/DP-4.png",  # noqa
                 type="atlassian"),
        JIRAUser(name="Denys Smirnov",
                 avatar="https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/initials/DS-1.png",  # noqa
                 type="atlassian"),
        JIRAUser(name="Kuba Podgórski",
                 avatar="https://secure.gravatar.com/avatar/ec2f95fe07b5ffec5cde78781f433b68?d=https%3A%2F%2Favatar-management--avatars.us-west-2.prod.public.atl-paas.net%2Finitials%2FKP-3.png",  # noqa
                 type="atlassian"),
        JIRAUser(name="Lou Marvin Caraig",
                 avatar="https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/initials/LC-0.png",  # noqa
                 type="atlassian"),
        JIRAUser(name="Marcelo Novaes",
                 avatar="https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/initials/MN-4.png",  # noqa
                 type="atlassian"),
        JIRAUser(name="Oleksandr Chabaiev",
                 avatar="https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/initials/OC-5.png",  # noqa
                 type="atlassian"),
        JIRAUser(name="Vadim Markovtsev",
                 avatar="https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/initials/VM-6.png",  # noqa
                 type="atlassian"),
        JIRAUser(name="Waren Long",
                 avatar="https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/initials/WL-5.png",  # noqa
                 type="atlassian")]
    assert model.priorities == [
        JIRAPriority(name="Medium",
                     image="https://athenianco.atlassian.net/images/icons/priorities/medium.svg",
                     rank=3),
        JIRAPriority(name="Low",
                     image="https://athenianco.atlassian.net/images/icons/priorities/low.svg",
                     rank=4),
        JIRAPriority(name="None",
                     image="https://athenianco.atlassian.net/images/icons/priorities/trivial.svg",
                     rank=6)]


@pytest.mark.parametrize("account, date_to, timezone, status", [
    (1, "2015-10-12", 0, 400),
    (2, "2020-10-12", 0, 422),
    (3, "2020-10-12", 0, 404),
    (1, "2020-10-12", 100500, 400),
])
async def test_filter_jira_nasty_input(client, headers, account, date_to, timezone, status):
    body = {
        "date_from": "2015-10-13",
        "date_to": date_to,
        "timezone": timezone,
        "account": account,
    }
    response = await client.request(
        method="POST", path="/v1/filter/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == status, "Response body is : " + body


@pytest.mark.parametrize("exclude_inactive", [False, True])
async def test_jira_metrics_smoke(client, headers, exclude_inactive):
    body = {
        "date_from": "2020-01-01",
        "date_to": "2020-10-23",
        "timezone": 120,
        "account": 1,
        "metrics": [JIRAMetricID.JIRA_RAISED, JIRAMetricID.JIRA_RESOLVED],
        "exclude_inactive": exclude_inactive,
        "granularities": ["all", "2 month"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = json.loads(body)
    assert len(body) == 2
    items = [CalculatedJIRAMetricValues.from_dict(i) for i in body]
    assert items[0].granularity == "all"
    assert items[0].values == [CalculatedLinearMetricValues(
        date=date(2019, 12, 31),
        values=[1765, 826],
        confidence_mins=[None] * 2,
        confidence_maxs=[None] * 2,
        confidence_scores=[None] * 2,
    )]
    assert items[1].granularity == "2 month"
    assert len(items[1].values) == 5
    assert items[1].values[0] == CalculatedLinearMetricValues(
        date=date(2019, 12, 31),
        values=[160, 39],
        confidence_mins=[None] * 2,
        confidence_maxs=[None] * 2,
        confidence_scores=[None] * 2,
    )
    assert items[1].values[-1] == CalculatedLinearMetricValues(
        date=date(2020, 8, 31),
        values=[266, 1],
        confidence_mins=[None] * 2,
        confidence_maxs=[None] * 2,
        confidence_scores=[None] * 2,
    )


@pytest.mark.parametrize("account, metrics, date_from, date_to, timezone, granularities, status", [
    (1, [JIRAMetricID.JIRA_RAISED], "2020-01-01", "2020-04-01", 120, ["all"], 200),
    (2, [JIRAMetricID.JIRA_RAISED], "2020-01-01", "2020-04-01", 120, ["all"], 422),
    (3, [JIRAMetricID.JIRA_RAISED], "2020-01-01", "2020-04-01", 120, ["all"], 404),
    (1, [], "2020-01-01", "2020-04-01", 120, ["all"], 200),
    (1, None, "2020-01-01", "2020-04-01", 120, ["all"], 400),
    (1, [JIRAMetricID.JIRA_RAISED], "2020-05-01", "2020-04-01", 120, ["all"], 400),
    (1, [JIRAMetricID.JIRA_RAISED], "2020-01-01", "2020-04-01", 100500, ["all"], 400),
    (1, [JIRAMetricID.JIRA_RAISED], "2020-01-01", "2020-04-01", 120, ["whatever"], 400),
])
async def test_jira_metrics_nasty_input1(
        client, headers, account, metrics, date_from, date_to, timezone, granularities, status):
    body = {
        "date_from": date_from,
        "date_to": date_to,
        "timezone": timezone,
        "account": account,
        "metrics": metrics,
        "exclude_inactive": True,
        "granularities": granularities,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == status, "Response body is : " + body


async def test_jira_metrics_priorities(client, headers):
    body = {
        "date_from": "2020-01-01",
        "date_to": "2020-10-20",
        "timezone": 0,
        "account": 1,
        "metrics": [JIRAMetricID.JIRA_RAISED],
        "exclude_inactive": True,
        "granularities": ["all"],
        "priorities": ["high"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = json.loads(body)
    assert len(body) == 1
    items = [CalculatedJIRAMetricValues.from_dict(i) for i in body]
    assert items[0].granularity == "all"
    assert items[0].values[0].values == [410]


async def test_jira_metrics_types(client, headers):
    body = {
        "date_from": "2020-01-01",
        "date_to": "2020-10-20",
        "timezone": 0,
        "account": 1,
        "metrics": [JIRAMetricID.JIRA_RAISED],
        "exclude_inactive": True,
        "granularities": ["all"],
        "types": ["tASK"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = json.loads(body)
    assert len(body) == 1
    items = [CalculatedJIRAMetricValues.from_dict(i) for i in body]
    assert items[0].granularity == "all"
    assert items[0].values[0].values == [686]


@pytest.mark.parametrize("assignees, reporters, commenters, count", [
    (["Vadim markovtsev"], ["waren long"], ["lou Marvin caraig"], 1177),
    (["Vadim markovtsev"], [], [], 536),
    ([], ["waren long"], [], 567),
    ([], [], ["lou Marvin caraig"], 252),
])
async def test_jira_metrics_people(client, headers, assignees, reporters, commenters, count):
    body = {
        "date_from": "2020-01-01",
        "date_to": "2020-10-20",
        "timezone": 0,
        "account": 1,
        "metrics": [JIRAMetricID.JIRA_RAISED],
        "exclude_inactive": True,
        "granularities": ["all"],
        "assignees": assignees,
        "reporters": reporters,
        "commenters": commenters,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = json.loads(body)
    assert len(body) == 1
    items = [CalculatedJIRAMetricValues.from_dict(i) for i in body]
    assert items[0].granularity == "all"
    assert items[0].values[0].values == [count]


@pytest.mark.parametrize("exclude_inactive, n", [(False, 1005), (True, 993)])
async def test_jira_metrics_open(client, headers, exclude_inactive, n):
    body = {
        "date_from": "2020-06-01",
        "date_to": "2020-10-23",
        "timezone": 120,
        "account": 1,
        "metrics": [JIRAMetricID.JIRA_OPEN],
        "exclude_inactive": exclude_inactive,
        "granularities": ["all"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = json.loads(body)
    assert len(body) == 1
    items = [CalculatedJIRAMetricValues.from_dict(i) for i in body]
    assert items[0].granularity == "all"
    assert items[0].values == [CalculatedLinearMetricValues(
        date=date(2020, 5, 31),
        values=[n],
        confidence_mins=[None],
        confidence_maxs=[None],
        confidence_scores=[None],
    )]


async def test_jira_metrics_bugs_restore(client, headers):
    np.random.seed(7)
    body = {
        "date_from": "2016-01-01",
        "date_to": "2020-10-23",
        "timezone": 120,
        "account": 1,
        "metrics": [JIRAMetricID.JIRA_MTT_RESTORE],
        "types": ["BUG"],
        "exclude_inactive": False,
        "granularities": ["all", "1 year"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    body = json.loads(body)
    assert len(body) == 2
    items = [CalculatedJIRAMetricValues.from_dict(i) for i in body]
    assert items[0].granularity == "all"
    assert items[0].values == [CalculatedLinearMetricValues(
        date=date(2015, 12, 31),
        values=["560503s"],
        confidence_mins=["461188s"],
        confidence_maxs=["657270s"],
        confidence_scores=[66],
    )]
