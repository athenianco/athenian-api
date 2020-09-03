from datetime import datetime
import json

from dateutil.tz import tzutc
import pytest

from athenian.api.models.web import FoundJIRAStuff, JIRAEpic, JIRALabel


async def test_filter_jira_smoke(client, headers):
    body = {
        "date_from": "2015-10-13",
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
                 children=["DEV-364", "DEV-315", "DEV-365", "DEV-228", "DEV-183"]),
        JIRAEpic(id="ENG-1", title="Evaluate our product and process internally",
                 updated=datetime(2020, 6, 1, 7, 19, tzinfo=tzutc()), children=[]),
        JIRAEpic(id="PRO-1", title="Dogfooding instance",
                 updated=datetime(2020, 8, 14, 10, 53, 9, tzinfo=tzutc()), children=[]),
    ]
    assert model.issue_types == ["Design document", "Epic", "Story", "Subtask", "Task"]


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
