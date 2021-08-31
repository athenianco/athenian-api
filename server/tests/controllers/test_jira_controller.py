from datetime import date, datetime, timedelta
import json
from typing import List

from dateutil.tz import tzutc
import numpy as np
import pytest
from sqlalchemy import delete, insert, update

from athenian.api.models.metadata.github import NodePullRequestJiraIssues, PullRequest
from athenian.api.models.metadata.jira import Issue
from athenian.api.models.web import CalculatedJIRAHistogram, CalculatedJIRAMetricValues, \
    CalculatedLinearMetricValues, FilteredJIRAStuff, JIRAEpic, JIRAEpicChild, JIRAFilterReturn, \
    JIRAIssueType, JIRALabel, JIRAMetricID, JIRAPriority, JIRAStatus, JIRAUser
from athenian.api.serialization import FriendlyJson


@pytest.mark.parametrize("return_, checked", [
    (None, set(JIRAFilterReturn)),
    ([], set(JIRAFilterReturn) - {JIRAFilterReturn.ONLY_FLYING}),
    [*([list(set(JIRAFilterReturn) - {JIRAFilterReturn.ONLY_FLYING})] * 2)],
    ([JIRAFilterReturn.EPICS], {JIRAFilterReturn.EPICS}),
    [*([[JIRAFilterReturn.EPICS, JIRAFilterReturn.PRIORITIES, JIRAFilterReturn.STATUSES]] * 2)],
    ([JIRAFilterReturn.ISSUES], ()),
    [*([[JIRAFilterReturn.ISSUES, JIRAFilterReturn.PRIORITIES, JIRAFilterReturn.STATUSES,
         JIRAFilterReturn.ISSUE_TYPES, JIRAFilterReturn.USERS, JIRAFilterReturn.LABELS]] * 2)],
])
async def test_filter_jira_return(client, headers, return_, checked):
    body = {
        "date_from": "2019-10-13",
        "date_to": "2020-01-23",
        "timezone": 120,
        "account": 1,
        "exclude_inactive": False,
    }
    if return_ is not None:
        body["return"] = return_
    response = await client.request(
        method="POST", path="/v1/filter/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = FilteredJIRAStuff.from_dict(json.loads(body))
    if "labels" in checked:
        assert model.labels == [
            JIRALabel(title="API",
                      last_used=datetime(2020, 7, 13, 17, 45, 58, tzinfo=tzutc()),
                      issues_count=4, kind="component"),
            JIRALabel(title="Webapp",
                      last_used=datetime(2020, 7, 13, 17, 45, 58, tzinfo=tzutc()),
                      issues_count=1, kind="component"),
            JIRALabel(title="accounts",
                      last_used=datetime(2020, 12, 15, 10, 16, 15, tzinfo=tzutc()),
                      issues_count=1, kind="regular"),
            JIRALabel(title="bug",
                      last_used=datetime(2020, 6, 1, 7, 15, 7, tzinfo=tzutc()),
                      issues_count=16, kind="regular"),
            JIRALabel(title="code-quality",
                      last_used=datetime(2020, 6, 4, 11, 35, 12, tzinfo=tzutc()),
                      issues_count=1, kind="regular"),
            JIRALabel(title="discarded",
                      last_used=datetime(2020, 6, 1, 1, 27, 23, tzinfo=tzutc()),
                      issues_count=4, kind="regular"),
            JIRALabel(title="discussion",
                      last_used=datetime(2020, 3, 31, 21, 16, 11, tzinfo=tzutc()),
                      issues_count=3, kind="regular"),
            JIRALabel(title="feature",
                      last_used=datetime(2020, 4, 3, 18, 48, tzinfo=tzutc()),
                      issues_count=6, kind="regular"),
            JIRALabel(title="functionality",
                      last_used=datetime(2020, 6, 4, 11, 35, 15, tzinfo=tzutc()), issues_count=1,
                      kind="regular"),
            JIRALabel(title="internal-story",
                      last_used=datetime(2020, 6, 1, 7, 15, 7, tzinfo=tzutc()),
                      issues_count=11, kind="regular"),
            JIRALabel(title="needs-specs",
                      last_used=datetime(2020, 4, 6, 13, 25, 2, tzinfo=tzutc()),
                      issues_count=4, kind="regular"),
            JIRALabel(title="onboarding",
                      last_used=datetime(2020, 7, 13, 17, 45, 58, tzinfo=tzutc()),
                      issues_count=1, kind="regular"),
            JIRALabel(title="performance",
                      last_used=datetime(2020, 3, 31, 21, 16, 5, tzinfo=tzutc()),
                      issues_count=1, kind="regular"),
            JIRALabel(title="user-story",
                      last_used=datetime(2020, 4, 3, 18, 48, tzinfo=tzutc()),
                      issues_count=5, kind="regular"),
            JIRALabel(title="webapp",
                      last_used=datetime(2020, 4, 3, 18, 47, 6, tzinfo=tzutc()),
                      issues_count=1, kind="regular"),
        ]
    else:
        assert model.labels is None
    if "epics" in checked:
        true_epics = [
            JIRAEpic(id="ENG-1", title="Evaluate our product and process internally",
                     created=datetime(2019, 12, 2, 14, 19, 58, 762),
                     updated=datetime(2020, 6, 1, 7, 19, 0, 316),
                     work_began=datetime(2020, 6, 1, 7, 19, 0, 335),
                     resolved=datetime(2020, 6, 1, 7, 19, 0, 335),
                     lead_time=timedelta(0), life_time=timedelta(days=181, seconds=61141),
                     reporter="Lou Marvin Caraig", assignee="Waren Long", comments=0,
                     priority="Medium", status="Done", prs=0, project="10003", children=[],
                     url="https://athenianco.atlassian.net/browse/ENG-1"),
            JIRAEpic(id="DEV-70", title="Show the installation progress in the waiting page",
                     created=datetime(2020, 1, 22, 16, 57, 10, 253),
                     updated=datetime(2020, 7, 13, 17, 45, 58, 294),
                     work_began=datetime(2020, 6, 2, 11, 40, 42, 905),
                     resolved=datetime(2020, 7, 13, 17, 45, 58, 305),
                     lead_time=timedelta(days=41, seconds=21915),
                     life_time=timedelta(days=173, seconds=2928),
                     reporter="Lou Marvin Caraig", assignee="David Pordomingo",
                     url="https://athenianco.atlassian.net/browse/DEV-70",
                     comments=8, priority="Low", status="Released",
                     prs=0, project="10009", children=[
                         JIRAEpicChild(
                             id="DEV-183",
                             title="Implement the endpoint that returns the installation progress",
                             created=datetime(2020, 6, 2, 11, 7, 36, 558),
                             updated=datetime(2020, 6, 2, 11, 40, 42, 891),
                             work_began=datetime(2020, 6, 2, 11, 40, 42, 905),
                             resolved=datetime(2020, 6, 2, 11, 40, 42, 905),
                             lead_time=timedelta(0), life_time=timedelta(seconds=1986),
                             reporter="Waren Long", assignee="Vadim Markovtsev",
                             comments=1, priority="Low", status="Closed", prs=0, type="Task",
                             subtasks=0, url="https://athenianco.atlassian.net/browse/DEV-183"),
                         JIRAEpicChild(
                             id="DEV-228",
                             title="Consider installation progress without updates during 3 hours as complete",  # noqa
                             created=datetime(2020, 6, 8, 9, 0, 22, 517),
                             updated=datetime(2020, 6, 16, 18, 12, 38, 634),
                             work_began=datetime(2020, 6, 9, 10, 8, 15, 357),
                             resolved=datetime(2020, 6, 9, 10, 34, 7, 221),
                             lead_time=timedelta(seconds=1551),
                             life_time=timedelta(days=1, seconds=5624),
                             reporter="Vadim Markovtsev", assignee="Vadim Markovtsev",
                             comments=1, priority="Medium",
                             status="Released", prs=0, type="Task", subtasks=0,
                             url="https://athenianco.atlassian.net/browse/DEV-228"),
                        JIRAEpicChild(
                            id="DEV-315",
                            title="Add a progress bar in the waiting page to show the installation progress",  # noqa
                            created=datetime(2020, 6, 18, 21, 51, 19, 344),
                            updated=datetime(2020, 7, 27, 16, 56, 20, 144),
                            work_began=datetime(2020, 6, 25, 17, 8, 11, 311),
                            resolved=datetime(2020, 7, 13, 17, 43, 20, 317),
                            lead_time=timedelta(days=18, seconds=2109),
                            life_time=timedelta(days=24, seconds=71520),
                            reporter="Waren Long", assignee="David Pordomingo",
                            comments=4, priority="High", status="Released", prs=0, type="Story",
                            subtasks=0, url="https://athenianco.atlassian.net/browse/DEV-315"),
                        JIRAEpicChild(
                            id="DEV-364",
                            title="Block the access to the Overview page until the installation is 100% complete",  # noqa
                            created=datetime(2020, 6, 25, 16, 12, 34, 233),
                            updated=datetime(2020, 7, 27, 16, 56, 22, 968),
                            work_began=datetime(2020, 7, 2, 4, 23, 20, 3),
                            resolved=datetime(2020, 7, 13, 17, 46, 15, 634),
                            lead_time=timedelta(days=11, seconds=48175),
                            life_time=timedelta(days=18, seconds=5621),
                            reporter="Waren Long", assignee="David Pordomingo",
                            comments=1, priority="Medium", status="Released", prs=0, type="Story",
                            subtasks=0, url="https://athenianco.atlassian.net/browse/DEV-364"),
                        JIRAEpicChild(
                            id="DEV-365",
                            title="Design the success view telling the user the installation is complete and the account ready to use",  # noqa
                            created=datetime(2020, 6, 25, 16, 17, 14, 436),
                            updated=datetime(2020, 6, 26, 9, 38, 36, 635),
                            work_began=datetime(2020, 6, 26, 9, 33, 9, 184),
                            resolved=datetime(2020, 6, 26, 9, 35, 43, 579),
                            lead_time=timedelta(seconds=154),
                            life_time=timedelta(seconds=62309),
                            reporter="Waren Long", assignee="Zuri Negrin",
                            comments=2, priority="Medium", status="Released", prs=0, type="Story",
                            subtasks=0, url="https://athenianco.atlassian.net/browse/DEV-365"),
                     ]),
        ]
        assert model.epics == true_epics
    else:
        assert model.epics is None
    if "issue_types" in checked:
        assert model.issue_types == [
            JIRAIssueType(name="Design document", count=10, project="10003", is_subtask=False,
                          normalized_name="designdocument",
                          image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10322&avatarType=issuetype"),  # noqa
            JIRAIssueType(name="Epic", count=1, project="10003", is_subtask=False,
                          normalized_name="epic",
                          image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10307&avatarType=issuetype"),  # noqa
            JIRAIssueType(name="Epic", count=1,
                          image="https://athenianco.atlassian.net/images/icons/issuetypes/epic.svg",  # noqa
                          project="10009", is_subtask=False, normalized_name="epic"),
            JIRAIssueType(name="Story", count=49, project="10003", is_subtask=False,
                          normalized_name="story",
                          image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10315&avatarType=issuetype"),  # noqa
            JIRAIssueType(name="Subtask", count=98, project="10003", is_subtask=True,
                          normalized_name="subtask",
                          image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10316&avatarType=issuetype"),  # noqa
            JIRAIssueType(name="Task", count=4, project="10003", is_subtask=False,
                          normalized_name="task",
                          image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10318&avatarType=issuetype"),  # noqa
            JIRAIssueType(name="Task", count=5,
                          image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10318&avatarType=issuetype",  # noqa
                          project="10009", is_subtask=False, normalized_name="task"),
        ]
    else:
        assert model.issue_types is None
    if "users" in checked:
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
                     type="atlassian"),
        ]
    else:
        assert model.users is None
    if "priorities" in checked:
        true_priorities = [
            JIRAPriority(name="High",
                         image="https://athenianco.atlassian.net/images/icons/priorities/high.svg",
                         rank=2,
                         color="EA4444"),
            JIRAPriority(name="Medium",
                         image="https://athenianco.atlassian.net/images/icons/priorities/medium.svg",  # noqa
                         rank=3,
                         color="EA7D24"),
            JIRAPriority(name="Low",
                         image="https://athenianco.atlassian.net/images/icons/priorities/low.svg",
                         rank=4,
                         color="2A8735"),
            JIRAPriority(name="None",
                         image="https://athenianco.atlassian.net/images/icons/priorities/trivial.svg",  # noqa
                         rank=6,
                         color="9AA1B2"),
        ]
        if "issues" not in checked:
            true_priorities.pop(-1)
        if "epics" not in checked:
            # some children of the filtered epics do not belong to the given time interval
            true_priorities.pop(0)
        assert model.priorities == true_priorities
    else:
        assert model.priorities is None
    if "statuses" in checked:
        true_statuses = [
            JIRAStatus(name="Backlog", stage="To Do", project="10009"),
            JIRAStatus(name="Closed", stage="Done", project="10009"),
            JIRAStatus(name="Done", stage="Done", project="10003"),
            JIRAStatus(name="Released", stage="Done", project="10009"),
        ]
        if "issues" not in checked:
            true_statuses.pop(0)
        assert model.statuses == true_statuses
    else:
        assert model.statuses is None
    if "issue_bodies" in checked:
        assert len(model.issues) == 168
        work_begans = resolveds = assignees = comments = 0
        ids = set()
        for issue in model.issues:
            assert issue.id
            ids.add(issue.id)
            assert issue.title
            assert issue.created
            assert issue.updated
            work_begans += bool(issue.work_began)
            resolveds += bool(issue.resolved)
            if issue.resolved:
                assert issue.lead_time is not None
            assert issue.reporter
            assignees += bool(issue.assignee)
            comments += bool(issue.comments)
            assert issue.priority
            assert issue.status
            assert issue.type
            assert issue.project
            assert issue.url
            if issue.work_began:
                assert issue.lead_time is not None
            else:
                assert issue.lead_time is None
            assert issue.life_time
            # they are not mapped for this time range
            assert not issue.prs
        assert work_begans == resolveds
        assert resolveds == 164
        assert assignees == 149
        assert len(ids) == len(model.issues)
        # assert comments > 0 # FIXME(vmarkovtsev): DEV-1658
    else:
        assert not model.issues


async def test_filter_jira_epics_no_time(client, headers):
    body = {
        "date_from": None,
        "date_to": None,
        "timezone": 120,
        "account": 1,
        "exclude_inactive": True,
        "return": ["epics", "priorities", "statuses"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = FilteredJIRAStuff.from_dict(json.loads(body))
    assert len(model.epics) == 81
    assert len(model.priorities) == 6
    assert len(model.statuses) == 7


@pytest.mark.parametrize("ikey", ("DEV-162", "DEV-163"))
async def test_filter_jira_epics_deleted(client, headers, ikey, mdb):
    await mdb.execute(update(Issue).where(Issue.key == ikey).values({Issue.is_deleted: True}))
    try:
        body = {
            "date_from": "2020-04-09",
            "date_to": "2020-04-09",
            "timezone": 120,
            "account": 1,
            "exclude_inactive": True,
            "return": ["epics"],
        }
        response = await client.request(
            method="POST", path="/v1/filter/jira", headers=headers, json=body,
        )
        body = (await response.read()).decode("utf-8")
        assert response.status == 200, "Response body is : " + body
        model = FilteredJIRAStuff.from_dict(json.loads(body))
        noticed_epic = False
        for epic in model.epics:
            if epic.id == "DEV-163":
                noticed_epic = True
                assert not epic.children
        if ikey == "DEV-163":
            assert not noticed_epic
        assert len(model.epics) == 17 + noticed_epic
    finally:
        await mdb.execute(update(Issue).where(Issue.key == ikey).values({Issue.is_deleted: False}))


@pytest.mark.parametrize("exclude_inactive, labels, epics, types, users, priorities", [
    [False, 33, 34, [
        JIRAIssueType(name="Bug", count=2,
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10303&avatarType=issuetype",  # noqa
                      project="10003", is_subtask=False, normalized_name="bug"),
        JIRAIssueType(name="Bug", count=94, project="10009", is_subtask=False,
                      normalized_name="bug",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10303&avatarType=issuetype"),  # noqa
        JIRAIssueType(name="Design Document", count=7, project="10009", is_subtask=False,
                      normalized_name="designdocument",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10322&avatarType=issuetype"),  # noqa
        JIRAIssueType(name="Epic", count=3,
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10307&avatarType=issuetype",  # noqa
                      project="10003", is_subtask=False, normalized_name="epic"),
        JIRAIssueType(name="Epic", count=31, project="10009", is_subtask=False,
                      normalized_name="epic",
                      image="https://athenianco.atlassian.net/images/icons/issuetypes/epic.svg"),
        JIRAIssueType(name="Incident", count=3, project="10009", is_subtask=False,
                      normalized_name="incident",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10304&avatarType=issuetype"),  # noqa
        JIRAIssueType(name="Story", count=26, project="10009", is_subtask=False,
                      normalized_name="story",
                      image="https://athenianco.atlassian.net/images/icons/issuetypes/story.svg"),
        JIRAIssueType(name="Subtask", count=1, project="10003", is_subtask=True,
                      normalized_name="subtask",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10316&avatarType=issuetype"),  # noqa
        JIRAIssueType(name="Sub-task", count=26, project="10009", is_subtask=True,
                      normalized_name="subtask",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10316&avatarType=issuetype"),  # noqa
        JIRAIssueType(name="Task", count=194, project="10009", is_subtask=False,
                      normalized_name="task",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10318&avatarType=issuetype")],  # noqa
     15, 6],
    [True, 32, 32, [
        JIRAIssueType(name="Bug", count=1,
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10303&avatarType=issuetype",  # noqa
                      project="10003", is_subtask=False, normalized_name="bug"),
        JIRAIssueType(name="Bug", count=84, project="10009", is_subtask=False,
                      normalized_name="bug",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10303&avatarType=issuetype"),  # noqa
        JIRAIssueType(name="Design Document", count=4, project="10009", is_subtask=False,
                      normalized_name="designdocument",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10322&avatarType=issuetype"),  # noqa
        JIRAIssueType(name="Epic", count=1,
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10307&avatarType=issuetype",  # noqa
                      project="10003", is_subtask=False, normalized_name="epic"),
        JIRAIssueType(name="Epic", count=31, project="10009", is_subtask=False,
                      normalized_name="epic",
                      image="https://athenianco.atlassian.net/images/icons/issuetypes/epic.svg"),
        JIRAIssueType(name="Incident", count=3, project="10009", is_subtask=False,
                      normalized_name="incident",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10304&avatarType=issuetype"),  # noqa
        JIRAIssueType(name="Story", count=14, project="10009", is_subtask=False,
                      normalized_name="story",
                      image="https://athenianco.atlassian.net/images/icons/issuetypes/story.svg"),
        JIRAIssueType(name="Sub-task", count=26, project="10009", is_subtask=True,
                      normalized_name="subtask",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10316&avatarType=issuetype"),  # noqa
        JIRAIssueType(name="Task", count=156, project="10009", is_subtask=False,
                      normalized_name="task",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10318&avatarType=issuetype")],  # noqa
     13, 6],
])
async def test_filter_jira_exclude_inactive(
        client, headers, exclude_inactive, labels, epics, types, users, priorities):
    body = {
        "date_from": "2020-09-13",
        "date_to": "2020-10-23",
        "timezone": 120,
        "account": 1,
        "exclude_inactive": exclude_inactive,
    }
    response = await client.request(
        method="POST", path="/v1/filter/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = FilteredJIRAStuff.from_dict(json.loads(body))
    assert len(model.labels) == labels
    assert len(model.epics) == epics
    assert model.issue_types == types
    assert len(model.users) == users
    assert len(model.priorities) == priorities


async def test_filter_jira_disabled_projects(client, headers, disabled_dev):
    body = {
        "date_from": "2019-10-13",
        "date_to": "2020-01-23",
        "timezone": 120,
        "account": 1,
        "exclude_inactive": False,
    }
    response = await client.request(
        method="POST", path="/v1/filter/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = FilteredJIRAStuff.from_dict(json.loads(body))
    _check_filter_jira_no_dev_project(model)


async def test_filter_jira_selected_projects(client, headers):
    body = {
        "date_from": "2019-10-13",
        "date_to": "2020-01-23",
        "timezone": 120,
        "account": 1,
        "exclude_inactive": False,
        "projects": ["PRO", "OPS", "ENG", "GRW", "CS", "CON"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = FilteredJIRAStuff.from_dict(json.loads(body))
    _check_filter_jira_no_dev_project(model)


def _check_filter_jira_no_dev_project(model: FilteredJIRAStuff) -> None:
    assert model.labels == [
        JIRALabel(title="accounts", last_used=datetime(2020, 12, 15, 10, 16, 15, tzinfo=tzutc()),
                  issues_count=1, kind="regular"),
        JIRALabel(title="bug", last_used=datetime(2020, 6, 1, 7, 15, 7, tzinfo=tzutc()),
                  issues_count=16, kind="regular"),
        JIRALabel(title="discarded", last_used=datetime(2020, 6, 1, 1, 27, 23, tzinfo=tzutc()),
                  issues_count=4, kind="regular"),
        JIRALabel(title="discussion", last_used=datetime(2020, 3, 31, 21, 16, 11, tzinfo=tzutc()),
                  issues_count=3, kind="regular"),
        JIRALabel(title="feature", last_used=datetime(2020, 4, 3, 18, 48, tzinfo=tzutc()),
                  issues_count=6, kind="regular"),
        JIRALabel(title="internal-story", last_used=datetime(2020, 6, 1, 7, 15, 7, tzinfo=tzutc()),
                  issues_count=11, kind="regular"),
        JIRALabel(title="needs-specs", last_used=datetime(2020, 4, 6, 13, 25, 2, tzinfo=tzutc()),
                  issues_count=4, kind="regular"),
        JIRALabel(title="performance", last_used=datetime(2020, 3, 31, 21, 16, 5, tzinfo=tzutc()),
                  issues_count=1, kind="regular"),
        JIRALabel(title="user-story", last_used=datetime(2020, 4, 3, 18, 48, tzinfo=tzutc()),
                  issues_count=5, kind="regular"),
        JIRALabel(title="webapp", last_used=datetime(2020, 4, 3, 18, 47, 6, tzinfo=tzutc()),
                  issues_count=1, kind="regular")]
    assert model.epics == [
        JIRAEpic(id="ENG-1", title="Evaluate our product and process internally",
                 created=datetime(2019, 12, 2, 14, 19, 58, 762),
                 updated=datetime(2020, 6, 1, 7, 19, 0, 316),
                 work_began=datetime(2020, 6, 1, 7, 19, 0, 335),
                 resolved=datetime(2020, 6, 1, 7, 19, 0, 335),
                 lead_time=timedelta(0),
                 life_time=timedelta(days=181, seconds=61141),
                 reporter="Lou Marvin Caraig", assignee="Waren Long",
                 comments=0, priority="Medium", status="Done", prs=0, project="10003",
                 children=[], url="https://athenianco.atlassian.net/browse/ENG-1"),
    ]
    assert model.issue_types == [
        JIRAIssueType(name="Design document", count=10, project="10003", is_subtask=False,
                      normalized_name="designdocument",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10322&avatarType=issuetype"),  # noqa
        JIRAIssueType(name="Epic", count=1, project="10003", is_subtask=False,
                      normalized_name="epic",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10307&avatarType=issuetype"),  # noqa
        JIRAIssueType(name="Story", count=49, project="10003", is_subtask=False,
                      normalized_name="story",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10315&avatarType=issuetype"),  # noqa
        JIRAIssueType(name="Subtask", count=98, project="10003", is_subtask=True,
                      normalized_name="subtask",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10316&avatarType=issuetype"),  # noqa
        JIRAIssueType(name="Task", count=4, project="10003", is_subtask=False,
                      normalized_name="task",
                      image="https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId=10318&avatarType=issuetype"),  # noqa
    ]


async def test_filter_jira_no_epics(client, headers):
    body = {
        "date_from": None,
        "date_to": None,
        "timezone": 120,
        "account": 1,
        "exclude_inactive": False,
        "priorities": ["Impossible"],
        "return": ["epics", "priorities", "statuses"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = FilteredJIRAStuff.from_dict(json.loads(body))
    assert len(model.epics) == 0


async def test_filter_jira_extended_filters(client, headers):
    body = {
        "date_from": None,
        "date_to": None,
        "timezone": 120,
        "account": 1,
        "exclude_inactive": False,
        "priorities": ["High", "Medium"],
        "with": {"assignees": ["Vadim Markovtsev", None]},
        "return": ["epics", "priorities", "statuses"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = FilteredJIRAStuff.from_dict(json.loads(body))
    assert len(model.epics) == 48, str(sorted(epic.id for epic in model.epics))
    assert len(model.priorities) == 6  # two projects
    assert len(model.statuses) == 7


async def test_filter_jira_issue_types_filter(client, headers):
    body = {
        "date_from": None,
        "date_to": None,
        "timezone": 120,
        "account": 1,
        "exclude_inactive": False,
        "types": ["Bug"],
        "with": {"assignees": ["Vadim Markovtsev", None]},
        "return": ["issues", "issue_types"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = FilteredJIRAStuff.from_dict(json.loads(body))
    assert len(model.issue_types) == 2
    assert not model.issues


async def test_filter_jira_issue_prs_comments(client, headers):
    body = {
        "date_from": "2020-09-01",
        "date_to": "2021-01-01",
        "timezone": 120,
        "account": 1,
        "exclude_inactive": True,
        "return": ["issues", "issue_bodies"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = FilteredJIRAStuff.from_dict(json.loads(body))
    assert len(model.issues) == 389
    prs = 0
    comments = 0
    for issue in model.issues:
        prs += bool(issue.prs)
        comments += issue.comments
        for pr in issue.prs or []:
            assert pr.number > 0
            assert not pr.jira
    assert prs == 6
    assert comments == 1113


async def test_filter_jira_issue_only_flying(client, headers):
    body = {
        "date_from": "2020-09-01",
        "date_to": "2021-01-01",
        "timezone": 120,
        "account": 1,
        "exclude_inactive": True,
        "return": ["issues", "issue_bodies", "only_flying"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = FilteredJIRAStuff.from_dict(json.loads(body))
    assert len(model.issues) == 235


async def test_filter_jira_issue_disabled(client, headers, mdb):
    ikey = "ENG-303"
    await mdb.execute(update(Issue).where(Issue.key == ikey).values({Issue.is_deleted: True}))
    try:
        body = {
            "date_from": "2020-09-01",
            "date_to": "2021-01-01",
            "timezone": 120,
            "account": 1,
            "exclude_inactive": True,
            "return": ["issues", "issue_bodies"],
        }
        response = await client.request(
            method="POST", path="/v1/filter/jira", headers=headers, json=body,
        )
        body = (await response.read()).decode("utf-8")
        assert response.status == 200, "Response body is : " + body
        model = FilteredJIRAStuff.from_dict(json.loads(body))
        assert len(model.issues) == 389 - 1
    finally:
        await mdb.execute(update(Issue).where(Issue.key == ikey).values({Issue.is_deleted: False}))


async def test_filter_jira_deleted_repositories(client, headers, mdb):
    # DEV-2082
    await mdb.execute(insert(NodePullRequestJiraIssues).values(dict(
        node_id=1234,
        node_acc=6366825,
        jira_acc=1,
        jira_id="12541",
    )))
    await mdb.execute(insert(PullRequest).values(dict(
        node_id=1234,
        acc_id=6366825,
        repository_full_name="athenianco/athenian-api",
        repository_node_id=4321,
        base_ref="base_ref",
        head_ref="head_ref",
        number=100500,
        closed=True,
    )))
    try:
        body = {
            "date_from": "2020-11-01",
            "date_to": "2020-12-01",
            "timezone": 120,
            "account": 1,
            "exclude_inactive": True,
            "return": ["issues", "issue_bodies"],
        }
        response = await client.request(
            method="POST", path="/v1/filter/jira", headers=headers, json=body,
        )
        body = (await response.read()).decode("utf-8")
        assert response.status == 200, "Response body is : " + body
        model = FilteredJIRAStuff.from_dict(json.loads(body))
        prs = sum(bool(issue.prs) for issue in model.issues)
        assert len(model.issues) == 112
        assert prs == 1
    finally:
        await mdb.execute(delete(NodePullRequestJiraIssues)
                          .where(NodePullRequestJiraIssues.node_id == 1234))
        await mdb.execute(delete(PullRequest)
                          .where(PullRequest.node_id == 1234))


@pytest.mark.parametrize("account, date_to, tz, status", [
    (1, "2015-10-12", 0, 400),
    (1, None, 0, 400),
    (2, "2020-10-12", 0, 422),
    (3, "2020-10-12", 0, 404),
    (1, "2020-10-12", 100500, 400),
])
async def test_filter_jira_nasty_input(client, headers, account, date_to, tz, status):
    body = {
        "date_from": "2015-10-13",
        "date_to": date_to,
        "timezone": tz,
        "account": account,
        "exclude_inactive": True,
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
    assert items[0].with_ is None
    assert items[0].values == [CalculatedLinearMetricValues(
        date=date(2020, 1, 1),
        values=[1765, 1628],
        confidence_mins=[None] * 2,
        confidence_maxs=[None] * 2,
        confidence_scores=[None] * 2,
    )]
    assert items[1].granularity == "2 month"
    assert items[1].with_ is None
    assert len(items[1].values) == 5
    assert items[1].values[0] == CalculatedLinearMetricValues(
        date=date(2020, 1, 1),
        values=[160, 39],
        confidence_mins=[None] * 2,
        confidence_maxs=[None] * 2,
        confidence_scores=[None] * 2,
    )
    assert items[1].values[-1] == CalculatedLinearMetricValues(
        date=date(2020, 9, 1),
        values=[266, 243],
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
        "with": [],
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


async def test_jira_metrics_epics(client, headers):
    body = {
        "date_from": "2020-01-01",
        "date_to": "2020-10-20",
        "timezone": 0,
        "account": 1,
        "metrics": [JIRAMetricID.JIRA_RAISED],
        "exclude_inactive": True,
        "granularities": ["all"],
        "epics": ["DEV-70", "DEV-843"],
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
    assert items[0].values[0].values == [38]


async def test_jira_metrics_labels(client, headers):
    body = {
        "date_from": "2020-01-01",
        "date_to": "2020-10-20",
        "timezone": 0,
        "account": 1,
        "metrics": [JIRAMetricID.JIRA_RAISED],
        "exclude_inactive": True,
        "granularities": ["all"],
        "labels_include": ["PERFORmance"],
        "labels_exclude": ["buG"],
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
    assert items[0].values[0].values == [147]  # it is 148 without labels_exclude


@pytest.mark.parametrize("assignees, reporters, commenters, count", [
    (["Vadim markovtsev"], ["waren long"], ["lou Marvin caraig"], 1177),
    (["Vadim markovtsev"], [], [], 536),
    ([None, "Vadim MARKOVTSEV"], [], [], 708),
    ([], ["waren long"], [], 567),
    ([], [], ["lou Marvin caraig"], 252),
    ([None], [], ["lou Marvin caraig"], 403),
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
        "with": [{
            "assignees": assignees,
            "reporters": reporters,
            "commenters": commenters,
        }],
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
    with_ = {}
    for key, val in {
        "assignees": assignees,
        "reporters": reporters,
        "commenters": commenters,
    }.items():
        if val:
            with_[key] = val
    assert items[0].with_.to_dict() == with_
    assert items[0].values[0].values == [count]


async def test_jira_metrics_teams(client, headers):
    body = {
        "date_from": "2020-01-01",
        "date_to": "2020-10-20",
        "timezone": 0,
        "account": 1,
        "metrics": [JIRAMetricID.JIRA_RAISED],
        "exclude_inactive": True,
        "granularities": ["all"],
        "with": [{
            "assignees": ["vadim Markovtsev"],
        }, {
            "reporters": ["waren Long"],
        }],
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
    assert items[0].values[0].values == [536]
    assert items[0].with_.to_dict() == {"assignees": ["vadim Markovtsev"]}
    assert items[1].values[0].values == [567]
    assert items[1].with_.to_dict() == {"reporters": ["waren Long"]}


@pytest.mark.parametrize("metric, exclude_inactive, n", [
    (JIRAMetricID.JIRA_OPEN, False, 208),
    (JIRAMetricID.JIRA_OPEN, True, 199),
    (JIRAMetricID.JIRA_RESOLVED, False, 850),
    (JIRAMetricID.JIRA_RESOLVED, True, 850),
    (JIRAMetricID.JIRA_ACKNOWLEDGED, False, 805),
    (JIRAMetricID.JIRA_ACKNOWLEDGED_Q, False, 805),
    (JIRAMetricID.JIRA_RESOLUTION_RATE, False, 0.9593679458239278),
])
async def test_jira_metrics_counts(client, headers, metric, exclude_inactive, n):
    body = {
        "date_from": "2020-06-01",
        "date_to": "2020-10-23",
        "timezone": 120,
        "account": 1,
        "metrics": [metric],
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
        date=date(2020, 6, 1),
        values=[n],
        confidence_mins=[None],
        confidence_maxs=[None],
        confidence_scores=[None],
    )]


@pytest.mark.parametrize("metric, value, score, cmin, cmax", [
    (JIRAMetricID.JIRA_LIFE_TIME, "758190s", 72, "647080s", "859905s"),
    (JIRAMetricID.JIRA_LEAD_TIME, "304289s", 53, "229359s", "374686s"),
    (JIRAMetricID.JIRA_ACKNOWLEDGE_TIME, "448797s", 63, "365868s", "532126s"),
])
async def test_jira_metrics_bug_times(client, headers, metric, value, score, cmin, cmax):
    np.random.seed(7)
    body = {
        "date_from": "2016-01-01",
        "date_to": "2020-10-23",
        "timezone": 120,
        "account": 1,
        "metrics": [metric],
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
        date=date(2016, 1, 1),
        values=[value],
        confidence_mins=[cmin],
        confidence_maxs=[cmax],
        confidence_scores=[score],
    )]


async def test_jira_metrics_disabled_projects(client, headers, disabled_dev):
    body = {
        "date_from": "2020-01-01",
        "date_to": "2020-10-23",
        "timezone": 120,
        "account": 1,
        "metrics": [JIRAMetricID.JIRA_RAISED, JIRAMetricID.JIRA_RESOLVED],
        "exclude_inactive": False,
        "granularities": ["all", "2 month"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    items = [CalculatedJIRAMetricValues.from_dict(i) for i in json.loads(body)]
    _check_metrics_no_dev_project(items)


async def test_jira_metrics_selected_projects(client, headers):
    body = {
        "date_from": "2020-01-01",
        "date_to": "2020-10-23",
        "timezone": 120,
        "account": 1,
        "metrics": [JIRAMetricID.JIRA_RAISED, JIRAMetricID.JIRA_RESOLVED],
        "exclude_inactive": False,
        "projects": ["PRO", "OPS", "ENG", "GRW", "CS", "CON"],
        "granularities": ["all", "2 month"],
    }
    response = await client.request(
        method="POST", path="/v1/metrics/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    items = [CalculatedJIRAMetricValues.from_dict(i) for i in json.loads(body)]
    _check_metrics_no_dev_project(items)


def _check_metrics_no_dev_project(items: List[CalculatedJIRAMetricValues]) -> None:
    assert items[0].values == [CalculatedLinearMetricValues(
        date=date(2020, 1, 1),
        values=[768, 829],
        confidence_mins=[None] * 2,
        confidence_maxs=[None] * 2,
        confidence_scores=[None] * 2,
    )]


async def test_jira_metrics_group_by_label_smoke(client, headers):
    body = {
        "date_from": "2020-01-01",
        "date_to": "2020-10-20",
        "timezone": 0,
        "account": 1,
        "metrics": [JIRAMetricID.JIRA_RAISED],
        "exclude_inactive": True,
        "granularities": ["all"],
        "group_by_jira_label": True,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/jira", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    rbody = json.loads(rbody)
    assert len(rbody) == 49
    items = [CalculatedJIRAMetricValues.from_dict(i) for i in rbody]
    assert items[0].granularity == "all"
    assert items[0].jira_label == "performance"
    assert items[0].values[0].values == [148]
    assert items[1].jira_label == "webapp"
    assert items[1].values[0].values == [143]
    assert items[-1].jira_label is None
    assert items[-1].values[0].values == [749]

    body["labels_include"] = ["performance"]
    body["labels_exclude"] = ["security"]
    response = await client.request(
        method="POST", path="/v1/metrics/jira", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    rbody = json.loads(rbody)
    assert len(rbody) == 1
    items = [CalculatedJIRAMetricValues.from_dict(i) for i in rbody]
    assert items[0].granularity == "all"
    assert items[0].jira_label == "performance"
    assert items[0].values[0].values == [147]


async def test_jira_metrics_group_by_label_empty(client, headers):
    body = {
        "date_from": "2019-12-02",
        "date_to": "2019-12-03",
        "timezone": 0,
        "account": 1,
        "metrics": [JIRAMetricID.JIRA_RAISED],
        "exclude_inactive": True,
        "granularities": ["all"],
        "group_by_jira_label": True,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/jira", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    rbody = json.loads(rbody)
    assert len(rbody) == 1
    items = [CalculatedJIRAMetricValues.from_dict(i) for i in rbody]
    assert len(items) == 1
    assert items[0].granularity == "all"
    assert items[0].jira_label is None
    assert items[0].values[0].values == [9]

    body["labels_include"] = ["whatever"]
    response = await client.request(
        method="POST", path="/v1/metrics/jira", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    rbody = json.loads(rbody)
    assert len(rbody) == 1
    items = [CalculatedJIRAMetricValues.from_dict(i) for i in rbody]
    assert len(items) == 1
    assert items[0].granularity == "all"
    assert items[0].jira_label is None
    assert items[0].values[0].values == [0]


@pytest.mark.parametrize("with_, ticks, frequencies, interquartile", [
    [None,
     [["60s", "122s", "249s", "507s", "1033s", "2105s", "4288s", "8737s", "17799s", "36261s",
       "73870s", "150489s", "306576s", "624554s", "1272338s", "2591999s"]],
     [[351, 7, 12, 27, 38, 70, 95, 103, 68, 76, 120, 116, 132, 114, 285]],
     [{"left": "1255s", "right": "618082s"}],
     ],
    [[{"assignees": ["Vadim Markovtsev"]}, {"reporters": ["Waren Long"]}],
     [["60s", "158s", "417s", "1102s", "2909s", "7676s", "20258s", "53456s", "141062s", "372237s",
       "982262s", "2591999s"],
      ["60s", "136s", "309s", "704s", "1601s", "3639s", "8271s", "18801s", "42732s", "97125s",
       "220753s", "501745s", "1140405s", "2591999s"]],
     [[60, 4, 18, 36, 76, 88, 42, 33, 31, 19, 81],
      [129, 3, 6, 9, 18, 23, 21, 32, 56, 43, 57, 46, 59]],
     [{"left": "3062s", "right": "194589s"}, {"left": "60s", "right": "364828s"}],
     ],
])
async def test_jira_histograms_smoke(client, headers, with_, ticks, frequencies, interquartile):
    for _ in range(2):
        body = {
            "histograms": [{
                "metric": JIRAMetricID.JIRA_LEAD_TIME,
                "scale": "log",
            }],
            **({"with": with_} if with_ is not None else {}),
            "date_from": "2015-10-13",
            "date_to": "2020-11-01",
            "exclude_inactive": False,
            "account": 1,
        }
        response = await client.request(
            method="POST", path="/v1/histograms/jira", headers=headers, json=body,
        )
        body = (await response.read()).decode("utf-8")
        assert response.status == 200, "Response body is : " + body
        body = FriendlyJson.loads(body)
        for item in body:
            CalculatedJIRAHistogram.from_dict(item)
        for histogram, hticks, hfrequencies, hinterquartile, hwith_ in zip(
                body, ticks, frequencies, interquartile, with_ or [None]):
            assert histogram == {
                "metric": JIRAMetricID.JIRA_LEAD_TIME,
                "scale": "log",
                "ticks": hticks,
                "frequencies": hfrequencies,
                "interquartile": hinterquartile,
                **({"with": hwith_} if hwith_ is not None else {}),
            }


async def test_jira_histogram_disabled_projects(client, headers, disabled_dev):
    body = {
        "histograms": [{
            "metric": JIRAMetricID.JIRA_LEAD_TIME,
            "scale": "log",
        }],
        "date_from": "2015-10-13",
        "date_to": "2020-11-01",
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    histogram = FriendlyJson.loads(body)[0]
    _check_histogram_no_dev_project(histogram)


def _check_histogram_no_dev_project(histogram: dict) -> None:
    assert histogram == {
        "metric": JIRAMetricID.JIRA_LEAD_TIME,
        "scale": "log",
        "ticks": ["60s", "128s", "275s", "590s", "1266s", "2714s", "5818s", "12470s", "26730s",
                  "57293s", "122803s", "263218s", "564186s", "1209285s", "2591999s"],
        "frequencies": [214, 3, 6, 20, 25, 31, 33, 33, 31, 55, 54, 55, 74, 222],
        "interquartile": {"left": "149s", "right": "1273387s"},
    }


async def test_jira_histogram_selected_projects(client, headers):
    body = {
        "histograms": [{
            "metric": JIRAMetricID.JIRA_LEAD_TIME,
            "scale": "log",
        }],
        "date_from": "2015-10-13",
        "date_to": "2020-11-01",
        "exclude_inactive": False,
        "projects": ["PRO", "OPS", "ENG", "GRW", "CS", "CON"],
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    histogram = FriendlyJson.loads(body)[0]
    _check_histogram_no_dev_project(histogram)


@pytest.mark.parametrize(
    "metric, date_to, bins, scale, ticks, quantiles, account, status",
    [
        (JIRAMetricID.JIRA_RAISED, "2020-01-23", 10, "log", None, [0, 1], 1, 400),
        (JIRAMetricID.JIRA_LEAD_TIME, "2020-01-23", -1, "log", None, [0, 1], 1, 400),
        (JIRAMetricID.JIRA_LEAD_TIME, "2020-01-23", 10, "xxx", None, [0, 1], 1, 400),
        (JIRAMetricID.JIRA_LEAD_TIME, "2015-01-23", 10, "linear", None, [0, 1], 1, 400),
        (JIRAMetricID.JIRA_LEAD_TIME, "2020-01-23", 10, "linear", None, [0, 1], 2, 422),
        (JIRAMetricID.JIRA_LEAD_TIME, "2020-01-23", 10, "linear", None, [0, 1], 4, 404),
        (JIRAMetricID.JIRA_LEAD_TIME, "2015-11-23", 10, "linear", None, [-1, 1], 1, 400),
        (JIRAMetricID.JIRA_LEAD_TIME, "2015-11-23", None, None, None, [0, 1], 1, 200),
        (JIRAMetricID.JIRA_LEAD_TIME, "2015-11-23", None, None, [], [0, 1], 1, 400),
    ],
)
async def test_jira_histograms_nasty_input(
        client, headers, metric, date_to, bins, scale, ticks, quantiles, account, status):
    body = {
        "histograms": [{
            "metric": metric,
            **({"scale": scale} if scale is not None else {}),
            **({"bins": bins} if bins is not None else {}),
            **({"ticks": ticks} if ticks is not None else {}),
        }],
        "date_from": "2015-10-13",
        "date_to": date_to,
        "quantiles": quantiles,
        "exclude_inactive": False,
        "account": account,
    }
    response = await client.request(
        method="POST", path="/v1/histograms/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == status, "Response body is : " + body
