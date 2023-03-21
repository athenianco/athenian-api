from datetime import datetime, timedelta, timezone
import json
from unittest import mock

from dateutil.tz import tzutc
import pytest
from sqlalchemy import delete, insert, update

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.jira.epic import filter_epics
from athenian.api.internal.settings import LogicalRepositorySettings
from athenian.api.models.metadata.github import NodePullRequestJiraIssues, PullRequest
from athenian.api.models.metadata.jira import Issue
from athenian.api.models.web import (
    CalculatedJIRAHistogram,
    DeployedComponent,
    DeploymentNotification,
    FilteredJIRAStuff,
    JIRAEpic,
    JIRAEpicChild,
    JIRAFilterReturn,
    JIRAIssueType,
    JIRALabel,
    JIRAMetricID,
    JIRAPriority,
    JIRAStatus,
    JIRAUser,
)
from athenian.api.serialization import FriendlyJson
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.wizards import (
    insert_repo,
    jira_issue_models,
    pr_jira_issue_mappings,
    pr_models,
)
from tests.testutils.time import dt


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize(
    "return_, checked",
    [
        (None, set(JIRAFilterReturn)),
        ([], set(JIRAFilterReturn) - {JIRAFilterReturn.ONLY_FLYING}),
        [*([list(set(JIRAFilterReturn) - {JIRAFilterReturn.ONLY_FLYING})] * 2)],
        ([JIRAFilterReturn.EPICS], {JIRAFilterReturn.EPICS}),
        [
            *(
                [
                    [
                        JIRAFilterReturn.EPICS,
                        JIRAFilterReturn.PRIORITIES,
                        JIRAFilterReturn.STATUSES,
                    ],
                ]
                * 2
            ),
        ],
        ([JIRAFilterReturn.ISSUES], ()),
        [
            *(
                [
                    [
                        JIRAFilterReturn.ISSUES,
                        JIRAFilterReturn.PRIORITIES,
                        JIRAFilterReturn.STATUSES,
                        JIRAFilterReturn.ISSUE_TYPES,
                        JIRAFilterReturn.USERS,
                        JIRAFilterReturn.LABELS,
                    ],
                ]
                * 2
            ),
        ],
        ([JIRAFilterReturn.ISSUES, JIRAFilterReturn.ISSUE_TYPES], {JIRAFilterReturn.ISSUE_TYPES}),
        [
            *(
                [
                    [
                        JIRAFilterReturn.ISSUES,
                        JIRAFilterReturn.ISSUE_TYPES,
                        JIRAFilterReturn.STATUSES,
                        JIRAFilterReturn.PRIORITIES,
                    ],
                ]
                * 2
            ),
        ],
    ],
)
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
            JIRALabel(
                title="API",
                last_used=datetime(2020, 7, 13, 17, 45, 58, tzinfo=tzutc()),
                issues_count=4,
                kind="component",
            ),
            JIRALabel(
                title="Webapp",
                last_used=datetime(2020, 7, 13, 17, 45, 58, tzinfo=tzutc()),
                issues_count=1,
                kind="component",
            ),
            JIRALabel(
                title="accounts",
                last_used=datetime(2020, 12, 15, 10, 16, 15, tzinfo=tzutc()),
                issues_count=1,
                kind="regular",
            ),
            JIRALabel(
                title="bug",
                last_used=datetime(2020, 6, 1, 7, 15, 7, tzinfo=tzutc()),
                issues_count=16,
                kind="regular",
            ),
            JIRALabel(
                title="code-quality",
                last_used=datetime(2020, 6, 4, 11, 35, 12, tzinfo=tzutc()),
                issues_count=1,
                kind="regular",
            ),
            JIRALabel(
                title="discarded",
                last_used=datetime(2020, 6, 1, 1, 27, 23, tzinfo=tzutc()),
                issues_count=4,
                kind="regular",
            ),
            JIRALabel(
                title="discussion",
                last_used=datetime(2020, 3, 31, 21, 16, 11, tzinfo=tzutc()),
                issues_count=3,
                kind="regular",
            ),
            JIRALabel(
                title="feature",
                last_used=datetime(2020, 4, 3, 18, 48, tzinfo=tzutc()),
                issues_count=6,
                kind="regular",
            ),
            JIRALabel(
                title="functionality",
                last_used=datetime(2020, 6, 4, 11, 35, 15, tzinfo=tzutc()),
                issues_count=1,
                kind="regular",
            ),
            JIRALabel(
                title="internal-story",
                last_used=datetime(2020, 6, 1, 7, 15, 7, tzinfo=tzutc()),
                issues_count=11,
                kind="regular",
            ),
            JIRALabel(
                title="needs-specs",
                last_used=datetime(2020, 4, 6, 13, 25, 2, tzinfo=tzutc()),
                issues_count=4,
                kind="regular",
            ),
            JIRALabel(
                title="onboarding",
                last_used=datetime(2020, 7, 13, 17, 45, 58, tzinfo=tzutc()),
                issues_count=1,
                kind="regular",
            ),
            JIRALabel(
                title="performance",
                last_used=datetime(2020, 3, 31, 21, 16, 5, tzinfo=tzutc()),
                issues_count=1,
                kind="regular",
            ),
            JIRALabel(
                title="user-story",
                last_used=datetime(2020, 4, 3, 18, 48, tzinfo=tzutc()),
                issues_count=5,
                kind="regular",
            ),
            JIRALabel(
                title="webapp",
                last_used=datetime(2020, 4, 3, 18, 47, 6, tzinfo=tzutc()),
                issues_count=1,
                kind="regular",
            ),
        ]
    else:
        assert model.labels == []
    if "epics" in checked:
        true_epics = [
            JIRAEpic(
                id="ENG-1",
                title="Evaluate our product and process internally",
                created=datetime(2019, 12, 2, 14, 19, 58, tzinfo=timezone.utc),
                updated=datetime(2020, 6, 1, 7, 19, 0, tzinfo=timezone.utc),
                work_began=datetime(2020, 6, 1, 7, 19, 0, tzinfo=timezone.utc),
                resolved=datetime(2020, 6, 1, 7, 19, 0, tzinfo=timezone.utc),
                lead_time=timedelta(0),
                life_time=timedelta(days=181, seconds=61141),
                reporter="Lou Marvin Caraig",
                assignee="Waren Long",
                comments=0,
                priority="Medium",
                status="Done",
                prs=0,
                project="10003",
                children=[],
                type="Epic",
                url="https://athenianco.atlassian.net/browse/ENG-1",
                story_points=None,
            ),
            JIRAEpic(
                id="DEV-70",
                title="Show the installation progress in the waiting page",
                created=datetime(2020, 1, 22, 16, 57, 10, tzinfo=timezone.utc),
                updated=datetime(2020, 7, 13, 17, 45, 58, tzinfo=timezone.utc),
                work_began=datetime(2020, 6, 2, 11, 40, 42, tzinfo=timezone.utc),
                resolved=datetime(2020, 7, 13, 17, 45, 58, tzinfo=timezone.utc),
                lead_time=timedelta(days=41, seconds=21915),
                life_time=timedelta(days=173, seconds=2928),
                reporter="Lou Marvin Caraig",
                assignee="David Pordomingo",
                url="https://athenianco.atlassian.net/browse/DEV-70",
                comments=8,
                priority="Low",
                status="Released",
                type="Epic",
                prs=0,
                project="10009",
                story_points=1.899999976158142,
                children=[
                    JIRAEpicChild(
                        id="DEV-183",
                        title="Implement the endpoint that returns the installation progress",
                        created=datetime(2020, 6, 2, 11, 7, 36, tzinfo=timezone.utc),
                        updated=datetime(2020, 6, 2, 11, 40, 42, tzinfo=timezone.utc),
                        work_began=datetime(2020, 6, 2, 11, 40, 42, tzinfo=timezone.utc),
                        resolved=datetime(2020, 6, 2, 11, 40, 42, tzinfo=timezone.utc),
                        lead_time=timedelta(0),
                        life_time=timedelta(seconds=1986),
                        reporter="Waren Long",
                        assignee="Vadim Markovtsev",
                        comments=1,
                        priority="Low",
                        status="Closed",
                        prs=0,
                        type="Task",
                        subtasks=0,
                        url="https://athenianco.atlassian.net/browse/DEV-183",
                        story_points=None,
                    ),
                    JIRAEpicChild(
                        id="DEV-228",
                        title=(
                            "Consider installation progress without updates during 3 hours "
                            "as complete"
                        ),
                        created=datetime(2020, 6, 8, 9, 0, 22, tzinfo=timezone.utc),
                        updated=datetime(2020, 6, 16, 18, 12, 38, tzinfo=timezone.utc),
                        work_began=datetime(2020, 6, 9, 10, 8, 15, tzinfo=timezone.utc),
                        resolved=datetime(2020, 6, 9, 10, 34, 7, tzinfo=timezone.utc),
                        lead_time=timedelta(seconds=1551),
                        life_time=timedelta(days=1, seconds=5624),
                        reporter="Vadim Markovtsev",
                        assignee="Vadim Markovtsev",
                        comments=1,
                        priority="Medium",
                        status="Released",
                        prs=0,
                        type="Task",
                        subtasks=0,
                        url="https://athenianco.atlassian.net/browse/DEV-228",
                        story_points=3.0,
                    ),
                    JIRAEpicChild(
                        id="DEV-315",
                        title=(
                            "Add a progress bar in the waiting page to show the installation"
                            " progress"
                        ),
                        created=datetime(2020, 6, 18, 21, 51, 19, tzinfo=timezone.utc),
                        updated=datetime(2020, 7, 27, 16, 56, 20, tzinfo=timezone.utc),
                        work_began=datetime(2020, 6, 25, 17, 8, 11, tzinfo=timezone.utc),
                        resolved=datetime(2020, 7, 13, 17, 43, 20, tzinfo=timezone.utc),
                        lead_time=timedelta(days=18, seconds=2109),
                        life_time=timedelta(days=24, seconds=71520),
                        reporter="Waren Long",
                        assignee="David Pordomingo",
                        comments=4,
                        priority="High",
                        status="Released",
                        prs=0,
                        type="Story",
                        subtasks=0,
                        url="https://athenianco.atlassian.net/browse/DEV-315",
                        story_points=None,
                    ),
                    JIRAEpicChild(
                        id="DEV-364",
                        title=(
                            "Block the access to the Overview page until the installation is 100%"
                            " complete"
                        ),
                        created=datetime(2020, 6, 25, 16, 12, 34, tzinfo=timezone.utc),
                        updated=datetime(2020, 7, 27, 16, 56, 22, tzinfo=timezone.utc),
                        work_began=datetime(2020, 7, 2, 4, 23, 20, tzinfo=timezone.utc),
                        resolved=datetime(2020, 7, 13, 17, 46, 15, tzinfo=timezone.utc),
                        lead_time=timedelta(days=11, seconds=48175),
                        life_time=timedelta(days=18, seconds=5621),
                        reporter="Waren Long",
                        assignee="David Pordomingo",
                        comments=1,
                        priority="Medium",
                        status="Released",
                        prs=0,
                        type="Story",
                        subtasks=0,
                        url="https://athenianco.atlassian.net/browse/DEV-364",
                        story_points=0.4000000059604645,
                    ),
                    JIRAEpicChild(
                        id="DEV-365",
                        title=(
                            "Design the success view telling the user the installation is complete"
                            " and the account ready to use"
                        ),
                        created=datetime(2020, 6, 25, 16, 17, 14, tzinfo=timezone.utc),
                        updated=datetime(2020, 6, 26, 9, 38, 36, tzinfo=timezone.utc),
                        work_began=datetime(2020, 6, 26, 9, 33, 9, tzinfo=timezone.utc),
                        resolved=datetime(2020, 6, 26, 9, 35, 43, tzinfo=timezone.utc),
                        lead_time=timedelta(seconds=154),
                        life_time=timedelta(seconds=62309),
                        reporter="Waren Long",
                        assignee="Zuri Negrin",
                        comments=2,
                        priority="Medium",
                        status="Released",
                        prs=0,
                        type="Story",
                        subtasks=0,
                        url="https://athenianco.atlassian.net/browse/DEV-365",
                        story_points=None,
                    ),
                ],
            ),
        ]
        assert model.epics == true_epics
    else:
        assert model.epics == []
    if "issue_types" in checked:
        assert model.issue_types == [
            JIRAIssueType(
                name="Design document",
                count=10,
                project="10003",
                is_subtask=False,
                normalized_name="designdocument",
                is_epic=False,
                image=(
                    "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                    "=10322&avatarType=issuetype"
                ),
            ),
            JIRAIssueType(
                name="Epic",
                count=1,
                project="10003",
                is_subtask=False,
                normalized_name="epic",
                is_epic=True,
                image=(
                    "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                    "=10307&avatarType=issuetype"
                ),
            ),
            JIRAIssueType(
                name="Epic",
                count=1,
                is_epic=True,
                image="https://athenianco.atlassian.net/images/icons/issuetypes/epic.svg",
                project="10009",
                is_subtask=False,
                normalized_name="epic",
            ),
            JIRAIssueType(
                name="Story",
                count=49,
                project="10003",
                is_subtask=False,
                normalized_name="story",
                is_epic=False,
                image=(
                    "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                    "=10315&avatarType=issuetype"
                ),
            ),
            JIRAIssueType(
                name="Subtask",
                count=98,
                project="10003",
                is_subtask=True,
                normalized_name="subtask",
                is_epic=False,
                image=(
                    "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                    "=10316&avatarType=issuetype"
                ),
            ),
            JIRAIssueType(
                name="Task",
                count=4,
                project="10003",
                is_subtask=False,
                normalized_name="task",
                is_epic=False,
                image=(
                    "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                    "=10318&avatarType=issuetype"
                ),
            ),
            JIRAIssueType(
                name="Task",
                count=5,
                is_epic=False,
                image=(
                    "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                    "=10318&avatarType=issuetype"
                ),
                project="10009",
                is_subtask=False,
                normalized_name="task",
            ),
        ]
    else:
        assert model.issue_types == []
    if "users" in checked:
        assert model.users == [
            JIRAUser(
                name="David Pordomingo",
                avatar=(
                    "https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/"
                    "initials/DP-4.png"
                ),
                type="atlassian",
            ),
            JIRAUser(
                name="Denys Smirnov",
                avatar=(
                    "https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/"
                    "initials/DS-1.png"
                ),
                type="atlassian",
            ),
            JIRAUser(
                name="Kuba Podgórski",
                avatar=(
                    "https://secure.gravatar.com/avatar/ec2f95fe07b5ffec5cde78781f433b68?d="
                    "https%3A%2F%2Favatar-management--avatars.us-west-2.prod.public.atl-paas"
                    ".net%2Finitials%2FKP-3.png"
                ),
                type="atlassian",
            ),
            JIRAUser(
                name="Lou Marvin Caraig",
                avatar=(
                    "https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net"
                    "/initials/LC-0.png"
                ),
                type="atlassian",
            ),
            JIRAUser(
                name="Marcelo Novaes",
                avatar=(
                    "https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/"
                    "initials/MN-4.png"
                ),
                type="atlassian",
            ),
            JIRAUser(
                name="Oleksandr Chabaiev",
                avatar=(
                    "https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/"
                    "initials/OC-5.png"
                ),
                type="atlassian",
            ),
            JIRAUser(
                name="Vadim Markovtsev",
                avatar=(
                    "https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/"
                    "initials/VM-6.png"
                ),
                type="atlassian",
            ),
            JIRAUser(
                name="Waren Long",
                avatar=(
                    "https://avatar-management--avatars.us-west-2.prod.public.atl-paas.net/"
                    "initials/WL-5.png"
                ),
                type="atlassian",
            ),
        ]
    else:
        assert model.users == []
    if "priorities" in checked:
        true_priorities = [
            JIRAPriority(
                name="High",
                image="https://athenianco.atlassian.net/images/icons/priorities/high.svg",
                rank=2,
                color="EA4444",
            ),
            JIRAPriority(
                name="Medium",
                image="https://athenianco.atlassian.net/images/icons/priorities/medium.svg",
                rank=3,
                color="EA7D24",
            ),
            JIRAPriority(
                name="Low",
                image="https://athenianco.atlassian.net/images/icons/priorities/low.svg",
                rank=4,
                color="2A8735",
            ),
            JIRAPriority(
                name="None",
                image="https://athenianco.atlassian.net/images/icons/priorities/trivial.svg",
                rank=6,
                color="9AA1B2",
            ),
        ]
        if "issues" not in checked:
            true_priorities.pop(-1)
        if "epics" not in checked:
            # some children of the filtered epics do not belong to the given time interval
            true_priorities.pop(0)
        assert model.priorities == true_priorities
    else:
        assert model.priorities == []
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
        assert model.statuses == []
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
    assert len(model.epics) == 60
    assert len(model.priorities) == 6
    assert len(model.statuses) == 7


@with_defer
async def test_missing_release_settings(client, headers, mdb_rw, sdb):
    body = {
        "date_from": "2023-02-01",
        "date_to": "2023-03-01",
        "timezone": 0,
        "account": 1,
        "exclude_inactive": True,
        "return": ["issues", "users"],
    }

    issue_kwargs = {"project_id": "1", "created": dt(2023, 1, 1)}
    pr_kwargs = {"repository_full_name": "org/repo", "created_at": dt(2023, 1, 1)}
    async with DBCleaner(mdb_rw) as mdb_cleaner:
        repo = md_factory.RepositoryFactory(node_id=99, full_name="org/repo")
        await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
        models = [
            md_factory.JIRAProjectFactory(id="1", key="P1"),
            md_factory.JIRAUserFactory(id="u0", display_name="U 0", avatar_url="http://a.co/0"),
            *jira_issue_models("1", resolved=dt(2023, 1, 30), assignee_id="u0", **issue_kwargs),
            *pr_models(99, 1, 1, closed_at=dt(2023, 2, 2), **pr_kwargs),
            *pr_jira_issue_mappings((1, "1"), (4, "4")),
        ]
        mdb_cleaner.add_models(*models)
        await models_insert(mdb_rw, *models)

        # let precompute, mapped prs are read only from pdb
        search_body = {"date_from": "2023-02-01", "date_to": "2023-03-01", "account": 1}
        await client.request(
            method="POST", path="/private/search/pull_requests", headers=headers, json=search_body,
        )
        await wait_deferred()
        response = await client.request(
            method="POST", path="/v1/filter/jira", headers=headers, json=body,
        )
    assert response.status == 200
    res_body = await response.json()
    assert len(res_body["users"]) == 1
    assert res_body["users"][0]["name"] == "U 0"


@with_defer
async def test_with_cache(client, headers, client_cache):
    body = {
        "date_from": "2020-06-20",
        "date_to": "2020-07-01",
        "timezone": 120,
        "account": 1,
        "exclude_inactive": True,
        "return": ["epics", "priorities", "statuses"],
    }

    # TOFIX: BranchMiner.load_branches gives inconsistent results for
    # first call (!) and breaks filter_epics caching
    response = await client.request(
        method="POST", path="/v1/filter/jira", headers=headers, json=body,
    )
    assert response.status == 200

    with mock.patch(
        "athenian.api.controllers.jira_controller.filter_epics", wraps=filter_epics,
    ) as filter_epics_mock:
        response = await client.request(
            method="POST", path="/v1/filter/jira", headers=headers, json=body,
        )
        filter_epics_mock.assert_called_once()
        res_body = await response.json()
        assert response.status == 200
        assert len(res_body["epics"]) == 16

        await wait_deferred()

        response = await client.request(
            method="POST", path="/v1/filter/jira", headers=headers, json=body,
        )
        # cache HIT, filter_epics not called again
        filter_epics_mock.assert_called_once()
        res_body = await response.json()
        assert response.status == 200
        assert len(res_body["epics"]) == 16


@pytest.mark.parametrize("ikey", ("DEV-162", "DEV-163"))
async def test_filter_jira_epics_deleted(client, headers, ikey, mdb_rw):
    mdb = mdb_rw
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


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize(
    "exclude_inactive, labels, epics, types, users, priorities",
    [
        [
            False,
            32,
            13,
            [
                JIRAIssueType(
                    name="Bug",
                    count=2,
                    is_epic=False,
                    image=(
                        "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                        "=10303&avatarType=issuetype"
                    ),
                    project="10003",
                    is_subtask=False,
                    normalized_name="bug",
                ),
                JIRAIssueType(
                    name="Bug",
                    count=88,
                    project="10009",
                    is_subtask=False,
                    normalized_name="bug",
                    is_epic=False,
                    image=(
                        "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                        "=10303&avatarType=issuetype"
                    ),
                ),
                JIRAIssueType(
                    name="Design Document",
                    count=4,
                    project="10009",
                    is_subtask=False,
                    normalized_name="designdocument",
                    is_epic=False,
                    image=(
                        "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                        "=10322&avatarType=issuetype"
                    ),
                ),
                JIRAIssueType(
                    name="Epic",
                    count=2,
                    is_epic=True,
                    image=(
                        "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                        "=10307&avatarType=issuetype"
                    ),
                    project="10003",
                    is_subtask=False,
                    normalized_name="epic",
                ),
                JIRAIssueType(
                    name="Epic",
                    count=11,
                    project="10009",
                    is_subtask=False,
                    normalized_name="epic",
                    is_epic=True,
                    image="https://athenianco.atlassian.net/images/icons/issuetypes/epic.svg",
                ),
                JIRAIssueType(
                    name="Incident",
                    count=2,
                    project="10009",
                    is_subtask=False,
                    normalized_name="incident",
                    is_epic=False,
                    image=(
                        "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                        "=10304&avatarType=issuetype"
                    ),
                ),
                JIRAIssueType(
                    name="Story",
                    count=25,
                    project="10009",
                    is_subtask=False,
                    normalized_name="story",
                    is_epic=False,
                    image="https://athenianco.atlassian.net/images/icons/issuetypes/story.svg",
                ),
                JIRAIssueType(
                    name="Subtask",
                    count=1,
                    project="10003",
                    is_subtask=True,
                    normalized_name="subtask",
                    is_epic=False,
                    image=(
                        "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                        "=10316&avatarType=issuetype"
                    ),
                ),
                JIRAIssueType(
                    name="Sub-task",
                    count=20,
                    project="10009",
                    is_subtask=True,
                    normalized_name="subtask",
                    is_epic=False,
                    image=(
                        "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                        "=10316&avatarType=issuetype"
                    ),
                ),
                JIRAIssueType(
                    name="Task",
                    count=166,
                    project="10009",
                    is_subtask=False,
                    normalized_name="task",
                    is_epic=False,
                    image=(
                        "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                        "=10318&avatarType=issuetype"
                    ),
                ),
            ],
            14,
            6,
        ],
        [
            True,
            31,
            11,
            [
                JIRAIssueType(
                    name="Bug",
                    count=1,
                    is_epic=False,
                    image=(
                        "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                        "=10303&avatarType=issuetype"
                    ),
                    project="10003",
                    is_subtask=False,
                    normalized_name="bug",
                ),
                JIRAIssueType(
                    name="Bug",
                    count=78,
                    project="10009",
                    is_subtask=False,
                    normalized_name="bug",
                    is_epic=False,
                    image=(
                        "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                        "=10303&avatarType=issuetype"
                    ),
                ),
                JIRAIssueType(
                    name="Design Document",
                    count=1,
                    project="10009",
                    is_subtask=False,
                    normalized_name="designdocument",
                    is_epic=False,
                    image=(
                        "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                        "=10322&avatarType=issuetype"
                    ),
                ),
                JIRAIssueType(
                    name="Epic",
                    count=11,
                    project="10009",
                    is_subtask=False,
                    normalized_name="epic",
                    is_epic=True,
                    image="https://athenianco.atlassian.net/images/icons/issuetypes/epic.svg",
                ),
                JIRAIssueType(
                    name="Incident",
                    count=2,
                    project="10009",
                    is_subtask=False,
                    normalized_name="incident",
                    is_epic=False,
                    image=(
                        "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                        "=10304&avatarType=issuetype"
                    ),
                ),
                JIRAIssueType(
                    name="Story",
                    count=13,
                    project="10009",
                    is_subtask=False,
                    normalized_name="story",
                    is_epic=False,
                    image="https://athenianco.atlassian.net/images/icons/issuetypes/story.svg",
                ),
                JIRAIssueType(
                    name="Sub-task",
                    count=20,
                    project="10009",
                    is_subtask=True,
                    normalized_name="subtask",
                    is_epic=False,
                    image=(
                        "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                        "=10316&avatarType=issuetype"
                    ),
                ),
                JIRAIssueType(
                    name="Task",
                    count=128,
                    is_epic=False,
                    image=(
                        "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                        "=10318&avatarType=issuetype"
                    ),
                    project="10009",
                    is_subtask=False,
                    normalized_name="task",
                ),
            ],
            12,
            6,
        ],
    ],
)
async def test_filter_jira_exclude_inactive(
    client,
    headers,
    exclude_inactive,
    labels,
    epics,
    types,
    users,
    priorities,
):
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


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
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


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
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
        JIRALabel(
            title="accounts",
            last_used=datetime(2020, 12, 15, 10, 16, 15, tzinfo=tzutc()),
            issues_count=1,
            kind="regular",
        ),
        JIRALabel(
            title="bug",
            last_used=datetime(2020, 6, 1, 7, 15, 7, tzinfo=tzutc()),
            issues_count=16,
            kind="regular",
        ),
        JIRALabel(
            title="discarded",
            last_used=datetime(2020, 6, 1, 1, 27, 23, tzinfo=tzutc()),
            issues_count=4,
            kind="regular",
        ),
        JIRALabel(
            title="discussion",
            last_used=datetime(2020, 3, 31, 21, 16, 11, tzinfo=tzutc()),
            issues_count=3,
            kind="regular",
        ),
        JIRALabel(
            title="feature",
            last_used=datetime(2020, 4, 3, 18, 48, tzinfo=tzutc()),
            issues_count=6,
            kind="regular",
        ),
        JIRALabel(
            title="internal-story",
            last_used=datetime(2020, 6, 1, 7, 15, 7, tzinfo=tzutc()),
            issues_count=11,
            kind="regular",
        ),
        JIRALabel(
            title="needs-specs",
            last_used=datetime(2020, 4, 6, 13, 25, 2, tzinfo=tzutc()),
            issues_count=4,
            kind="regular",
        ),
        JIRALabel(
            title="performance",
            last_used=datetime(2020, 3, 31, 21, 16, 5, tzinfo=tzutc()),
            issues_count=1,
            kind="regular",
        ),
        JIRALabel(
            title="user-story",
            last_used=datetime(2020, 4, 3, 18, 48, tzinfo=tzutc()),
            issues_count=5,
            kind="regular",
        ),
        JIRALabel(
            title="webapp",
            last_used=datetime(2020, 4, 3, 18, 47, 6, tzinfo=tzutc()),
            issues_count=1,
            kind="regular",
        ),
    ]
    assert model.epics == [
        JIRAEpic(
            id="ENG-1",
            title="Evaluate our product and process internally",
            created=datetime(2019, 12, 2, 14, 19, 58, tzinfo=timezone.utc),
            updated=datetime(2020, 6, 1, 7, 19, 0, tzinfo=timezone.utc),
            work_began=datetime(2020, 6, 1, 7, 19, 0, tzinfo=timezone.utc),
            resolved=datetime(2020, 6, 1, 7, 19, 0, tzinfo=timezone.utc),
            lead_time=timedelta(0),
            life_time=timedelta(days=181, seconds=61141),
            reporter="Lou Marvin Caraig",
            assignee="Waren Long",
            type="Epic",
            comments=0,
            priority="Medium",
            status="Done",
            prs=0,
            project="10003",
            children=[],
            url="https://athenianco.atlassian.net/browse/ENG-1",
        ),
    ]
    assert model.issue_types == [
        JIRAIssueType(
            name="Design document",
            count=10,
            project="10003",
            is_subtask=False,
            normalized_name="designdocument",
            is_epic=False,
            image=(
                "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                "=10322&avatarType=issuetype"
            ),
        ),
        JIRAIssueType(
            name="Epic",
            count=1,
            project="10003",
            is_subtask=False,
            normalized_name="epic",
            is_epic=True,
            image=(
                "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                "=10307&avatarType=issuetype"
            ),
        ),
        JIRAIssueType(
            name="Story",
            count=49,
            project="10003",
            is_subtask=False,
            normalized_name="story",
            is_epic=False,
            image=(
                "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                "=10315&avatarType=issuetype"
            ),
        ),
        JIRAIssueType(
            name="Subtask",
            count=98,
            project="10003",
            is_subtask=True,
            normalized_name="subtask",
            is_epic=False,
            image=(
                "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                "=10316&avatarType=issuetype"
            ),
        ),
        JIRAIssueType(
            name="Task",
            count=4,
            project="10003",
            is_subtask=False,
            normalized_name="task",
            is_epic=False,
            image=(
                "https://athenianco.atlassian.net/secure/viewavatar?size=medium&avatarId"
                "=10318&avatarType=issuetype"
            ),
        ),
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
    assert len(model.epics) == 38, str(sorted(epic.id for epic in model.epics))
    assert len(model.priorities) == 6  # two projects
    assert len(model.statuses) == 7


async def test_filter_jira_issue_types_filter(client, headers, metadata_db):
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
    if metadata_db.startswith("sqlite://"):
        body["return"].append("epics")  # hack
    response = await client.request(
        method="POST", path="/v1/filter/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = FilteredJIRAStuff.from_dict(json.loads(body))
    assert len(model.issue_types) == 2
    assert not model.issues


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
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
    assert len(model.issues) == 323
    prs = 0
    comments = 0
    for issue in model.issues:
        prs += bool(issue.prs)
        comments += issue.comments
        for pr in issue.prs or []:
            assert pr.number > 0
            assert not pr.jira
    assert prs == 5
    assert comments == 872
    assert not model.deployments


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@pytest.mark.parametrize("with_pdb", [True])
@with_defer
async def test_filter_jira_issue_prs_deployments(
    client,
    headers,
    mdb_rw,
    precomputed_deployments,
    with_pdb,
    pr_facts_calculator_factory,
    prefixer,
    release_match_setting_tag,
    bots,
):
    body = {
        "date_from": "2018-09-01",
        "date_to": "2020-01-01",
        "timezone": 120,
        "account": 1,
        "exclude_inactive": True,
        "return": ["issues", "issue_bodies"],
    }
    if with_pdb:
        args = (
            datetime(2019, 6, 3, tzinfo=timezone.utc),
            datetime(2019, 6, 19, tzinfo=timezone.utc),
            {"src-d/go-git"},
            {},
            LabelFilter.empty(),
            JIRAFilter.empty(),
            False,
            bots,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            prefixer,
            False,
            0,
        )
        await pr_facts_calculator_factory(1, (6366825,))(*args)
        await wait_deferred()
    await mdb_rw.execute_many(
        insert(NodePullRequestJiraIssues),
        [
            {
                NodePullRequestJiraIssues.jira_acc.name: 1,
                NodePullRequestJiraIssues.node_acc.name: 6366825,
                NodePullRequestJiraIssues.node_id.name: 163373,
                NodePullRequestJiraIssues.jira_id.name: "10100",
            },
            {
                NodePullRequestJiraIssues.jira_acc.name: 1,
                NodePullRequestJiraIssues.node_acc.name: 6366825,
                NodePullRequestJiraIssues.node_id.name: 163221,
                NodePullRequestJiraIssues.jira_id.name: "10100",
            },
        ],
    )
    try:
        response = await client.request(
            method="POST", path="/v1/filter/jira", headers=headers, json=body,
        )
    finally:
        await mdb_rw.execute(
            delete(NodePullRequestJiraIssues).where(
                NodePullRequestJiraIssues.node_id.in_([163373, 163221]),
            ),
        )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body
    model = FilteredJIRAStuff.from_dict(json.loads(body))
    assert len(model.issues) == 98
    prs = 0
    for issue in model.issues:
        if issue.id == "DEV-100":
            prs += 1
            assert issue.prs and len(issue.prs) == 2
            assert {issue.prs[i].number for i in range(2)} == {1160, 880}
    assert prs == 1
    assert model.deployments == {
        "Dummy deployment": DeploymentNotification(
            components=[
                DeployedComponent(
                    repository="github.com/src-d/go-git",
                    reference="v4.13.1 (0d1a009cbb604db18be960db5f1525b99a55d727)",
                ),
            ],
            environment="production",
            name="Dummy deployment",
            url=None,
            date_started=datetime(2019, 11, 1, 12, 0, tzinfo=timezone.utc),
            date_finished=datetime(2019, 11, 1, 12, 15, tzinfo=timezone.utc),
            conclusion="SUCCESS",
            labels=None,
        ),
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_filter_jira_issue_prs_logical(
    client,
    headers,
    logical_settings_db,
    release_match_setting_tag_logical_db,
):
    body = {
        "date_from": "2019-09-01",
        "date_to": "2022-01-01",
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
    assert len(model.issues) == 1797
    prs = {}
    for issue in model.issues:
        for pr in issue.prs or []:
            prs[pr.repository] = prs.setdefault(pr.repository, 0) + 1
    assert prs == {
        "github.com/src-d/go-git": 24,
        "github.com/src-d/go-git/beta": 21,
        "github.com/src-d/go-git/alpha": 17,
    }


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
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
    assert len(model.issues) == 189


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_filter_jira_issue_disabled(client, headers, mdb_rw):
    ikey = "ENG-303"
    mdb = mdb_rw
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
        assert len(model.issues) == 323 - 1
    finally:
        await mdb.execute(update(Issue).where(Issue.key == ikey).values({Issue.is_deleted: False}))


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_filter_jira_deleted_repositories(client, headers, mdb_rw):
    # DEV-2082
    mdb = mdb_rw
    await mdb.execute(
        insert(NodePullRequestJiraIssues).values(
            node_id=1234,
            node_acc=6366825,
            jira_acc=1,
            jira_id="12541",
        ),
    )
    await mdb.execute(
        insert(PullRequest).values(
            node_id=1234,
            acc_id=6366825,
            repository_full_name="athenianco/athenian-api",
            repository_node_id=4321,
            base_ref="base_ref",
            head_ref="head_ref",
            number=100500,
            closed=True,
            additions=0,
            deletions=0,
            changed_files=0,
            commits=1,
            user_node_id=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
    )
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
        assert len(model.issues) == 46
        assert prs == 0
    finally:
        await mdb.execute(
            delete(NodePullRequestJiraIssues).where(NodePullRequestJiraIssues.node_id == 1234),
        )
        await mdb.execute(delete(PullRequest).where(PullRequest.node_id == 1234))


@pytest.mark.parametrize(
    "account, date_to, tz, status",
    [
        (1, "2015-10-12", 0, 400),
        (1, None, 0, 400),
        (2, "2020-10-12", 0, 422),
        (3, "2020-10-12", 0, 404),
        (1, "2020-10-12", 100500, 400),
    ],
)
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


@pytest.mark.parametrize(
    "with_, ticks, frequencies, interquartile",
    [
        [
            None,
            [
                [
                    "60s",
                    "122s",
                    "249s",
                    "507s",
                    "1033s",
                    "2105s",
                    "4288s",
                    "8737s",
                    "17799s",
                    "36261s",
                    "73870s",
                    "150489s",
                    "306576s",
                    "624554s",
                    "1272338s",
                    "2591999s",
                ],
            ],
            [[351, 7, 12, 27, 38, 70, 95, 103, 68, 76, 120, 116, 132, 114, 285]],
            [{"left": "1255s", "right": "618082s"}],
        ],
        [
            [{"assignees": ["Vadim Markovtsev"]}, {"reporters": ["Waren Long"]}],
            [
                [
                    "60s",
                    "158s",
                    "417s",
                    "1102s",
                    "2909s",
                    "7676s",
                    "20258s",
                    "53456s",
                    "141062s",
                    "372237s",
                    "982262s",
                    "2591999s",
                ],
                [
                    "60s",
                    "136s",
                    "309s",
                    "704s",
                    "1601s",
                    "3639s",
                    "8271s",
                    "18801s",
                    "42732s",
                    "97125s",
                    "220753s",
                    "501745s",
                    "1140405s",
                    "2591999s",
                ],
            ],
            [
                [60, 4, 18, 36, 76, 88, 42, 33, 31, 19, 81],
                [129, 3, 6, 9, 18, 23, 21, 32, 56, 43, 57, 46, 59],
            ],
            [
                {"left": "3062s", "right": "194589s"},
                {"left": "60s", "right": "364828s"},
            ],
        ],
    ],
)
async def test_jira_histograms_smoke(client, headers, with_, ticks, frequencies, interquartile):
    for _ in range(2):
        body = {
            "histograms": [
                {
                    "metric": JIRAMetricID.JIRA_LEAD_TIME,
                    "scale": "log",
                },
            ],
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
            body, ticks, frequencies, interquartile, with_ or [None],
        ):
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
        "histograms": [
            {
                "metric": JIRAMetricID.JIRA_LEAD_TIME,
                "scale": "log",
            },
        ],
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
        "ticks": [
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
        "frequencies": [214, 3, 6, 20, 25, 31, 33, 33, 31, 55, 54, 55, 74, 222],
        "interquartile": {"left": "149s", "right": "1273387s"},
    }


async def test_jira_histogram_selected_projects(client, headers):
    body = {
        "histograms": [
            {
                "metric": JIRAMetricID.JIRA_LEAD_TIME,
                "scale": "log",
            },
        ],
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
    client,
    headers,
    metric,
    date_to,
    bins,
    scale,
    ticks,
    quantiles,
    account,
    status,
):
    body = {
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
        method="POST", path="/v1/histograms/jira", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == status, "Response body is : " + body
