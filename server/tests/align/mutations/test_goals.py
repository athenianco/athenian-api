from datetime import datetime, timezone
from typing import Any, Optional

from freezegun import freeze_time
import pytest
import sqlalchemy as sa

from athenian.api.align.goals.templates import TEMPLATES_COLLECTION
from athenian.api.align.models import (
    CreateGoalInputFields,
    TeamGoalChangeFields,
    TeamGoalInputFields,
    UpdateGoalInputFields,
    UpdateRepositoriesInputFields,
)
from athenian.api.db import Database, ensure_db_datetime_tz
from athenian.api.models.state.models import Goal, Team, TeamGoal
from athenian.api.models.web import PullRequestMetricID
from tests.align.utils import (
    align_graphql_request,
    assert_extension_error,
    get_extension_error_obj,
)
from tests.conftest import DEFAULT_USER_ID
from tests.testutils.auth import force_request_auth
from tests.testutils.db import (
    assert_existing_row,
    assert_missing_row,
    model_insert_stmt,
    models_insert,
)
from tests.testutils.factory.state import (
    AccountFactory,
    GoalFactory,
    GodFactory,
    TeamFactory,
    TeamGoalFactory,
    UserAccountFactory,
)
from tests.testutils.requester import Requester
from tests.testutils.time import dt

_USER_ID = "github|1"


@pytest.fixture(scope="function")
async def _create_user(sdb):
    await sdb.execute(model_insert_stmt(UserAccountFactory(user_id=_USER_ID)))


@pytest.mark.usefixtures("_create_user")
class BaseGoalTest(Requester):
    _TEMPLATE_NAME = "T0"
    _TEMPLATE_METRIC = PullRequestMetricID.PR_REVIEW_TIME

    @classmethod
    async def _assert_no_goal_exists(cls, sdb: Database) -> None:
        assert await sdb.fetch_one(sa.select(Goal)) is None
        assert await sdb.fetch_one(sa.select(TeamGoal)) is None

    async def _base_request(self, json, user_id: Optional[str]) -> dict:
        with force_request_auth(user_id, self.headers) as headers:
            return await align_graphql_request(self.client, headers=headers, json=json)


class BaseCreateGoalTest(BaseGoalTest):
    async def _request(self, variables: dict, user_id: Optional[str] = _USER_ID) -> dict:
        body = {"query": self._QUERY, "variables": variables}
        return await self._base_request(body, user_id)

    _QUERY = """
        mutation ($accountId: Int!, $createGoalInput: CreateGoalInput!) {
          createGoal(accountId: $accountId, input: $createGoalInput) {
            goal {
              id
            }
          }
        }
    """

    @classmethod
    def _mk_input(cls, **kwargs: Any) -> dict:
        template = TEMPLATES_COLLECTION[0]
        kwargs.setdefault(CreateGoalInputFields.name, template[CreateGoalInputFields.name])
        kwargs.setdefault(CreateGoalInputFields.metric, template[CreateGoalInputFields.metric])
        kwargs.setdefault(CreateGoalInputFields.validFrom, "2022-01-01")
        kwargs.setdefault(CreateGoalInputFields.expiresAt, "2022-03-31")
        kwargs.setdefault(CreateGoalInputFields.teamGoals, [])
        return kwargs


class TestCreateGoalErrors(BaseCreateGoalTest):
    async def test_invalid_metric(self, sdb: Database) -> None:
        variables = {
            "createGoalInput": self._mk_input(metric="xxx-cuckold"),
            "accountId": 1,
        }
        res = await self._request(variables)
        assert_extension_error(res, 'Unsupported metric "xxx-cuckold"')
        await self._assert_no_goal_exists(sdb)

    async def test_invalid_date(self, sdb: Database) -> None:
        variables = {"createGoalInput": self._mk_input(validFrom="not-date"), "accountId": 1}
        res = await self._request(variables)
        assert "not-date" in res["errors"][0]["message"]
        await self._assert_no_goal_exists(sdb)

    async def test_missing_date(self, sdb: Database) -> None:
        variables = {
            "createGoalInput": self._mk_input(expiresAt=None),
            "accountId": 1,
        }
        res = await self._request(variables)
        assert "expiresAt" in res["errors"][0]["message"]
        await self._assert_no_goal_exists(sdb)

    async def test_account_mismatch(self, sdb: Database) -> None:
        variables = {"createGoalInput": self._mk_input(), "accountId": 3}
        res = await self._request(variables)
        assert "data" not in res
        await self._assert_no_goal_exists(sdb)

    async def test_no_team_goals(self, sdb: Database) -> None:
        variables = {"createGoalInput": self._mk_input(teamGoals=[]), "accountId": 1}
        res = await self._request(variables)
        assert "data" not in res
        assert_extension_error(res, "At least one teamGoals is required")
        await self._assert_no_goal_exists(sdb)

    async def test_more_goals_for_same_team(self, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(owner_id=1, id=10)))

        team_goals = [
            {TeamGoalInputFields.teamId: 10, TeamGoalInputFields.target: {"int": 42}},
            {TeamGoalInputFields.teamId: 10, TeamGoalInputFields.target: {"int": 44}},
        ]
        variables = {
            "createGoalInput": self._mk_input(teamGoals=team_goals),
            "accountId": 1,
        }
        res = await self._request(variables)
        assert "data" not in res
        assert_extension_error(res, "More than one team goal with the same teamId")
        await self._assert_no_goal_exists(sdb)

    async def test_unexisting_team(self, sdb: Database) -> None:
        variables = {
            "createGoalInput": self._mk_input(
                teamGoals={
                    TeamGoalInputFields.teamId: 10,
                    TeamGoalInputFields.target: {"int": 10},
                },
            ),
            "accountId": 1,
        }
        res = await self._request(variables)
        assert "data" not in res
        assert_extension_error(res, "Some teams don't exist or access denied: 10")
        await self._assert_no_goal_exists(sdb)

    async def test_goals_for_other_account_team(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10), TeamFactory(owner_id=2, id=20))

        team_goals = [
            {TeamGoalInputFields.teamId: 10, TeamGoalInputFields.target: {"int": 42}},
            {TeamGoalInputFields.teamId: 20, TeamGoalInputFields.target: {"int": 44}},
        ]
        variables = {
            "createGoalInput": self._mk_input(teamGoals=team_goals),
            "accountId": 1,
        }
        res = await self._request(variables)
        assert "data" not in res
        assert_extension_error(res, "Some teams don't exist or access denied: 20")
        await self._assert_no_goal_exists(sdb)

    async def test_inverted_dates(self, sdb: Database):
        await sdb.execute(model_insert_stmt(TeamFactory(owner_id=1, id=10)))

        team_goals = [{TeamGoalInputFields.teamId: 10, TeamGoalInputFields.target: {"int": 42}}]

        variables = {
            "createGoalInput": self._mk_input(
                validFrom="2022-04-01",
                expiresAt="2022-01-01",
                teamGoals=team_goals,
            ),
            "accountId": 1,
        }
        res = await self._request(variables)
        assert "data" not in res
        assert_extension_error(
            res, f"Goal {CreateGoalInputFields.expiresAt} cannot precede validFrom",
        )
        await assert_missing_row(sdb, Goal, account_id=1)

    async def test_default_user_forbidden(self, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(owner_id=1, id=10)))

        team_goals = [{TeamGoalInputFields.teamId: 10, TeamGoalInputFields.target: {"int": 42}}]

        variables = {
            "createGoalInput": self._mk_input(teamGoals=team_goals),
            "accountId": 1,
        }
        res = await self._request(variables, None)
        assert "data" not in res

        assert get_extension_error_obj(res)["type"] == "/errors/ForbiddenError"
        await assert_missing_row(sdb, Goal, account_id=1)

    async def test_bad_repositories(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(owner_id=1, id=10),
        )
        variables = {
            "createGoalInput": self._mk_input(
                teamGoals=[
                    {TeamGoalInputFields.teamId: 10, TeamGoalInputFields.target: {"int": 42}},
                ],
                repositories=["github.com/athenianco/xxx"],
            ),
            "accountId": 1,
        }
        res = await self._request(variables)
        assert "repository" in res["errors"][0]["extensions"]["detail"]
        await self._assert_no_goal_exists(sdb)

    async def test_invalid_goal_metric_params(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10))
        team_goals = [{TeamGoalInputFields.teamId: 10, TeamGoalInputFields.target: {"int": 42}}]
        variables = {
            "createGoalInput": self._mk_input(teamGoals=team_goals, metricParams="123"),
            "accountId": 1,
        }
        res = await self._request(variables)
        assert "metricParams" in res["errors"][0]["message"]
        await self._assert_no_goal_exists(sdb)

    async def test_invalid_team_goal_metric_params(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10))
        team_goals = [{"teamId": 10, "target": {"int": 42}, "metricParams": False}]
        variables = {
            "createGoalInput": self._mk_input(teamGoals=team_goals),
            "accountId": 1,
        }
        res = await self._request(variables)
        assert "teamGoals[0].metricParams" in res["errors"][0]["message"]
        await self._assert_no_goal_exists(sdb)


class TestCreateGoals(BaseCreateGoalTest):
    async def test_create_single_team_goal(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10))

        team_goals = [{TeamGoalInputFields.teamId: 10, TeamGoalInputFields.target: {"int": 42}}]

        variables = {
            "createGoalInput": self._mk_input(
                name="G0",
                metric=PullRequestMetricID.PR_COMMENTS_PER,
                validFrom="2022-01-01",
                expiresAt="2022-12-31",
                teamGoals=team_goals,
            ),
            "accountId": 1,
        }

        new_goal_id = await self._create(variables)

        goal_row = await assert_existing_row(
            sdb,
            Goal,
            id=new_goal_id,
            account_id=1,
            name="G0",
            metric=PullRequestMetricID.PR_COMMENTS_PER,
        )
        assert goal_row[Goal.repositories.name] is None
        assert goal_row[Goal.metric_params.name] is None
        assert ensure_db_datetime_tz(goal_row[Goal.valid_from.name], sdb) == datetime(
            2022, 1, 1, tzinfo=timezone.utc,
        )

        # expires_at is moved to the day after the one received in api
        assert ensure_db_datetime_tz(goal_row[Goal.expires_at.name], sdb) == datetime(
            2023, 1, 1, tzinfo=timezone.utc,
        )

        tg_row = await assert_existing_row(sdb, TeamGoal, goal_id=new_goal_id, team_id=10)
        assert tg_row[TeamGoal.target.name] == 42
        assert tg_row[TeamGoal.repositories.name] is None
        assert tg_row[Goal.metric_params.name] is None

    async def test_create_multiple_team_goals(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(owner_id=1, id=10),
            TeamFactory(owner_id=1, id=20),
        )

        team_goals = [
            {TeamGoalInputFields.teamId: 10, TeamGoalInputFields.target: {"int": 42}},
            {TeamGoalInputFields.teamId: 20, TeamGoalInputFields.target: {"str": "foobar"}},
        ]
        variables = {
            "createGoalInput": self._mk_input(teamGoals=team_goals, name="G1"),
            "accountId": 1,
        }
        new_goal_id = await self._create(variables)

        await assert_existing_row(
            sdb,
            Goal,
            id=new_goal_id,
            name="G1",
            metric=PullRequestMetricID.PR_REVIEW_TIME_BELOW_THRESHOLD_RATIO,
        )

        team_goals = await sdb.fetch_all(
            sa.select(TeamGoal).where(TeamGoal.goal_id == 1).order_by(TeamGoal.team_id),
        )
        assert len(team_goals) == 2

        assert team_goals[0]["team_id"] == 10
        assert team_goals[0][TeamGoalInputFields.target] == 42
        assert team_goals[1]["team_id"] == 20
        assert team_goals[1][TeamGoalInputFields.target] == "foobar"

    async def test_datetime_as_date(self, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(owner_id=1, id=10)))
        team_goals = [{TeamGoalInputFields.teamId: 10, TeamGoalInputFields.target: {"int": 42}}]

        variables = {
            "createGoalInput": self._mk_input(
                validFrom="2022-05-04T08:01:32.485897",
                expiresAt="2022-06-01",
                teamGoals=team_goals,
            ),
            "accountId": 1,
        }
        new_goal_id = await self._create(variables)
        goal_row = await assert_existing_row(sdb, Goal, id=new_goal_id)
        assert ensure_db_datetime_tz(goal_row[Goal.valid_from.name], sdb) == datetime(
            2022, 5, 4, tzinfo=timezone.utc,
        )

    async def test_similar_goals(self, sdb: Database) -> None:
        # test that the uc_goal constraint doesn't fail more than expected
        await models_insert(
            sdb,
            GoalFactory(
                id=100,
                name="G1",
                valid_from=datetime(2021, 1, 1, tzinfo=timezone.utc),
                expires_at=datetime(2021, 4, 1, tzinfo=timezone.utc),
            ),
            TeamFactory(id=100),
        )

        # same interval, different name
        template = TEMPLATES_COLLECTION[1]
        variables: dict = {
            "createGoalInput": self._mk_input(
                name=template[CreateGoalInputFields.name],
                metric=template[CreateGoalInputFields.metric],
                validFrom="2022-01-01",
                expiresAt="2022-03-31",
                teamGoals=[
                    {TeamGoalInputFields.teamId: 100, TeamGoalInputFields.target: {"int": 1}},
                ],
            ),
            "accountId": 1,
        }
        await self._create(variables)

        # same name, different interval
        template = TEMPLATES_COLLECTION[0]
        variables["createGoalInput"][CreateGoalInputFields.name] = template[
            CreateGoalInputFields.name
        ]
        variables["createGoalInput"][CreateGoalInputFields.metric] = template[
            CreateGoalInputFields.metric
        ]
        variables["createGoalInput"][CreateGoalInputFields.expiresAt] = "2022-06-30"

        await self._create(variables)

        # same interval but different account
        await models_insert(
            sdb,
            AccountFactory(id=5),
            UserAccountFactory(account_id=5, user_id="gh|XXX"),
            TeamFactory(id=200, owner_id=5),
        )
        variables["createGoalInput"][CreateGoalInputFields.name] = "G1"
        variables["createGoalInput"][CreateGoalInputFields.expiresAt] = "2022-03-31"
        variables["accountId"] = 5
        variables["createGoalInput"][CreateGoalInputFields.teamGoals] = [
            {TeamGoalInputFields.teamId: 200, TeamGoalInputFields.target: {"int": 1}},
        ]
        await self._create(variables, user_id="gh|XXX")

    @freeze_time("2022-03-01")
    async def test_future_dates_are_accepted(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=100))
        variables: dict = {
            "createGoalInput": self._mk_input(
                name=TEMPLATES_COLLECTION[0][CreateGoalInputFields.name],
                metric=TEMPLATES_COLLECTION[0][CreateGoalInputFields.metric],
                validFrom="2023-01-01",
                expiresAt="2023-12-31",
                teamGoals=[
                    {TeamGoalInputFields.teamId: 100, TeamGoalInputFields.target: {"int": 1}},
                ],
            ),
            "accountId": 1,
        }
        new_goal_id = await self._create(variables)
        await assert_existing_row(
            sdb, Goal, id=new_goal_id, valid_from=dt(2023, 1, 1), expires_at=dt(2024, 1, 1),
        )

    async def test_repositories(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamFactory(id=11, parent_id=10))

        team_goals = [
            {TeamGoalInputFields.teamId: 10, TeamGoalInputFields.target: {"int": 42}},
            {TeamGoalInputFields.teamId: 11, TeamGoalInputFields.target: {"int": 43}},
        ]

        variables = {
            "createGoalInput": self._mk_input(
                repositories=["github.com/src-d/go-git/alpha", "github.com/src-d/go-git/beta"],
                teamGoals=team_goals,
            ),
            "accountId": 1,
        }

        new_goal_id = await self._create(variables)

        goal_row = await assert_existing_row(sdb, Goal, id=new_goal_id, account_id=1)
        assert goal_row[Goal.repositories.name] == [[40550, "alpha"], [40550, "beta"]]

        # goal repositories are copied to team goals
        tg_row_10 = await assert_existing_row(sdb, TeamGoal, goal_id=new_goal_id, team_id=10)
        assert tg_row_10[TeamGoal.target.name] == 42
        assert tg_row_10[TeamGoal.repositories.name] == [[40550, "alpha"], [40550, "beta"]]

        tg_row_11 = await assert_existing_row(sdb, TeamGoal, goal_id=new_goal_id, team_id=11)
        assert tg_row_11[TeamGoal.target.name] == 43
        assert tg_row_11[TeamGoal.repositories.name] == [[40550, "alpha"], [40550, "beta"]]

    async def test_jira_fields(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10))

        team_goals = [
            {TeamGoalInputFields.teamId: 10, TeamGoalInputFields.target: {"int": 42}},
        ]

        variables = {
            "createGoalInput": self._mk_input(
                jiraProjects=["P0"],
                jiraIssueTypes=["Bugs", "tasks", "Story", "Task"],
                teamGoals=team_goals,
            ),
            "accountId": 1,
        }
        new_goal_id = await self._create(variables)
        goal_row = await assert_existing_row(sdb, Goal, id=new_goal_id, account_id=1)

        assert goal_row[Goal.repositories.name] is None
        assert goal_row[Goal.jira_projects.name] == ["P0"]
        assert goal_row[Goal.jira_priorities.name] is None
        assert goal_row[Goal.jira_issue_types.name] == ["bug", "story", "task"]

        # fields are also copied to the TeamGoal row
        tg_row = await assert_existing_row(sdb, TeamGoal, goal_id=new_goal_id, team_id=10)
        assert tg_row[TeamGoal.repositories.name] is None
        assert tg_row[TeamGoal.jira_projects.name] == ["P0"]
        assert tg_row[TeamGoal.jira_priorities.name] is None
        assert tg_row[TeamGoal.jira_issue_types.name] == ["bug", "story", "task"]

    async def test_jira_priorities(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10))

        team_goals = [
            {TeamGoalInputFields.teamId: 10, TeamGoalInputFields.target: {"int": 42}},
        ]

        variables = {
            "createGoalInput": self._mk_input(
                jiraPriorities=["High", "low", "high"], teamGoals=team_goals,
            ),
            "accountId": 1,
        }
        new_goal_id = await self._create(variables)
        goal_row = await assert_existing_row(sdb, Goal, id=new_goal_id, account_id=1)

        for col in (Goal.repositories, Goal.jira_projects, Goal.jira_issue_types):
            assert goal_row[col.name] is None

        assert goal_row[Goal.jira_priorities.name] == ["high", "low"]

        tg_row = await assert_existing_row(sdb, TeamGoal, goal_id=new_goal_id, team_id=10)
        for col in (TeamGoal.repositories, TeamGoal.jira_projects, TeamGoal.jira_issue_types):
            assert tg_row[col.name] is None
        assert tg_row[TeamGoal.jira_priorities.name] == ["high", "low"]

    async def test_metric_params(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamFactory(id=11, parent_id=10))
        team_goals = [
            {"teamId": 10, "target": {"int": 42}},
            {"teamId": 11, "target": {"int": 43}, "metricParams": {"threshold": {"int": 23}}},
        ]
        variables = {
            "createGoalInput": self._mk_input(
                teamGoals=team_goals, metricParams={"threshold": {"str": "100s"}},
            ),
            "accountId": 1,
        }
        new_goal_id = await self._create(variables)
        goal_row = await assert_existing_row(sdb, Goal, id=new_goal_id, account_id=1)
        assert goal_row[Goal.metric_params.name] == {"threshold": "100s"}

        tg10_row = await assert_existing_row(sdb, TeamGoal, goal_id=new_goal_id, team_id=10)
        assert tg10_row[TeamGoal.metric_params.name] is None

        tg11_row = await assert_existing_row(sdb, TeamGoal, goal_id=new_goal_id, team_id=11)
        assert tg11_row[TeamGoal.metric_params.name] == {"threshold": 23}

    async def test_default_user_is_god(self, sdb: Database) -> None:
        await models_insert(
            sdb, TeamFactory(owner_id=1, id=10), GodFactory(user_id=DEFAULT_USER_ID),
        )

        team_goals = [{TeamGoalInputFields.teamId: 10, TeamGoalInputFields.target: {"int": 0}}]

        variables = {"createGoalInput": self._mk_input(teamGoals=team_goals), "accountId": 1}
        new_goal_id = await self._create(variables, None)
        await assert_existing_row(sdb, Goal, id=new_goal_id, account_id=1)

    async def _create(self, *args: Any, **kwargs: Any) -> int:
        res = await self._request(*args, **kwargs)
        assert "errors" not in res
        return res["data"]["createGoal"]["goal"]["id"]


class BaseRemoveGoalTest(BaseGoalTest):
    _QUERY = """
        mutation ($accountId: Int!, $goalId: Int!) {
          removeGoal(accountId: $accountId, goalId: $goalId) {
            success
          }
        }
    """

    async def _request(
        self,
        accountId: int,
        goalId: int,
        user_id: Optional[str] = _USER_ID,
    ) -> dict:
        variables = {"accountId": accountId, "goalId": goalId}
        body = {"query": self._QUERY, "variables": variables}
        return await self._base_request(body, user_id)


class TestRemoveGoalErrors(BaseRemoveGoalTest):
    async def test_non_existing_goal(self) -> None:
        res = await self._request(1, 999)
        assert_extension_error(res, "Goal 999 not found")

    async def test_account_mismatch(self, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(GoalFactory(id=100, account_id=2)))

        res = await self._request(1, 100)
        assert_extension_error(res, "Goal 100 not found")
        await assert_existing_row(sdb, Goal, id=100)

    async def test_default_user_forbidden(self, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(GoalFactory(id=100, account_id=2)))
        res = await self._request(1, 100, user_id=None)
        assert get_extension_error_obj(res)["type"] == "/errors/ForbiddenError"
        await assert_existing_row(sdb, Goal, id=100)


class TestRemoveGoal(BaseRemoveGoalTest):
    async def test_remove(self, sdb: Database):
        await sdb.execute(model_insert_stmt(GoalFactory(id=100, account_id=1)))

        res = await self._request(1, 100)
        assert "errors" not in res
        assert res["data"]["removeGoal"]["success"]
        await self._assert_no_goal_exists(sdb)

    async def test_remove_with_team_goals(self, sdb: Database):
        await models_insert(
            sdb,
            TeamFactory(owner_id=1, id=10),
            TeamFactory(owner_id=1, id=20),
            GoalFactory(id=100, account_id=1),
            GoalFactory(id=200, account_id=1),
            GoalFactory(id=300, account_id=2),
            TeamGoalFactory(team_id=10, goal_id=100),
            TeamGoalFactory(team_id=20, goal_id=100),
            TeamGoalFactory(team_id=20, goal_id=200),
        )

        res = await self._request(1, 100)
        assert "errors" not in res
        assert res["data"]["removeGoal"]["success"]

        await assert_missing_row(sdb, Goal, id=100, account_id=1)
        await assert_missing_row(sdb, TeamGoal, goal_id=100)

        await assert_existing_row(sdb, Goal, id=200)
        await assert_existing_row(sdb, Goal, id=300)
        await assert_existing_row(sdb, TeamGoal, goal_id=200)


class BaseUpdateGoalTest(BaseGoalTest):
    _QUERY = """
        mutation ($accountId: Int!, $input: UpdateGoalInput!) {
          updateGoal(accountId: $accountId, input: $input) {
            goal {
              id
            }
          }
        }
    """

    async def _request(
        self,
        variables: dict,
        user_id: Optional[str] = _USER_ID,
    ) -> dict:
        body = {"query": self._QUERY, "variables": variables}
        return await self._base_request(body, user_id)


class TestUpdateGoalErrors(BaseUpdateGoalTest):
    async def test_dupl_team_ids(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100), TeamFactory(id=10))

        team_changes = [
            {TeamGoalChangeFields.teamId: 10, TeamGoalChangeFields.target: {"int": 10}},
            {TeamGoalChangeFields.teamId: 10, TeamGoalChangeFields.remove: True},
        ]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        assert_extension_error(res, "Multiple changes for teamId-s: 10")

    async def test_invalid_team_changes(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100), TeamFactory(id=10), TeamFactory(id=20))

        team_changes = [
            {
                TeamGoalChangeFields.teamId: 10,
                TeamGoalChangeFields.target: {"int": 10},
                TeamGoalChangeFields.remove: True,
            },
            {TeamGoalChangeFields.teamId: 20, TeamGoalChangeFields.remove: False},
        ]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        error = get_extension_error_obj(res)["detail"]
        assert "Both remove and new target present for teamId-s: 10" in error
        assert "Invalid target for teamId-s: 20" in error

    async def test_invalid_team_target(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100), TeamFactory(id=10))

        team_changes = [{TeamGoalChangeFields.teamId: 10, TeamGoalChangeFields.target: {}}]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        error = get_extension_error_obj(res)["detail"]
        assert "Invalid target for teamId-s: 10" in error

    async def test_remove_account_mismatch(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100, account_id=2),
            TeamFactory(id=10, owner_id=2),
            TeamGoalFactory(team_id=10, goal_id=100),
        )

        team_changes = [{TeamGoalChangeFields.teamId: 10, TeamGoalChangeFields.remove: True}]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        assert_extension_error(res, "Goal 100 not found or access denied")
        await assert_existing_row(sdb, TeamGoal, team_id=10, goal_id=100)

    async def test_goal_account_mismatch(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100, account_id=2), TeamFactory(id=10))

        team_changes = [
            {TeamGoalChangeFields.teamId: 10, TeamGoalChangeFields.target: {"int": 10}},
        ]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        assert_extension_error(res, "Goal 100 not found or access denied")

    async def test_assign_team_account_mismatch(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100), TeamFactory(id=10, owner_id=2))
        team_changes = [
            {TeamGoalChangeFields.teamId: 10, TeamGoalChangeFields.target: {"int": 10}},
        ]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        assert_extension_error(res, "Team-s don't exist or access denied: 10")
        await assert_missing_row(sdb, TeamGoal, team_id=10)

    async def test_team_goals_to_remove_are_missing(
        self,
        sdb: Database,
    ) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100),
            TeamFactory(id=10),
            TeamFactory(id=20),
            TeamFactory(id=30),
            TeamFactory(id=40),
            TeamGoalFactory(team_id=10, goal_id=100),
            TeamGoalFactory(team_id=30, goal_id=100, target=9999),
        )

        team_changes = [
            {TeamGoalChangeFields.teamId: 10, TeamGoalChangeFields.remove: True},
            {TeamGoalChangeFields.teamId: 20, TeamGoalChangeFields.remove: True},
            {TeamGoalChangeFields.teamId: 30, TeamGoalChangeFields.target: {"float": 4.2}},
            {TeamGoalChangeFields.teamId: 40, TeamGoalChangeFields.target: {"int": 42}},
        ]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        assert_extension_error(res, "TeamGoal-s to remove not found for teams: 20")

        await assert_existing_row(sdb, TeamGoal, team_id=10, goal_id=100)
        # goal for team 30 not changed
        team30_row = await assert_existing_row(sdb, TeamGoal, team_id=30, goal_id=100)
        assert team30_row[TeamGoalChangeFields.target] == 9999
        # goal for team 40 not created
        await assert_missing_row(sdb, TeamGoal, team_id=40, goal_id=100)

    async def test_default_user_forbidden(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100), TeamFactory(id=10))
        team_changes = [{TeamGoalChangeFields.teamId: 10, TeamGoalInputFields.target: {"int": 10}}]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables, user_id=None)
        assert get_extension_error_obj(res)["type"] == "/errors/ForbiddenError"
        await assert_missing_row(sdb, TeamGoal, team_id=10)

    async def test_cannot_delete_all_team_goals(self, sdb: Database) -> None:
        await models_insert(
            sdb, GoalFactory(id=20), TeamFactory(id=10), TeamGoalFactory(team_id=10, goal_id=20),
        )
        team_changes = {TeamGoalChangeFields.teamId: 10, TeamGoalChangeFields.remove: True}
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 20,
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        assert_extension_error(res, "Impossible to remove all TeamGoal-s from the Goal")

        await assert_existing_row(sdb, TeamGoal, team_id=10, goal_id=20)

    async def test_bad_repositories(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100))
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.repositories: {
                    UpdateRepositoriesInputFields.value: ["github.com/athenianco/xxx"],
                },
            },
        }
        res = await self._request(variables)
        assert_extension_error(res, "repository")
        row = await assert_existing_row(sdb, Goal, id=100)
        assert row[Goal.repositories.name] is None


class TestUpdateGoal(BaseUpdateGoalTest):
    async def test_no_team_goal_changes(self, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(GoalFactory(id=100, account_id=1)))
        variables = {"accountId": 1, "input": {UpdateGoalInputFields.goalId: 100}}
        res = await self._request(variables)
        assert "errors" not in res

    async def test_deletions(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamFactory(id=20),
            TeamFactory(id=30),
            GoalFactory(id=101),
            TeamGoalFactory(team_id=10, goal_id=101),
            TeamGoalFactory(team_id=20, goal_id=101),
            TeamGoalFactory(team_id=30, goal_id=101),
        )
        team_changes = [
            {TeamGoalChangeFields.teamId: 10, TeamGoalChangeFields.remove: True},
            {TeamGoalChangeFields.teamId: 20, TeamGoalChangeFields.remove: True},
        ]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 101,
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        assert res["data"]["updateGoal"]["goal"]["id"] == 101
        await assert_missing_row(sdb, TeamGoal, team_id=10)
        await assert_missing_row(sdb, TeamGoal, team_id=20)
        await assert_existing_row(sdb, Goal, id=101)
        await assert_existing_row(sdb, Team, id=10)
        await assert_existing_row(sdb, Team, id=20)
        await assert_existing_row(sdb, TeamGoal, team_id=30)

    @freeze_time("2022-04-01T09:30:00")
    async def test_some_changes(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100, archived=False),
            TeamFactory(id=10),
            TeamFactory(id=20),
            TeamFactory(id=30),
            TeamGoalFactory(team_id=10, goal_id=100),
            TeamGoalFactory(
                team_id=20,
                goal_id=100,
                target=9999,
                created_at=datetime(1, 1, 2, tzinfo=timezone.utc),
                updated_at=datetime(1, 1, 3, tzinfo=timezone.utc),
            ),
        )

        team_changes = [
            {TeamGoalChangeFields.teamId: 10, TeamGoalChangeFields.remove: True},
            {TeamGoalChangeFields.teamId: 20, TeamGoalChangeFields.target: {"int": 8888}},
            {TeamGoalChangeFields.teamId: 30, TeamGoalChangeFields.target: {"int": 7777}},
        ]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        assert "errors" not in res
        assert res["data"]["updateGoal"]["goal"]["id"] == 100

        await assert_existing_row(sdb, Goal, id=100, archived=False)

        await assert_missing_row(sdb, TeamGoal, team_id=10, goal_id=100)

        team_goal_20_row = await assert_existing_row(sdb, TeamGoal, team_id=20, goal_id=100)
        assert team_goal_20_row[TeamGoalChangeFields.target] == 8888
        expected_created_at = datetime(1, 1, 2, tzinfo=timezone.utc)
        assert (
            ensure_db_datetime_tz(team_goal_20_row[TeamGoal.created_at.name], sdb)
            == expected_created_at
        )
        expected_updated_at = datetime(2022, 4, 1, 9, 30, tzinfo=timezone.utc)
        assert (
            ensure_db_datetime_tz(team_goal_20_row[TeamGoal.updated_at.name], sdb)
            == expected_updated_at
        )

        team_goal_30_row = await assert_existing_row(sdb, TeamGoal, team_id=30, goal_id=100)
        assert team_goal_30_row[TeamGoalChangeFields.target] == 7777

    async def test_update_archived(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100, archived=False))
        variables = {
            "accountId": 1,
            "input": {UpdateGoalInputFields.goalId: 100, UpdateGoalInputFields.archived: True},
        }
        res = await self._request(variables)
        assert "errors" not in res
        await assert_existing_row(sdb, Goal, id=100, archived=True)

    async def test_update_name(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100, archived=False))
        variables = {
            "accountId": 1,
            "input": {UpdateGoalInputFields.goalId: 100, UpdateGoalInputFields.name: "Vadim"},
        }
        res = await self._request(variables)
        assert "errors" not in res
        await assert_existing_row(sdb, Goal, id=100, name="Vadim")

    async def test_update_repositories(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100))
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.repositories: {
                    UpdateRepositoriesInputFields.value: ["github.com/src-d/go-git"],
                },
            },
        }
        res = await self._request(variables)
        assert "errors" not in res
        row = await assert_existing_row(sdb, Goal, id=100)
        assert row[Goal.repositories.name] == [[40550, ""]]

        variables = {
            "accountId": 1,
            "input": {UpdateGoalInputFields.goalId: 100},
        }
        res = await self._request(variables)
        assert "errors" not in res
        row = await assert_existing_row(sdb, Goal, id=100)
        assert row[Goal.repositories.name] == [[40550, ""]]

        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.repositories: {
                    UpdateRepositoriesInputFields.value: None,
                },
            },
        }
        res = await self._request(variables)
        assert "errors" not in res
        row = await assert_existing_row(sdb, Goal, id=100)
        assert row[Goal.repositories.name] is None

    async def test_update_repos_and_change_team_goals(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100),
            TeamFactory(id=10),
            TeamFactory(id=11, parent_id=10),
            TeamGoalFactory(team_id=10, goal_id=100, target=1, repositories=[[40550, "alpha"]]),
        )
        team_changes = [
            {TeamGoalChangeFields.teamId: 10, TeamGoalChangeFields.target: {"int": 2}},
            {TeamGoalChangeFields.teamId: 11, TeamGoalChangeFields.target: {"int": 3}},
        ]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.repositories: {
                    UpdateRepositoriesInputFields.value: ["github.com/src-d/go-git"],
                },
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        assert "errors" not in res
        assert res["data"]["updateGoal"]["goal"]["id"] == 100

        row = await assert_existing_row(sdb, Goal, id=100)
        assert row[Goal.repositories.name] == [[40550, ""]]

        team_goal_row_10 = await assert_existing_row(sdb, TeamGoal, goal_id=100, team_id=10)
        assert team_goal_row_10[TeamGoal.repositories.name] == [[40550, ""]]

        team_goal_row_11 = await assert_existing_row(sdb, TeamGoal, goal_id=100, team_id=11)
        assert team_goal_row_11[TeamGoal.repositories.name] == [[40550, ""]]

    async def test_new_assigned_teams_inherit_goal_values(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100, repositories=[[40550, ""]]),
            TeamFactory(id=10),
            TeamFactory(id=11, parent_id=10),
            TeamGoalFactory(team_id=10, goal_id=100, target=1, repositories=[[40550, ""]]),
        )
        team_changes = [
            {TeamGoalChangeFields.teamId: 11, TeamGoalChangeFields.target: {"int": 3}},
        ]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        assert "errors" not in res
        assert res["data"]["updateGoal"]["goal"]["id"] == 100

        row = await assert_existing_row(sdb, Goal, id=100)
        assert row[Goal.repositories.name] == [[40550, ""]]
        team_goal_row_10 = await assert_existing_row(sdb, TeamGoal, goal_id=100, team_id=10)
        assert team_goal_row_10[TeamGoal.repositories.name] == [[40550, ""]]
        team_goal_row_11 = await assert_existing_row(sdb, TeamGoal, goal_id=100, team_id=11)
        assert team_goal_row_11[TeamGoal.repositories.name] == [[40550, ""]]

    async def test_update_jira_fields(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100, jira_priorities=["high"]),
            TeamFactory(id=10),
            TeamGoalFactory(
                team_id=10, goal_id=100, jira_projects=["P2"], jira_priorities=["high"],
            ),
        )
        team_changes = [
            {TeamGoalChangeFields.teamId: 10, TeamGoalChangeFields.target: {"int": 2}},
        ]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.jiraProjects: {
                    UpdateRepositoriesInputFields.value: ["P0", "P1"],
                },
                UpdateGoalInputFields.jiraPriorities: {},
                UpdateGoalInputFields.jiraIssueTypes: {
                    UpdateRepositoriesInputFields.value: ["Tasks", "bugs"],
                },
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        assert "errors" not in res
        assert res["data"]["updateGoal"]["goal"]["id"] == 100

        row = await assert_existing_row(sdb, Goal, id=100)
        assert row[Goal.jira_projects.name] == ["P0", "P1"]
        assert row[Goal.jira_priorities.name] is None
        assert row[Goal.jira_issue_types.name] == ["bug", "task"]

        tg_row = await assert_existing_row(sdb, TeamGoal, goal_id=100, team_id=10)
        # team goal jira_projects are overwritten
        assert tg_row[TeamGoal.jira_projects.name] == ["P0", "P1"]
        assert tg_row[TeamGoal.jira_priorities.name] is None
        assert tg_row[TeamGoal.jira_issue_types.name] == ["bug", "task"]

    async def test_unset_then_set_last_team_goal(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=20),
            TeamFactory(id=10),
            TeamFactory(id=11, parent_id=10),
            TeamGoalFactory(team_id=10, goal_id=20),
        )
        team_changes = [
            {TeamGoalChangeFields.teamId: 10, TeamGoalChangeFields.remove: True},
            {TeamGoalChangeFields.teamId: 11, TeamGoalChangeFields.target: {"int": 2}},
        ]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 20,
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        assert "errors" not in res

        team_11_row = await assert_existing_row(sdb, TeamGoal, team_id=11, goal_id=20)
        assert team_11_row[TeamGoal.target.name] == 2
        await assert_missing_row(sdb, TeamGoal, team_id=10, goal_id=20)

    async def test_set_teams_metric_params(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100, metric_params={"threshold": 1}),
            *[TeamFactory(id=id_) for id_ in (10, 20, 30)],
            TeamGoalFactory(team_id=10, goal_id=100, metric_params={"threshold": 2}),
            TeamGoalFactory(team_id=30, goal_id=100, metric_params={"threshold": 3}),
        )

        team_changes = [
            {"teamId": 10, "target": {"int": 1}, "metricParams": {"threshold": {"int": 10}}},
            {"teamId": 20, "target": {"int": 1}, "metricParams": None},
            {"teamId": 30, "target": {"int": 1}, "metricParams": {"threshold": {"int": 20}}},
        ]
        variables = {
            "accountId": 1,
            "input": {
                UpdateGoalInputFields.goalId: 100,
                UpdateGoalInputFields.teamGoalChanges: team_changes,
            },
        }
        res = await self._request(variables)
        assert "errors" not in res
        assert res["data"]["updateGoal"]["goal"]["id"] == 100
        await assert_existing_row(
            sdb, TeamGoal, goal_id=100, team_id=10, metric_params={"threshold": 10},
        )
        row_20 = await assert_existing_row(sdb, TeamGoal, goal_id=100, team_id=20)
        assert row_20[TeamGoal.metric_params.name] is None
        await assert_existing_row(
            sdb, TeamGoal, goal_id=100, team_id=30, metric_params={"threshold": 20},
        )
