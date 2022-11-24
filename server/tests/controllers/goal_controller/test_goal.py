"""Tests for the goal CRUD controllers."""

from datetime import date
from typing import Any, Optional, Sequence

from freezegun import freeze_time
import pytest
import sqlalchemy as sa

from athenian.api.db import Database, ensure_db_datetime_tz
from athenian.api.models.state.models import Base, Goal, Team, TeamGoal
from athenian.api.models.web import (
    GoalCreateRequest,
    GoalUpdateRequest,
    PullRequestMetricID,
    TeamGoalAssociation,
)
from athenian.api.models.web.goal import MetricValue
from tests.testutils.auth import force_request_auth
from tests.testutils.db import assert_existing_row, assert_missing_row, models_insert
from tests.testutils.factory.state import (
    AccountFactory,
    GoalFactory,
    TeamFactory,
    TeamGoalFactory,
    UserAccountFactory,
)
from tests.testutils.requester import Requester
from tests.testutils.time import dt

_USER_ID = "github|1"


class BaseGoalTest(Requester):
    @pytest.fixture(scope="function", autouse=True)
    async def _create_user(self, sdb):
        await models_insert(sdb, UserAccountFactory(user_id=_USER_ID))

    @classmethod
    async def _assert_no_goal_exists(cls, sdb: Database) -> None:
        assert await sdb.fetch_one(sa.select(Goal)) is None
        assert await sdb.fetch_one(sa.select(TeamGoal)) is None


class BaseDeleteGoalTest(BaseGoalTest):
    async def _request(
        self,
        goal_id: int,
        assert_status: int = 204,
        user_id: Optional[str] = _USER_ID,
        **kwargs: Any,
    ) -> Optional[dict]:
        path = f"/private/align/goal/{goal_id}"
        with force_request_auth(user_id, self.headers) as headers:
            response = await self.client.request(
                method="DELETE", path=path, headers=headers, **kwargs,
            )
        assert response.status == assert_status
        if assert_status == 204:
            return None
        return await response.json()


class TestDeleteGoalErrors(BaseDeleteGoalTest):
    async def test_non_existing_goal(self) -> None:
        res = await self._request(999, 404)
        assert res is not None
        assert res["detail"] == "Goal 999 not found or access denied."

    async def test_account_mismatch(self, sdb: Database) -> None:
        await models_insert(sdb, AccountFactory(id=99), GoalFactory(id=100, account_id=99))
        res = await self._request(100, 404)
        assert res is not None
        assert "99" not in res["detail"]
        await assert_existing_row(sdb, Goal, id=100)

    async def test_default_user_forbidden(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100, account_id=2))
        await self._request(100, 403, user_id=None)
        await assert_existing_row(sdb, Goal, id=100)


class TestDeleteGoal(BaseDeleteGoalTest):
    async def test_delete(self, sdb: Database):
        await models_insert(sdb, GoalFactory(id=100))
        await self._request(100)
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

        await self._request(100)
        await assert_missing_row(sdb, Goal, id=100, account_id=1)
        await assert_missing_row(sdb, TeamGoal, goal_id=100)

        await assert_existing_row(sdb, Goal, id=200)
        await assert_existing_row(sdb, Goal, id=300)
        await assert_existing_row(sdb, TeamGoal, goal_id=200)


def _team_goals_assoc_from_tuples(
    *raw_team_goals: tuple[int, MetricValue] | tuple[int, MetricValue, dict],
) -> list[TeamGoalAssociation]:
    normalized_team_goals = [tg if len(tg) > 2 else (*tg, None) for tg in raw_team_goals]
    return [
        TeamGoalAssociation(team_id=t_id, target=target, metric_params=params)
        for t_id, target, params in normalized_team_goals
    ]


class BaseCreateGoalTest(BaseGoalTest):
    async def _request(
        self,
        assert_status: int = 200,
        user_id: Optional[str] = _USER_ID,
        **kwargs: Any,
    ) -> dict:
        path = "/private/align/goal/create"
        with force_request_auth(user_id, self.headers) as headers:
            response = await self.client.request(
                method="POST", path=path, headers=headers, **kwargs,
            )
        assert response.status == assert_status
        return await response.json()

    @classmethod
    def _body(
        self,
        account=1,
        name: str = "My Goal",
        metric: str = PullRequestMetricID.PR_ALL_COUNT,
        valid_from: date = date(2012, 10, 1),  # noqa: B008
        expires_at: date = date(2012, 12, 31),  # noqa: B008
        team_goals: Sequence[tuple[int, MetricValue] | tuple[int, MetricValue, dict]] = (),
        **kwargs,
    ) -> dict:
        request = GoalCreateRequest(
            account=account,
            name=name,
            metric=metric,
            valid_from=valid_from,
            expires_at=expires_at,
            team_goals=_team_goals_assoc_from_tuples(*team_goals),
            **kwargs,
        )
        body = request.to_dict()
        body["valid_from"] = body["valid_from"].isoformat()
        body["expires_at"] = body["expires_at"].isoformat()
        return body


class TestCreateGoalErrors(BaseCreateGoalTest):
    async def test_invalid_metric(self, sdb: Database) -> None:
        body = self._body(metric="xxx", team_goals=[(1, 2)])
        res = await self._request(400, json=body)
        assert res["title"] == "Bad Request"
        assert "not valid" in res["detail"]
        assert "xxx" in res["detail"]
        await self._assert_no_goal_exists(sdb)

    async def test_invalid_date(self, sdb: Database) -> None:
        body = self._body(team_goals=[(1, 2)])
        body["valid_from"] = "not-date"
        res = await self._request(400, json=body)
        assert res["title"] == "Bad Request"
        assert "not-date" in res["detail"]
        await self._assert_no_goal_exists(sdb)

    async def test_missing_date(self, sdb: Database) -> None:
        body = self._body(team_goals=[(1, 2)])
        body.pop("expires_at")
        res = await self._request(400, json=body)
        assert res["title"] == "Bad Request"
        assert "expires_at" in res["detail"]
        await self._assert_no_goal_exists(sdb)

    async def test_account_mismatch(self, sdb: Database) -> None:
        body = self._body(account=3)
        await self._request(404, json=body)

    async def test_no_team_goals(self, sdb: Database) -> None:
        body = self._body(team_goals=[])
        res = await self._request(400, json=body)
        assert res["title"] == "Bad Request"
        assert "team_goals" in res["detail"]
        await self._assert_no_goal_exists(sdb)

    async def test_more_goals_for_same_team(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10))
        body = self._body(team_goals=[(10, 42), (10, 44)])
        res = await self._request(400, json=body)
        assert res["title"] == "Bad Request"
        assert res["detail"] == "More than one team goal with the same teamId"
        await self._assert_no_goal_exists(sdb)

    async def test_unexisting_team(self, sdb: Database) -> None:
        body = self._body(team_goals=[(10, 10)])
        res = await self._request(404, json=body)
        assert res["detail"] == "Some teams don't exist or access denied: 10"
        await self._assert_no_goal_exists(sdb)

    async def test_goals_for_other_account_team(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10), TeamFactory(owner_id=2, id=20))
        body = self._body(team_goals=[(10, 42), (20, 44)])
        res = await self._request(404, json=body)
        assert res["detail"] == "Some teams don't exist or access denied: 20"
        await self._assert_no_goal_exists(sdb)

    async def test_inverted_dates(self, sdb: Database):
        await models_insert(sdb, TeamFactory(owner_id=1, id=10))
        body = self._body(
            team_goals=[(10, 42)], valid_from=date(2022, 4, 1), expires_at=date(2022, 1, 1),
        )
        res = await self._request(400, json=body)
        assert res["title"] == "Bad Request"
        assert res["detail"] == "Goal expires_at cannot precede valid_from"

    async def test_none_team_target(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10))
        body = self._body(team_goals=[(10, 0)])
        body["team_goals"][0]["target"] = None
        await self._request(400, json=body)

    async def test_default_user_forbidden(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10))
        body = self._body(team_goals=[(10, 42)])
        res = await self._request(403, json=body, user_id=None)
        assert res["type"] == "/errors/ForbiddenError"

    async def test_bad_repositories(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10))
        body = self._body(team_goals=[(10, 42)], repositories=["github.com/athenianco/xxx"])
        res = await self._request(400, json=body)
        assert res["detail"] == "Unknown repository github.com/athenianco/xxx"
        await self._assert_no_goal_exists(sdb)

    async def test_invalid_metric_params(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10))
        body = self._body(team_goals=[(10, 42)], metric_params=0)
        res = await self._request(400, json=body)
        assert res["detail"] == "0 is not of type 'object' - 'metric_params'"
        await self._assert_no_goal_exists(sdb)


class TestCreateGoals(BaseCreateGoalTest):
    async def test_create_single_team_goal(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10))

        body = self._body(
            name="G0",
            metric=PullRequestMetricID.PR_COMMENTS_PER,
            team_goals=[(10, 42)],
            valid_from=date(2022, 1, 1),
            expires_at=date(2022, 12, 31),
        )
        new_goal_id = (await self._request(json=body))["id"]

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
        assert ensure_db_datetime_tz(goal_row[Goal.valid_from.name], sdb) == dt(2022, 1, 1)

        # expires_at is moved to the day after the one received in api
        assert ensure_db_datetime_tz(goal_row[Goal.expires_at.name], sdb) == dt(2023, 1, 1)

        tg_row = await assert_existing_row(sdb, TeamGoal, goal_id=new_goal_id, team_id=10)
        assert tg_row[TeamGoal.target.name] == 42
        assert tg_row[TeamGoal.repositories.name] is None

    async def test_create_multiple_team_goals(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10), TeamFactory(owner_id=1, id=20))

        metric = PullRequestMetricID.PR_REVIEW_TIME
        body = self._body(name="G1", team_goals=[(10, 42), (20, "100s")], metric=metric)
        new_goal_id = (await self._request(json=body))["id"]

        await assert_existing_row(sdb, Goal, id=new_goal_id, name="G1", metric=metric)

        team_goals = await sdb.fetch_all(
            sa.select(TeamGoal).where(TeamGoal.goal_id == 1).order_by(TeamGoal.team_id),
        )
        assert len(team_goals) == 2

        assert team_goals[0][TeamGoal.team_id.name] == 10
        assert team_goals[0][TeamGoal.target.name] == 42
        assert team_goals[1][TeamGoal.team_id.name] == 20
        assert team_goals[1][TeamGoal.target.name] == "100s"

    async def test_datetime_as_date(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10))
        body = self._body(team_goals=[(10, 42)])
        body["valid_from"] = "2022-05-04T08:01:32.485897"
        body["expires_at"] = "2022-06-01"
        new_goal_id = (await self._request(json=body))["id"]
        goal_row = await assert_existing_row(sdb, Goal, id=new_goal_id)
        assert ensure_db_datetime_tz(goal_row[Goal.valid_from.name], sdb) == dt(2022, 5, 4)

    # ignore in athenian.api.serialization is needed to not break klass == date tests
    @freeze_time("2022-03-01", ignore=["athenian.api.serialization"])
    async def test_future_dates_are_accepted(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=100))
        body = self._body(
            valid_from=date(2023, 1, 1), expires_at=date(2023, 12, 31), team_goals=[(100, 1)],
        )
        new_goal_id = (await self._request(json=body))["id"]
        await assert_existing_row(
            sdb, Goal, id=new_goal_id, valid_from=dt(2023, 1, 1), expires_at=dt(2024, 1, 1),
        )

    async def test_repositories(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamFactory(id=11, parent_id=10))
        body = self._body(
            repositories=["github.com/src-d/go-git/alpha", "github.com/src-d/go-git/beta"],
            team_goals=[(10, 42), (11, 43)],
        )
        new_goal_id = (await self._request(json=body))["id"]

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
        body = self._body(
            jira_projects=["P0"],
            jira_issue_types=["Bugs", "tasks", "Story", "Task"],
            team_goals=[(10, 42)],
        )

        new_goal_id = (await self._request(json=body))["id"]
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
        body = self._body(jira_priorities=["High", "low", "high"], team_goals=[(10, 42)])
        new_goal_id = (await self._request(json=body))["id"]
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
        body = self._body(team_goals=[(10, 42), (11, 43, {"p1": 2})], metric_params={"p0": 1})
        new_goal_id = (await self._request(json=body))["id"]
        goal_row = await assert_existing_row(sdb, Goal, id=new_goal_id, account_id=1)
        assert goal_row[Goal.metric_params.name] == {"p0": 1}

        tg_10_row = await assert_existing_row(sdb, TeamGoal, goal_id=new_goal_id, team_id=10)
        assert tg_10_row[TeamGoal.metric_params.name] is None

        tg_11_row = await assert_existing_row(sdb, TeamGoal, goal_id=new_goal_id, team_id=11)
        assert tg_11_row[TeamGoal.metric_params.name] == {"p1": 2}


class BaseUpdateGoalTest(BaseGoalTest):
    async def _request(
        self,
        goal_id: int,
        assert_status: int = 200,
        user_id: Optional[str] = _USER_ID,
        **kwargs: Any,
    ) -> dict:
        path = f"/private/align/goal/{goal_id}"
        with force_request_auth(user_id, self.headers) as headers:
            response = await self.client.request(
                method="PUT", path=path, headers=headers, **kwargs,
            )
        assert response.status == assert_status
        return await response.json()

    def _body(
        self,
        name: str = "My Goal",
        archived: bool = False,
        team_goals: Sequence[tuple[int, MetricValue] | tuple[int, MetricValue, dict]] = (),
        **kwargs,
    ) -> dict:
        request = GoalUpdateRequest(
            name=name,
            archived=archived,
            team_goals=_team_goals_assoc_from_tuples(*team_goals),
            **kwargs,
        )
        return request.to_dict()


class TestUpdateGoalErrors(BaseUpdateGoalTest):
    async def test_not_found(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100), TeamFactory(id=10))
        res = await self._request(101, 404, json=self._body(team_goals=[(10, 1)]))
        assert res["detail"] == "Goal 101 not found or access denied."

    async def test_dupl_team_ids(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100), TeamFactory(id=10))
        body = self._body(team_goals=[(10, 10), (10, 20)])
        res = await self._request(100, 400, json=body)
        assert res["detail"] == "Duplicated teams: 10"
        assert res["title"] == "Bad Request"

    async def test_invalid_team_target(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100), TeamFactory(id=10))
        body = self._body(team_goals=[(10, False)])
        res = await self._request(100, 400, json=body)
        assert res["title"] == "Bad Request"

    async def test_goal_account_mismatch(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100, account_id=2), TeamFactory(id=10))
        body = self._body(team_goals=[(10, 10)])
        res = await self._request(100, 404, json=body)
        assert res["detail"] == "Goal 100 not found or access denied."
        assert res["title"] == "Goal not found"

    async def test_team_account_mismatch(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100, name="G0"), TeamFactory(id=10, owner_id=2))
        body = self._body(team_goals=[(10, 10)])
        res = await self._request(100, 404, json=body)
        assert res["detail"] == "Team-s don't exist or access denied: 10"
        await assert_missing_row(sdb, TeamGoal, team_id=10)

    async def test_invalid_team_id(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100, name="G0"), TeamFactory(id=10))
        body = self._body(team_goals=[(10, 1), (11, 1), (12, 1)])
        res = await self._request(100, 404, json=body)
        assert res["detail"] == "Team-s don't exist or access denied: 11,12"
        await assert_missing_row(sdb, TeamGoal, team_id=10)

    async def test_default_user_forbidden(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100), TeamFactory(id=10))
        body = self._body(team_goals=[(10, 1)])
        res = await self._request(100, 403, user_id=None, json=body)
        assert res["title"] == "Forbidden"

    async def test_bad_repositories(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100), TeamFactory(id=10))
        body = self._body(team_goals=[(10, 1)], repositories=["github.com/athenianco/xxx"])
        res = await self._request(100, 400, json=body)
        assert res["pointer"] == ".repositories"
        row = await assert_existing_row(sdb, Goal, id=100)
        assert row[Goal.repositories.name] is None

    async def test_no_team_goals(self, sdb: Database) -> None:
        await models_insert(
            sdb, GoalFactory(id=100), TeamFactory(id=10), TeamGoalFactory(team_id=10, goal_id=100),
        )
        body = self._body(team_goals=[])
        res = await self._request(100, 400, json=body)
        assert res["title"] == "Bad Request"
        assert "team_goals" in res["detail"]


class TestUpdateGoal(BaseUpdateGoalTest):
    async def test_deletions(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            *[TeamFactory(id=_id) for _id in (10, 20, 30)],
            GoalFactory(id=101, name="G0"),
            TeamGoalFactory(team_id=10, goal_id=101),
            TeamGoalFactory(team_id=20, goal_id=101),
            TeamGoalFactory(team_id=30, goal_id=101, target=41),
        )
        body = self._body(name="G1", team_goals=((30, 42),))
        await self._request(101, json=body)
        await assert_missing_row(sdb, TeamGoal, team_id=10)
        await assert_missing_row(sdb, TeamGoal, team_id=20)
        await assert_existing_row(sdb, Goal, name="G1", id=101)
        await assert_existing_row(sdb, Team, id=10)
        await assert_existing_row(sdb, Team, id=20)
        await assert_existing_row(sdb, TeamGoal, team_id=30, target=42)

    @freeze_time("2022-04-01T09:30:00")
    async def test_some_changes(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100, archived=False),
            *[TeamFactory(id=_id) for _id in (10, 20, 30)],
            TeamGoalFactory(team_id=10, goal_id=100),
            TeamGoalFactory(
                team_id=20, goal_id=100, target=1, created_at=dt(1, 1, 2), updated_at=dt(1, 1, 3),
            ),
        )
        body = self._body(team_goals=[(20, 8), (30, 7)])
        res = await self._request(100, json=body)
        assert res == {"id": 100}

        await assert_existing_row(sdb, Goal, id=100, archived=False)
        await assert_missing_row(sdb, TeamGoal, team_id=10, goal_id=100)

        team_goal_20_row = await assert_existing_row(sdb, TeamGoal, team_id=20, goal_id=100)
        assert team_goal_20_row[TeamGoal.target.name] == 8
        assert ensure_db_datetime_tz(team_goal_20_row[TeamGoal.created_at.name], sdb) == dt(
            1, 1, 2,
        )
        assert ensure_db_datetime_tz(team_goal_20_row[TeamGoal.updated_at.name], sdb) == dt(
            2022, 4, 1, 9, 30,
        )

        team_goal_30_row = await assert_existing_row(sdb, TeamGoal, team_id=30, goal_id=100)
        assert team_goal_30_row[TeamGoal.target.name] == 7

    async def test_update_archived(self, sdb: Database) -> None:
        await models_insert(sdb, *self._sample_goal_models(archived=False))
        body = self._body(archived=True, team_goals=[(10, 1)])
        await self._request(2, json=body)
        await assert_existing_row(sdb, Goal, id=2, archived=True)
        await assert_existing_row(sdb, TeamGoal, team_id=10, goal_id=2)

    async def test_update_name(self, sdb: Database) -> None:
        await models_insert(sdb, *self._sample_goal_models(name="G0"))
        body = self._body(name="G1", team_goals=[(10, 1)])
        await self._request(2, json=body)
        await assert_existing_row(sdb, Goal, id=2, name="G1")

    async def test_update_repositories(self, sdb: Database) -> None:
        await models_insert(sdb, *self._sample_goal_models())
        body = self._body(team_goals=[(10, 1)], repositories=["github.com/src-d/go-git"])
        await self._request(2, json=body)
        row = await assert_existing_row(sdb, Goal, id=2)
        assert row[Goal.repositories.name] == [[40550, ""]]

        body = self._body(team_goals=[(10, 1)], repositories=None)
        await self._request(2, json=body)
        row = await assert_existing_row(sdb, Goal, id=2)
        assert row[Goal.repositories.name] is None

    async def test_update_repos_and_change_team_goals(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100),
            TeamFactory(id=10),
            TeamFactory(id=11, parent_id=10),
            TeamGoalFactory(team_id=10, goal_id=100, target=1, repositories=[[40550, "alpha"]]),
        )
        body = self._body(
            repositories=["github.com/src-d/go-git", "github.com/src-d/lapjv"],
            team_goals=[(10, 2), (11, 3)],
        )
        res = await self._request(100, json=body)
        assert res == {"id": 100}

        row = await assert_existing_row(sdb, Goal, id=100)
        assert row[Goal.repositories.name] == [[40550, ""], [39652768, ""]]

        tg_10 = await assert_existing_row(sdb, TeamGoal, goal_id=100, team_id=10)
        assert tg_10[TeamGoal.repositories.name] == [[40550, ""], [39652768, ""]]

        tg_11 = await assert_existing_row(sdb, TeamGoal, goal_id=100, team_id=11)
        assert tg_11[TeamGoal.repositories.name] == [[40550, ""], [39652768, ""]]

    async def test_new_assigned_teams_inherit_goal_values(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100, repositories=[[40550, ""]]),
            TeamFactory(id=10),
            TeamFactory(id=11, parent_id=10),
            TeamGoalFactory(team_id=10, goal_id=100, target=1, repositories=[[40550, ""]]),
        )
        body = self._body(repositories=["github.com/src-d/go-git"], team_goals=[(10, 1), (11, 2)])
        res = await self._request(100, json=body)
        assert res == {"id": 100}

        row = await assert_existing_row(sdb, Goal, id=100)
        assert row[Goal.repositories.name] == [[40550, ""]]
        team_goal_row_10 = await assert_existing_row(sdb, TeamGoal, goal_id=100, team_id=10)
        assert team_goal_row_10[TeamGoal.repositories.name] == [[40550, ""]]
        assert team_goal_row_10[TeamGoal.target.name] == 1
        team_goal_row_11 = await assert_existing_row(sdb, TeamGoal, goal_id=100, team_id=11)
        assert team_goal_row_11[TeamGoal.repositories.name] == [[40550, ""]]
        assert team_goal_row_11[TeamGoal.target.name] == 2

    async def test_update_jira_fields(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100, jira_priorities=["high"]),
            TeamFactory(id=10),
            TeamGoalFactory(
                team_id=10, goal_id=100, jira_projects=["P2"], jira_priorities=["high"],
            ),
        )
        body = self._body(
            team_goals=[(10, 2)],
            jira_projects=["P0", "P1"],
            jira_priorities=None,
            jira_issue_types=["Tasks", "bugs"],
        )
        await self._request(100, json=body)
        row = await assert_existing_row(sdb, Goal, id=100)
        assert row[Goal.jira_projects.name] == ["P0", "P1"]
        assert row[Goal.jira_priorities.name] is None
        assert row[Goal.jira_issue_types.name] == ["bug", "task"]

        tg_row = await assert_existing_row(sdb, TeamGoal, goal_id=100, team_id=10)
        # team goal jira_projects are overwritten
        assert tg_row[TeamGoal.jira_projects.name] == ["P0", "P1"]
        assert tg_row[TeamGoal.jira_priorities.name] is None
        assert tg_row[TeamGoal.jira_issue_types.name] == ["bug", "task"]

    async def test_change_unique_team_goal(self, sdb: Database) -> None:
        await models_insert(
            sdb, GoalFactory(id=20), TeamFactory(id=10), TeamGoalFactory(team_id=10, goal_id=20),
        )
        body = self._body(team_goals=[(10, 2, {"threshold": 1.5})])
        await self._request(20, json=body)
        team_10_row = await assert_existing_row(sdb, TeamGoal, team_id=10, goal_id=20)
        assert team_10_row[TeamGoal.target.name] == 2
        assert team_10_row[TeamGoal.metric_params.name] == {"threshold": 1.5}

    async def test_set_teams_metric_params(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100, metric_params={"threshold": 1}),
            *[TeamFactory(id=id_) for id_ in (10, 20, 30)],
            TeamGoalFactory(team_id=10, goal_id=100, metric_params={"threshold": 2}),
            TeamGoalFactory(team_id=30, goal_id=100, metric_params={"threshold": 3}),
        )

        body = self._body(
            team_goals=[(10, 1, {"threshold": 10}), (20, 1), (30, 1, {"threshold": 20})],
        )
        res = await self._request(100, json=body)
        assert res["id"] == 100
        await assert_existing_row(
            sdb, TeamGoal, goal_id=100, team_id=10, metric_params={"threshold": 10},
        )
        row_20 = await assert_existing_row(sdb, TeamGoal, goal_id=100, team_id=20)
        assert row_20[TeamGoal.metric_params.name] is None
        await assert_existing_row(
            sdb, TeamGoal, goal_id=100, team_id=30, metric_params={"threshold": 20},
        )

    @classmethod
    def _sample_goal_models(cls, **kwargs: Any) -> Sequence[Base]:
        return (
            GoalFactory(id=2, **kwargs),
            TeamFactory(id=10),
            TeamGoalFactory(team_id=10, goal_id=2),
        )
