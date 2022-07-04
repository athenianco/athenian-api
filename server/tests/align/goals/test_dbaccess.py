from operator import itemgetter

import sqlalchemy as sa

from athenian.api.align.goals.dbaccess import delete_empty_goals, fetch_team_goals
from athenian.api.db import Database
from athenian.api.models.state.models import Goal, TeamGoal
from tests.testutils.db import models_insert
from tests.testutils.factory.state import GoalFactory, TeamFactory, TeamGoalFactory


class TestFetchTeamGoals:
    async def test(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamFactory(id=11),
            TeamFactory(id=12),
            GoalFactory(id=20),
            GoalFactory(id=21),
            TeamGoalFactory(team_id=10, goal_id=20),
            TeamGoalFactory(team_id=10, goal_id=21),
            TeamGoalFactory(team_id=11, goal_id=20),
            TeamGoalFactory(team_id=12, goal_id=20),
        )

        rows = await fetch_team_goals(1, [10, 11], sdb)
        assert len(rows) == 3

        assert rows[0][Goal.id.name] == 20
        assert rows[1][Goal.id.name] == 20
        assert rows[2][Goal.id.name] == 21

        # make sorting deterministic also about team_id
        rows = sorted(rows, key=itemgetter(Goal.id.name, TeamGoal.team_id.name))
        assert rows[0][TeamGoal.team_id.name] == 10
        assert rows[1][TeamGoal.team_id.name] == 11
        assert rows[2][TeamGoal.team_id.name] == 10


class TestDeleteEmptyGoals:
    async def test_delete(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            GoalFactory(id=20),
            GoalFactory(id=21),
            GoalFactory(id=22),
            GoalFactory(id=23),
            TeamGoalFactory(team_id=10, goal_id=20),
            TeamGoalFactory(team_id=10, goal_id=22),
        )

        await delete_empty_goals(1, sdb)
        goals = [r[0] for r in await sdb.fetch_all(sa.select(Goal.id).order_by(Goal.id))]
        assert goals == [20, 22]
