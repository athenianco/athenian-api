import sqlalchemy as sa

from athenian.api.align.goals.empty_goals import delete_empty_goals
from athenian.api.db import Database
from athenian.api.models.state.models import Goal
from tests.testutils.db import models_insert
from tests.testutils.factory.state import GoalFactory, TeamFactory, TeamGoalFactory


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
