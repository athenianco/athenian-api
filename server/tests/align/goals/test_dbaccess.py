from operator import itemgetter

from freezegun import freeze_time
import pytest
import sqlalchemy as sa

from athenian.api.align.exceptions import GoalMutationError, GoalTemplateNotFoundError
from athenian.api.align.goals.dbaccess import (
    create_default_goal_templates,
    delete_empty_goals,
    delete_goal_template_from_db,
    delete_team_goals,
    fetch_team_goals,
    get_goal_template_from_db,
    get_goal_templates_from_db,
    insert_goal_template,
    update_goal,
    update_goal_template_in_db,
)
from athenian.api.align.goals.templates import TEMPLATES_COLLECTION
from athenian.api.db import Database, ensure_db_datetime_tz, integrity_errors
from athenian.api.models.state.models import Goal, GoalTemplate, TeamGoal
from tests.testutils.db import (
    assert_existing_row,
    assert_existing_rows,
    assert_missing_row,
    models_insert,
    transaction_conn,
)
from tests.testutils.factory.state import (
    GoalFactory,
    GoalTemplateFactory,
    TeamFactory,
    TeamGoalFactory,
)
from tests.testutils.time import dt


class TestDeleteTeamGoals:
    async def test_cannot_delete_all_goals(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=20),
            TeamFactory(id=10),
            TeamFactory(id=11),
            TeamGoalFactory(team_id=10, goal_id=20),
            TeamGoalFactory(team_id=11, goal_id=20),
        )
        async with transaction_conn(sdb) as sdb_conn:
            with pytest.raises(GoalMutationError):
                await delete_team_goals(1, 20, [10, 11], sdb_conn)

        await assert_existing_row(sdb, TeamGoal, team_id=10, goal_id=20)
        await assert_existing_row(sdb, TeamGoal, team_id=11, goal_id=20)
        await assert_existing_row(sdb, Goal, id=20)

        async with transaction_conn(sdb) as sdb_conn:
            await delete_team_goals(1, 20, [11], sdb_conn)

        await assert_existing_row(sdb, TeamGoal, team_id=10, goal_id=20)
        await assert_missing_row(sdb, TeamGoal, team_id=11, goal_id=20)


class TestUpdateGoal:
    async def test_set_archived(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=10, archived=False))
        await self._update(1, 10, sdb, archived=True)

        goal = await assert_existing_row(sdb, Goal, id=10)
        assert goal[Goal.archived.name]

        await self._update(1, 10, sdb, archived=True)
        goal = await assert_existing_row(sdb, Goal, id=10)
        assert goal[Goal.archived.name]

        await self._update(1, 10, sdb, archived=False)
        goal = await assert_existing_row(sdb, Goal, id=10)
        assert not goal[Goal.archived.name]

    async def test_goal_not_found(self, sdb: Database) -> None:
        with pytest.raises(GoalMutationError):
            await self._update(1, 1, sdb, archived=False)

    @classmethod
    async def _update(cls, acc_id: int, goal_id: int, sdb: Database, *, archived: bool) -> None:
        async with transaction_conn(sdb) as sdb_conn:
            await update_goal(acc_id, goal_id, sdb_conn, archived=archived)


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

    async def test_archived_goals_are_excluded(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            GoalFactory(id=20),
            GoalFactory(id=21, archived=True),
            GoalFactory(id=22),
            TeamGoalFactory(team_id=10, goal_id=20),
            TeamGoalFactory(team_id=10, goal_id=21),
        )

        rows = await fetch_team_goals(1, [10], sdb)
        assert [r[Goal.id.name] for r in rows] == [20]


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


class TestGetGoalTemplateFromDB:
    async def test_found(self, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(id=102, name="Foo"))
        row = await get_goal_template_from_db(102, sdb)
        assert row["name"] == "Foo"

    async def test_not_found(self, sdb: Database) -> None:
        with pytest.raises(GoalTemplateNotFoundError):
            await get_goal_template_from_db(102, sdb)


class TestGetGoalTemplatesFromDB:
    async def test_base(self, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(id=102), GoalTemplateFactory(id=103))
        rows = await get_goal_templates_from_db(1, sdb)
        assert len(rows) == 2


class TestInsertGoalTemplate:
    async def test_base(self, sdb: Database) -> None:
        await insert_goal_template(1, "T", "m", sdb)
        await assert_existing_row(sdb, GoalTemplate, account_id=1, name="T", metric="m")

    async def test_duplicated_name(self, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(account_id=1, name="T"))
        with pytest.raises(integrity_errors):
            await insert_goal_template(1, "T", "m0", sdb)

        await assert_missing_row(sdb, GoalTemplate, metric="m")


class TestDeleteGoalTemplateFromDB:
    async def test_delete(self, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(id=120))
        await delete_goal_template_from_db(120, sdb)

    async def test_not_existing(self, sdb: Database) -> None:
        await delete_goal_template_from_db(120, sdb)


class TestUpdateGoalTemplateInDB:
    @freeze_time("2012-10-26")
    async def test_update_name(self, sdb: Database) -> None:
        await models_insert(
            sdb, GoalTemplateFactory(id=120, name="Tmpl 0", updated_at=dt(2012, 10, 23)),
        )
        await update_goal_template_in_db(120, "Tmpl new", sdb)
        row = await assert_existing_row(sdb, GoalTemplate, id=120, name="Tmpl new")
        assert ensure_db_datetime_tz(row[GoalTemplate.updated_at.name], sdb) == dt(2012, 10, 26)

    async def test_not_found(self, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(id=120, name="T0"))
        await update_goal_template_in_db(121, "T1", sdb)
        await assert_existing_row(sdb, GoalTemplate, id=120, name="T0")


class TestCreateDefaultGoalTemplates:
    async def test_base(self, sdb: Database) -> None:
        await create_default_goal_templates(1, sdb)
        rows = await assert_existing_rows(sdb, GoalTemplate)
        assert len(rows) == len(TEMPLATES_COLLECTION)
        assert sorted(r[GoalTemplate.name.name] for r in rows) == sorted(
            template_def["name"] for template_def in TEMPLATES_COLLECTION.values()
        )
        assert sorted(r[GoalTemplate.metric.name] for r in rows) == sorted(
            template_def["metric"] for template_def in TEMPLATES_COLLECTION.values()
        )

    async def test_ignore_existing_names(self, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(name=TEMPLATES_COLLECTION[1]["name"], id=555))
        await create_default_goal_templates(1, sdb)

        rows = await assert_existing_rows(sdb, GoalTemplate)
        assert len(rows) == len(TEMPLATES_COLLECTION)
        await assert_existing_rows(sdb, GoalTemplate, id=555, name=TEMPLATES_COLLECTION[1]["name"])
