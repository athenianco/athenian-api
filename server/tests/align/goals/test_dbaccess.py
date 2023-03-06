from datetime import timedelta
from functools import partial
from operator import itemgetter

from freezegun import freeze_time
import pytest

from athenian.api.align.exceptions import (
    GoalMutationError,
    GoalNotFoundError,
    GoalTemplateNotFoundError,
    TeamGoalNotFoundError,
)
from athenian.api.align.goals.dbaccess import (
    GoalColumnAlias,
    TeamGoalTargetAssignment,
    assign_team_goals,
    convert_metric_params_datatypes,
    create_default_goal_templates,
    delete_goal_template_from_db,
    delete_team_goals,
    fetch_goal_account,
    fetch_team_goals,
    get_goal_template_from_db,
    get_goal_templates_from_db,
    insert_goal_template,
    replace_team_goals,
    unassign_team_from_goal,
    unassign_team_from_goal_recursive,
    update_goal,
    update_goal_template_in_db,
)
from athenian.api.align.goals.templates import TEMPLATES_COLLECTION
from athenian.api.db import Database, ensure_db_datetime_tz, integrity_errors
from athenian.api.models.state.models import Goal, GoalTemplate, Team, TeamGoal
from tests.testutils.db import (
    SKIP_MODEL_FIELD,
    assert_existing_row,
    assert_existing_rows,
    assert_missing_row,
    models_insert,
    transaction_conn,
)
from tests.testutils.factory.state import (
    AccountFactory,
    GoalFactory,
    GoalTemplateFactory,
    TeamFactory,
    TeamGoalFactory,
)
from tests.testutils.time import dt


class TestFetchGoalAccount:
    async def test_success(self, sdb: Database) -> None:
        await models_insert(sdb, AccountFactory(id=456), GoalFactory(account_id=456, id=123))
        assert await fetch_goal_account(123, sdb) == 456

    async def test_not_found(self, sdb: Database) -> None:
        with pytest.raises(GoalNotFoundError):
            await fetch_goal_account(123, sdb)


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


class TestAssignTeamGoals:
    async def test_update_metric_params(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=20),
            TeamFactory(id=10),
            TeamFactory(id=11),
            TeamFactory(id=12),
            TeamGoalFactory(team_id=10, goal_id=20, metric_params=None),
        )
        assignments = [
            TeamGoalTargetAssignment(10, 1, {"threshold": 2}),
            TeamGoalTargetAssignment(11, 1, {"threshold": 3}),
            TeamGoalTargetAssignment(12, 1, None),
        ]
        async with transaction_conn(sdb) as sdb_conn:
            await assign_team_goals(1, 20, assignments, sdb_conn)

        await assert_existing_row(
            sdb, TeamGoal, team_id=10, goal_id=20, metric_params={"threshold": 2},
        )
        await assert_existing_row(
            sdb, TeamGoal, team_id=11, goal_id=20, metric_params={"threshold": 3},
        )
        row_12 = await assert_existing_row(sdb, TeamGoal, team_id=12, goal_id=20)
        assert row_12[TeamGoal.metric_params.name] is None


class TestReplaceTeamGoals:
    async def test_base(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=20),
            *[TeamFactory(id=_id) for _id in (10, 11, 12)],
            TeamGoalFactory(team_id=10, goal_id=20, target=1, metric_params={"threshold": 1}),
            TeamGoalFactory(team_id=11, goal_id=20),
        )

        assignments = [
            TeamGoalTargetAssignment(10, 2, None),
            TeamGoalTargetAssignment(12, 2, {"threshold": 2}),
        ]
        async with transaction_conn(sdb) as sdb_conn:
            await replace_team_goals(1, 20, assignments, sdb_conn)

        tg_10 = await assert_existing_row(sdb, TeamGoal, team_id=10, goal_id=20)
        assert tg_10[TeamGoal.target.name] == 2
        assert tg_10[TeamGoal.metric_params.name] is None

        await assert_missing_row(sdb, TeamGoal, team_id=11, goal_id=20)

        tg_12 = await assert_existing_row(sdb, TeamGoal, team_id=12, goal_id=20)
        assert tg_12[TeamGoal.target.name] == 2
        assert tg_12[TeamGoal.metric_params.name] == {"threshold": 2}

    async def test_goal_of_different_account(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=20),
            TeamFactory(id=10),
            TeamFactory(id=11),
            TeamGoalFactory(team_id=10, goal_id=20, target=1),
        )
        assignments = [TeamGoalTargetAssignment(11, 2, None)]
        async with transaction_conn(sdb) as sdb_conn:
            with pytest.raises(GoalMutationError):
                await replace_team_goals(2, 20, assignments, sdb_conn)

        await assert_missing_row(sdb, TeamGoal, team_id=11)
        await assert_existing_row(sdb, TeamGoal, team_id=10)

    @freeze_time("2022-10-26")
    async def test_updated_not_changed_if_unmodified(self, sdb: Database) -> None:
        TGF = partial(
            TeamGoalFactory, goal_id=20, created_at=dt(2001, 1, 1), updated_at=dt(2002, 1, 1),
        )

        await models_insert(
            sdb,
            GoalFactory(id=20),
            *[TeamFactory(id=_id) for _id in (10, 11, 12, 13)],
            *[
                TGF(team_id=tid, target=1, metric_params={"threshold": "1s"})
                for tid in (10, 11, 12)
            ],
        )
        assignments = [
            TeamGoalTargetAssignment(10, 1, {"threshold": "1s"}),  # not changed
            TeamGoalTargetAssignment(11, 1, {"threshold": "2s"}),  # params changed
            TeamGoalTargetAssignment(12, 2, {"threshold": "1s"}),  # target changed
            TeamGoalTargetAssignment(13, 1, {"threshold": "1s"}),  # added
        ]
        async with transaction_conn(sdb) as sdb_conn:
            await replace_team_goals(1, 20, assignments, sdb_conn)

        tg_10 = await assert_existing_row(sdb, TeamGoal, team_id=10, goal_id=20, target=1)
        assert tg_10[TeamGoal.metric_params.name] == {"threshold": "1s"}
        assert ensure_db_datetime_tz(tg_10[TeamGoal.created_at.name], sdb) == dt(2001, 1, 1)
        assert ensure_db_datetime_tz(tg_10[TeamGoal.updated_at.name], sdb) == dt(2002, 1, 1)

        tg_11 = await assert_existing_row(sdb, TeamGoal, team_id=11, goal_id=20, target=1)
        assert tg_11[TeamGoal.metric_params.name] == {"threshold": "2s"}
        assert ensure_db_datetime_tz(tg_11[TeamGoal.created_at.name], sdb) == dt(2001, 1, 1)
        assert ensure_db_datetime_tz(tg_11[TeamGoal.updated_at.name], sdb) == dt(2022, 10, 26)

        tg_12 = await assert_existing_row(sdb, TeamGoal, team_id=12, goal_id=20, target=2)
        assert tg_12[TeamGoal.metric_params.name] == {"threshold": "1s"}
        assert ensure_db_datetime_tz(tg_12[TeamGoal.created_at.name], sdb) == dt(2001, 1, 1)
        assert ensure_db_datetime_tz(tg_12[TeamGoal.updated_at.name], sdb) == dt(2022, 10, 26)

        tg_13 = await assert_existing_row(sdb, TeamGoal, team_id=13, goal_id=20, target=1)
        assert tg_13[TeamGoal.metric_params.name] == {"threshold": "1s"}
        assert ensure_db_datetime_tz(tg_13[TeamGoal.created_at.name], sdb) == dt(2022, 10, 26)
        assert ensure_db_datetime_tz(tg_13[TeamGoal.updated_at.name], sdb) == dt(2022, 10, 26)


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

    async def test_repositories_column(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            GoalFactory(id=20, repositories=[[1, None], [2, "logic"]]),
            TeamGoalFactory(team_id=10, goal_id=20, repositories=None),
        )
        rows = await fetch_team_goals(1, [10], sdb)
        assert len(rows) == 1
        row = rows[0]
        assert row[TeamGoal.repositories.name] is None
        assert row[GoalColumnAlias.REPOSITORIES.value] == [[1, None], [2, "logic"]]

    async def test_jira_columns(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            GoalFactory(id=20, jira_projects=["DEV", "DR"], jira_issue_types=["Task"]),
            TeamGoalFactory(
                team_id=10, goal_id=20, jira_projects=["DEV"], jira_priorities=["P0", "P1"],
            ),
        )
        rows = await fetch_team_goals(1, [10], sdb)
        assert len(rows) == 1
        row = rows[0]
        assert row[TeamGoal.jira_projects.name] == ["DEV"]
        assert row[TeamGoal.jira_priorities.name] == ["P0", "P1"]
        assert row[TeamGoal.jira_issue_types.name] is None

        assert row[GoalColumnAlias.JIRA_PROJECTS.value] == ["DEV", "DR"]
        assert row[GoalColumnAlias.JIRA_PRIORITIES.value] is None
        assert row[GoalColumnAlias.JIRA_ISSUE_TYPES.value] == ["Task"]

    async def test_metric_params_column(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamFactory(id=11, parent_id=10),
            GoalFactory(id=20, metric_params=None),
            GoalFactory(id=21, metric_params={"a": "b"}),
            TeamGoalFactory(team_id=10, goal_id=20, metric_params=None),
            TeamGoalFactory(team_id=11, goal_id=20, metric_params={"foo": "bar"}),
            TeamGoalFactory(team_id=10, goal_id=21, metric_params=None),
            TeamGoalFactory(team_id=11, goal_id=21, metric_params={"foo0": 1, "foo1": 0}),
        )
        rows = await fetch_team_goals(1, [10, 11], sdb)
        assert len(rows) == 4

        assert rows[0][TeamGoal.goal_id.name] == 20
        assert rows[0][TeamGoal.team_id.name] == 10
        assert rows[0][TeamGoal.metric_params.name] is None
        assert rows[0][GoalColumnAlias.METRIC_PARAMS.value] is None

        assert rows[1][TeamGoal.goal_id.name] == 20
        assert rows[1][TeamGoal.team_id.name] == 11
        assert rows[1][TeamGoal.metric_params.name] == {"foo": "bar"}
        assert rows[1][GoalColumnAlias.METRIC_PARAMS.value] is None

        assert rows[2][TeamGoal.goal_id.name] == 21
        assert rows[2][TeamGoal.team_id.name] == 10
        assert rows[2][TeamGoal.metric_params.name] is None
        assert rows[2][GoalColumnAlias.METRIC_PARAMS.value] == {"a": "b"}

        assert rows[3][TeamGoal.goal_id.name] == 21
        assert rows[3][TeamGoal.team_id.name] == 11
        assert rows[3][TeamGoal.metric_params.name] == {"foo0": 1, "foo1": 0}
        assert rows[3][GoalColumnAlias.METRIC_PARAMS.value] == {"a": "b"}


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
        rows = await get_goal_templates_from_db(1, False, sdb)
        assert len(rows) == 2
        assert {r[GoalTemplate.id.name] for r in rows} == {102, 103}

    async def test_exclude_with_metric_params(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalTemplateFactory(id=102, metric_params=SKIP_MODEL_FIELD),
            GoalTemplateFactory(id=103, metric_params={"threshold": 2}),
            GoalTemplateFactory(id=104),
        )
        rows = await get_goal_templates_from_db(1, True, sdb)
        assert len(rows) == 2
        assert {r[GoalTemplate.id.name] for r in rows} == {102, 104}


class TestInsertGoalTemplate:
    async def test_base(self, sdb: Database) -> None:
        await insert_goal_template(
            sdb, account_id=1, name="T", metric="m", repositories=[[1, None]],
        )
        await assert_existing_row(
            sdb, GoalTemplate, account_id=1, name="T", metric="m", repositories=[[1, None]],
        )

    async def test_duplicated_name(self, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(account_id=1, name="T"))
        with pytest.raises(integrity_errors):
            await insert_goal_template(sdb, account_id=1, name="T", metric="m0")

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
        await update_goal_template_in_db(120, sdb, name="Tmpl new")
        row = await assert_existing_row(sdb, GoalTemplate, id=120, name="Tmpl new")
        assert ensure_db_datetime_tz(row[GoalTemplate.updated_at.name], sdb) == dt(2012, 10, 26)

    async def test_not_found(self, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(id=120, name="T0"))
        await update_goal_template_in_db(121, sdb, name="T1")
        await assert_existing_row(sdb, GoalTemplate, id=120, name="T0")


class TestCreateDefaultGoalTemplates:
    async def test_base(self, sdb: Database) -> None:
        await create_default_goal_templates(1, sdb)
        rows = await assert_existing_rows(sdb, GoalTemplate)
        assert len(rows) == len(TEMPLATES_COLLECTION)
        row_by_name = {row[GoalTemplate.name.name]: row for row in rows}

        for template_def in TEMPLATES_COLLECTION:
            row = row_by_name[template_def["name"]]
            assert row[GoalTemplate.name.name] == template_def["name"]
            assert row[GoalTemplate.metric.name] == template_def["metric"]
            assert row[GoalTemplate.metric_params.name] == template_def.get("metric_params")

    async def test_ignore_existing_names(self, sdb: Database) -> None:
        await models_insert(sdb, GoalTemplateFactory(name=TEMPLATES_COLLECTION[0]["name"], id=555))
        await create_default_goal_templates(1, sdb)

        rows = await assert_existing_rows(sdb, GoalTemplate)
        assert len(rows) == len(TEMPLATES_COLLECTION)
        await assert_existing_rows(sdb, GoalTemplate, id=555, name=TEMPLATES_COLLECTION[0]["name"])


class TestConvertMetricParamsDatatypes:
    def test_none(self) -> None:
        assert convert_metric_params_datatypes(None) == {}

    def test_empty_dict(self) -> None:
        assert convert_metric_params_datatypes({}) == {}

    def test_int_threshold(self) -> None:
        assert convert_metric_params_datatypes({"threshold": 23}) == {"threshold": 23}

    def test_time_duration_threshold(self) -> None:
        metric_params = {"threshold": "100s"}
        res = convert_metric_params_datatypes(metric_params)

        assert res == {"threshold": timedelta(seconds=100)}
        assert res is not metric_params


class TestUnassignTeamFromGoal:
    async def test_assignment_not_existing(self, sdb: Database) -> None:
        await models_insert(
            sdb, GoalFactory(id=1), TeamFactory(id=9), TeamGoalFactory(goal_id=1, team_id=9),
        )

        params = ((1, 2, 9), (1, 1, 10), (2, 1, 9))
        for account_id, goal_id, team in params:
            async with transaction_conn(sdb) as sdb_conn:
                with pytest.raises(TeamGoalNotFoundError):
                    await unassign_team_from_goal(account_id, goal_id, team, sdb_conn)

    async def test_base(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=1),
            TeamFactory(id=8),
            TeamFactory(id=9),
            TeamGoalFactory(goal_id=1, team_id=8),
            TeamGoalFactory(goal_id=1, team_id=9),
        )

        async with transaction_conn(sdb) as sdb_conn:
            goal_still_exists = await unassign_team_from_goal(1, 1, 8, sdb_conn)

        assert goal_still_exists
        await assert_existing_row(sdb, Goal, id=1, account_id=1)
        await assert_existing_row(sdb, TeamGoal, team_id=9, goal_id=1)
        await assert_missing_row(sdb, TeamGoal, team_id=8, goal_id=1)

    async def test_last_team_unassigned(self, sdb: Database) -> None:
        await models_insert(
            sdb, GoalFactory(id=1), TeamFactory(id=9), TeamGoalFactory(goal_id=1, team_id=9),
        )

        async with transaction_conn(sdb) as sdb_conn:
            goal_still_exists = await unassign_team_from_goal(1, 1, 9, sdb_conn)

        assert not goal_still_exists
        await assert_missing_row(sdb, Goal, id=1)


class TestUnassignTeamFromGoalRecursive:
    async def test_base(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=1),
            TeamFactory(id=8),
            TeamFactory(id=9, parent_id=8),
            TeamFactory(id=10, parent_id=9),
            TeamFactory(id=11, parent_id=9),
            *[TeamGoalFactory(goal_id=1, team_id=t) for t in [8, 9, 10, 11]],
        )

        async with transaction_conn(sdb) as sdb_conn:
            goal_still_exists = await unassign_team_from_goal_recursive(1, 1, 9, sdb_conn)

        assert goal_still_exists
        await assert_existing_row(sdb, Goal, id=1, account_id=1)
        for t in (9, 10, 11):
            await assert_missing_row(sdb, TeamGoal, team_id=t, goal_id=1)
        await assert_existing_row(sdb, TeamGoal, goal_id=1, team_id=8)
        await assert_existing_row(sdb, Team, id=9)

    async def test_last_team_unassigned(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=1),
            TeamFactory(id=8),
            TeamFactory(id=9, parent_id=8),
            TeamFactory(id=10, parent_id=9),
            *[TeamGoalFactory(goal_id=1, team_id=t) for t in [9, 10]],
        )

        async with transaction_conn(sdb) as sdb_conn:
            goal_still_exists = await unassign_team_from_goal_recursive(1, 1, 9, sdb_conn)

        assert not goal_still_exists
        await assert_missing_row(sdb, Goal, id=1, account_id=1)
        for t in (9, 10):
            await assert_missing_row(sdb, TeamGoal, team_id=t, goal_id=1)
        await assert_existing_row(sdb, Team, id=9)
