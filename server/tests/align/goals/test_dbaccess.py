from operator import itemgetter

from freezegun import freeze_time
import pytest
import sqlalchemy as sa

from athenian.api.align.exceptions import (
    GoalMutationError,
    GoalNotFoundError,
    GoalTemplateNotFoundError,
)
from athenian.api.align.goals.dbaccess import (
    GoalColumnAlias,
    create_default_goal_templates,
    delete_empty_goals,
    delete_goal_template_from_db,
    delete_team_goals,
    dump_goal_repositories,
    fetch_goal_account,
    fetch_team_goals,
    get_goal_template_from_db,
    get_goal_templates_from_db,
    insert_goal_template,
    parse_goal_repositories,
    resolve_goal_repositories,
    update_goal,
    update_goal_template_in_db,
)
from athenian.api.align.goals.templates import TEMPLATES_COLLECTION
from athenian.api.db import Database, ensure_db_datetime_tz, integrity_errors
from athenian.api.internal.prefixer import RepositoryName, RepositoryReference
from athenian.api.models.state.models import Goal, GoalTemplate, TeamGoal
from tests.controllers.test_prefixer import mk_prefixer
from tests.testutils.db import (
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


class TestParseGoalRepositories:
    def test_unset(self) -> None:
        assert parse_goal_repositories(None) is None

    def test_empty(self) -> None:
        assert parse_goal_repositories([]) == []

    def test_base(self) -> None:
        val: list[list] = [[123, None], [456, "logical"]]
        identities = parse_goal_repositories(val)
        assert identities is not None
        assert all(isinstance(ident, RepositoryReference) for ident in identities)
        assert identities[0].node_id == 123
        assert identities[0].logical_name is None
        assert identities[1].node_id == 456
        assert identities[1].logical_name == "logical"


class TestDumpGoalRepositories:
    def test_none(self) -> None:
        assert dump_goal_repositories(None) is None

    def test_some_identities(self) -> None:
        idents = [RepositoryReference(1, "a"), RepositoryReference(2, None)]
        assert dump_goal_repositories(idents) == [(1, "a"), (2, None)]


class TestResolveGoalRepositories:
    def test_empty(self) -> None:
        prefixer = mk_prefixer()
        assert resolve_goal_repositories([], prefixer) == ()

    def test_base(self) -> None:
        prefixer = mk_prefixer(
            repo_node_to_prefixed_name={
                1: "github.com/athenianco/a",
                2: "github.com/athenianco/b",
            },
        )

        res = resolve_goal_repositories([(1, None), (2, None), (2, "logic")], prefixer)

        assert res == (
            RepositoryName("github.com", "athenianco", "a", None),
            RepositoryName("github.com", "athenianco", "b", None),
            RepositoryName("github.com", "athenianco", "b", "logic"),
        )

    def test_unknown_ids_are_ignored(self) -> None:
        prefixer = mk_prefixer(repo_node_to_prefixed_name={1: "github.com/athenianco/a"})
        res = resolve_goal_repositories([(1, None), (2, None)], prefixer)

        assert res == (RepositoryName("github.com", "athenianco", "a", None),)
