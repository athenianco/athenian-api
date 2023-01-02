from typing import Any

from athenian.api.db import Database
from athenian.api.models.state.models import DashboardChart, TeamDashboard
from athenian.api.models.web import PullRequestMetricID
from tests.testutils.db import DBCleaner, assert_existing_row, assert_missing_row, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.state import DashboardChartFactory, TeamDashboardFactory, TeamFactory
from tests.testutils.requester import Requester
from tests.testutils.time import dt


class BaseGetDashboardTest(Requester):
    path = "/private/team/{team_id}/dashboard/{dashboard_id}"

    async def get(self, team_id: int, dashboard_id: int, *args: Any, **kwargs: Any):
        path_kwargs = {"team_id": team_id, "dashboard_id": dashboard_id}
        res = await super().get(*args, path_kwargs=path_kwargs, **kwargs)
        return res


class TestGetDashboardErrors(BaseGetDashboardTest):
    async def test_team_not_found(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=1, team_id=10))
        await self.get(20, 1, 404)

    async def test_dashboard_not_found(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=1, team_id=10))
        await self.get(10, 2, 404)

    async def test_team_account_mismatch(self, sdb: Database) -> None:
        await models_insert(
            sdb, TeamFactory(id=10, owner_id=3), TeamDashboardFactory(id=1, team_id=10),
        )
        await self.get(10, 1, 404)

    async def test_default_dashboard_multiple_are_existing(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=20),
            TeamDashboardFactory(team_id=20),
            TeamDashboardFactory(team_id=20),
        )
        await self.get(20, 0, 500)


class TestGetDashboard(BaseGetDashboardTest):
    async def test_empty_dashboard(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=1, team_id=10))
        res = await self.get_json(10, 1)
        assert res["id"] == 1
        assert res["team"] == 10
        assert res["charts"] == []

    async def test_base(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=1, team_id=10),
            TeamDashboardFactory(id=2, team_id=10),
            DashboardChartFactory(
                id=1,
                dashboard_id=1,
                position=2,
                metric=PullRequestMetricID.PR_CYCLE_COUNT,
                time_interval="P1Y",
                name="chart1",
                description="my chart",
            ),
            DashboardChartFactory(
                id=2,
                dashboard_id=1,
                position=1,
                time_from=dt(2021, 1, 20),
                time_to=dt(2021, 3, 21, 23, 59),
            ),
            DashboardChartFactory(id=3, dashboard_id=2),  # another dashboard, not found in results
        )
        res = await self.get_json(10, 1)
        assert res["id"] == 1
        assert res["team"] == 10

        assert len(charts := res["charts"]) == 2

        assert charts[0]["id"] == 2
        assert charts[0]["date_from"] == "2021-01-20"
        assert charts[0]["date_to"] == "2021-03-20"

        assert charts[1]["id"] == 1
        assert charts[1]["metric"] == PullRequestMetricID.PR_CYCLE_COUNT
        assert charts[1]["name"] == "chart1"
        assert charts[1]["description"] == "my chart"
        assert charts[1]["time_interval"] == "P1Y"
        assert "date_from" not in charts[1]
        assert "date_to" not in charts[1]

    async def test_complex_charts_order(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=1, team_id=10),
            DashboardChartFactory(id=1, position=3, dashboard_id=1),
            DashboardChartFactory(id=2, position=2, dashboard_id=1),
            DashboardChartFactory(id=3, position=5, dashboard_id=1),
            DashboardChartFactory(id=4, position=1, dashboard_id=1),
            DashboardChartFactory(id=5, position=4, dashboard_id=1),
        )
        res = await self.get_json(10, 1)
        assert [chart["id"] for chart in res["charts"]] == [4, 2, 1, 5, 3]

    async def test_default_dashboard_not_existing(self, sdb: Database) -> None:
        await assert_missing_row(sdb, TeamDashboard)
        await models_insert(sdb, TeamFactory(id=11))
        res = await self.get_json(11, 0)

        assert res["team"] == 11
        assert res["charts"] == []
        await assert_existing_row(sdb, TeamDashboard, id=res["id"])
        await assert_missing_row(sdb, DashboardChart, dashboard_id=res["id"])

    async def test_default_dashboard_existing(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=5, team_id=10),
            DashboardChartFactory(id=1, position=2, dashboard_id=5),
            DashboardChartFactory(id=2, position=1, dashboard_id=5),
        )
        res = await self.get_json(10, 0)
        assert res["team"] == 10
        assert res["id"] == 5
        assert [chart["id"] for chart in res["charts"]] == [2, 1]


class TestGetDashboardFilters(BaseGetDashboardTest):
    async def test_repositories(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=5, team_id=10),
            DashboardChartFactory(
                id=1, position=1, dashboard_id=5, repositories=[[3, None], [4, None]],
            ),
            DashboardChartFactory(id=2, position=2, dashboard_id=5, repositories=None),
            DashboardChartFactory(
                id=3, position=3, dashboard_id=5, repositories=[[3, "logical1"]],
            ),
            DashboardChartFactory(id=4, position=4, dashboard_id=5, repositories=[[4, None]]),
        )
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_models = [
                md_factory.RepositoryFactory(node_id=3, full_name="org/repo-A"),
                md_factory.RepositoryFactory(node_id=4, full_name="org/repo-B"),
            ]
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            res = await self.get_json(10, 0)

            assert [chart["id"] for chart in res["charts"]] == [1, 2, 3, 4]

            assert res["charts"][0]["filters"] == {
                "repositories": ["github.com/org/repo-A", "github.com/org/repo-B"],
            }
            assert res["charts"][1].get("filters") is None
            assert res["charts"][2]["filters"] == {
                "repositories": ["github.com/org/repo-A/logical1"],
            }
            assert res["charts"][3]["filters"] == {"repositories": ["github.com/org/repo-B"]}

    async def test_jira(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=5, team_id=10),
            DashboardChartFactory(
                id=1,
                position=1,
                dashboard_id=5,
                jira_issue_types=["bug", "task"],
                jira_projects=["DEV", "PROD"],
                jira_labels=["l0"],
                jira_priorities=["high"],
            ),
        )
        res = await self.get_json(10, 5)
        assert res["charts"][0]["filters"]["jira"] == {
            "issue_types": ["bug", "task"],
            "labels_include": ["l0"],
            "projects": ["DEV", "PROD"],
            "priorities": ["high"],
        }

    async def test_multiple_filters(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=5, team_id=10),
            DashboardChartFactory(
                id=1,
                position=1,
                dashboard_id=5,
                repositories=None,
                environments=["production", "qa"],
                jira_issue_types=["bug"],
            ),
        )

        res = await self.get_json(10, 5)
        assert [chart["id"] for chart in res["charts"]] == [1]

        chart = res["charts"][0]
        assert chart["filters"] == {
            "environments": ["production", "qa"],
            "jira": {"issue_types": ["bug"]},
        }
