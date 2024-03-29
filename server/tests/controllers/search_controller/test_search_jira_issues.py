from datetime import date
from typing import Any, Sequence

from athenian.api.db import Database
from athenian.api.models.web import (
    JIRAMetricID,
    JIRAStatusCategory,
    OrderByDirection,
    SearchJIRAIssuesOrderByIssueTrait,
    SearchJIRAIssuesRequest,
)
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory, state as st_factory
from tests.testutils.factory.common import DEFAULT_ACCOUNT_ID
from tests.testutils.factory.wizards import (
    insert_repo,
    jira_issue_models,
    pr_jira_issue_mappings,
    pr_models,
)
from tests.testutils.requester import Requester
from tests.testutils.time import dt


class BaseSearchJIRAIssuesTest(Requester):
    path = "/private/search/jira_issues"

    def _body(
        self,
        account: int = DEFAULT_ACCOUNT_ID,
        date_from: date | None = date(2016, 1, 1),
        date_to: date | None = date(2016, 4, 1),
        **kwargs: Any,
    ) -> dict:
        req = SearchJIRAIssuesRequest(
            account=account, date_from=date_from, date_to=date_to, **kwargs,
        )
        body = req.to_dict()
        for f in ("date_from", "date_to"):
            if body.get(f):
                body[f] = body[f].isoformat()
        return body

    async def _fetch_ids(self, **kwargs) -> Sequence[str]:
        res = await self.post_json(**kwargs)
        return tuple(issue["id"] for issue in res["jira_issues"])


class TestSearchJIRAIssuesErrors(BaseSearchJIRAIssuesTest):
    async def test_account_mismatch(self, sdb) -> None:
        body = self._body(account=3)
        res = await self.post_json(json=body, assert_status=404)
        assert res["type"] == "/errors/AccountNotFound"


class TestSearchJIRAIssues(BaseSearchJIRAIssuesTest):
    async def test_smoke(self) -> None:
        body = self._body(date_from=date(2020, 1, 1), date_to=date(2020, 4, 1))
        res = await self._fetch_ids(json=body)
        assert len(res) == 431

    async def test_partial_interval(self, sdb: Database) -> None:
        body = self._body(date_from=None, date_to=date(2020, 1, 1))
        res = await self._fetch_ids(json=body)
        assert len(res) == 98

    async def test_no_interval(self, sdb: Database) -> None:
        body = self._body(date_from=None, date_to=None)
        res = await self._fetch_ids(json=body)
        assert len(res) == 1797


class TestSearchJIRAIssuesFiltering(BaseSearchJIRAIssuesTest):
    async def test_projects(self, sdb: Database, mdb_rw: Database) -> None:
        issue_kwargs = {"resolved": dt(2016, 2, 1), "created": dt(2016, 1, 1)}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="P1"),
            md_factory.JIRAProjectFactory(id="2", key="P2"),
            md_factory.JIRAProjectFactory(id="3", key="P3"),
            *jira_issue_models("1", key="P1-1", project_id="1", **issue_kwargs),
            *jira_issue_models("2", key="P1-2", project_id="1", **issue_kwargs),
            *jira_issue_models("3", key="P1-3", project_id="1", **issue_kwargs),
            *jira_issue_models("4", key="P2-1", project_id="2", **issue_kwargs),
            *jira_issue_models("5", key="P2-2", project_id="2", **issue_kwargs),
            *jira_issue_models(
                "6", key="P2-3", project_id="2", resolved=dt(2015, 1, 1), created=dt(2014, 1, 1),
            ),
            *jira_issue_models("7", key="P3-1", project_id="3", **issue_kwargs),
        ]

        body = self._body(filter={"projects": ["P1", "P2", "P4"]})

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            ids = sorted(await self._fetch_ids(json=body))
            assert ids == ["P1-1", "P1-2", "P1-3", "P2-1", "P2-2"]

            body["filter"] = {"projects": ["P1"]}
            ids = sorted(await self._fetch_ids(json=body))
            assert ids == ["P1-1", "P1-2", "P1-3"]

            body["filter"] = {"projects": ["P1", "P3"]}
            ids = sorted(await self._fetch_ids(json=body))
            assert ids == ["P1-1", "P1-2", "P1-3", "P3-1"]

            body["filter"] = {"projects": ["P4"]}
            ids = sorted(await self._fetch_ids(json=body))
            assert ids == []

    async def test_priorities_and_types(self, sdb: Database, mdb_rw: Database) -> None:
        issue_kwargs = {"resolved": dt(2016, 2, 1), "created": dt(2016, 1, 1), "project_id": "1"}
        prio0 = {"priority_id": "00", "priority_name": "p0"}
        prio1 = {"priority_id": "10", "priority_name": "p1"}
        type0 = {"type_id": "00", "type": "T0"}
        type1 = {"type_id": "01", "type": "T1"}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="P1"),
            md_factory.JIRAPriorityFactory(id="00", name="P0"),
            md_factory.JIRAPriorityFactory(id="10", name="P1"),
            md_factory.JIRAIssueTypeFactory(id="00", name="T0"),
            md_factory.JIRAIssueTypeFactory(id="100", name="T1"),
            *jira_issue_models("1", key="P1-1", **prio0, **type0, **issue_kwargs),
            *jira_issue_models("2", key="P1-2", **prio0, **type0, **issue_kwargs),
            *jira_issue_models("3", key="P1-3", **prio0, **type1, **issue_kwargs),
            *jira_issue_models("4", key="P1-4", **prio0, **issue_kwargs),
            *jira_issue_models("5", key="P1-5", **prio0, **type0, **issue_kwargs),
            *jira_issue_models("6", key="P1-6", **prio1, **type1, **issue_kwargs),
            *jira_issue_models("7", key="P1-7", **prio1, **type1, **issue_kwargs),
            *jira_issue_models("8", key="P1-8", **prio1, **issue_kwargs),
            *jira_issue_models("9", key="P1-9", **prio1, **issue_kwargs),
        ]

        body = self._body()

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            body["filter"] = {"issue_types": ["T0"]}
            ids = sorted(await self._fetch_ids(json=body))
            assert ids == ["P1-1", "P1-2", "P1-5"]

            body["filter"] = {"issue_types": ["T1"]}
            ids = sorted(await self._fetch_ids(json=body))
            assert ids == ["P1-3", "P1-6", "P1-7"]

            body["filter"] = {"issue_types": ["T0", "T1"], "priorities": ["P0"]}
            ids = sorted(await self._fetch_ids(json=body))
            assert ids == ["P1-1", "P1-2", "P1-3", "P1-5"]

            body["filter"] = {"issue_types": ["T0"], "priorities": ["P0"]}
            ids = sorted(await self._fetch_ids(json=body))
            assert ids == ["P1-1", "P1-2", "P1-5"]

            body["filter"] = {"issue_types": ["T1"], "priorities": ["P1"]}
            ids = sorted(await self._fetch_ids(json=body))
            assert ids == ["P1-6", "P1-7"]

            body["filter"] = {"issue_types": ["T1"], "priorities": ["P0", "P1"]}
            ids = sorted(await self._fetch_ids(json=body))
            assert ids == ["P1-3", "P1-6", "P1-7"]

    async def test_participants_fixture(self, sdb: Database) -> None:
        body = self._body(
            date_from=date(2019, 1, 1),
            date_to=date(2022, 4, 1),
            with_={"assignees": ["{2}"]},
        )
        await models_insert(
            sdb,
            st_factory.TeamFactory(id=1),
            st_factory.TeamFactory(id=2, members=[40020]),
            st_factory.TeamFactory(id=3, members=[29]),
            st_factory.MappedJIRAIdentityFactory(
                github_user_id=40020, jira_user_id="5de5049e2c5dd20d0f9040c1",
            ),
            st_factory.MappedJIRAIdentityFactory(
                github_user_id=29, jira_user_id="5dd58cb9c7ac480ee5674902",
            ),
        )

        res = await self._fetch_ids(json=body)
        assert len(res) == 537

        body["with"]["assignees"] = ["{3}"]
        res = await self._fetch_ids(json=body)
        assert len(res) == 53

    async def test_stages(self, sdb: Database, mdb_rw: Database) -> None:
        issue_kwargs = {"created": dt(2016, 1, 1), "project_id": "1"}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="P1"),
            md_factory.StatusFactory(id="10", category_name="To Do"),
            md_factory.StatusFactory(id="11", category_name="In Progress"),
            md_factory.StatusFactory(id="12", category_name="Done"),
            *jira_issue_models("1", key="P1-1", **issue_kwargs, status_id="10"),
            *jira_issue_models("2", key="P1-2", **issue_kwargs, status_id="10"),
            *jira_issue_models("3", key="P1-3", **issue_kwargs, status_id="11"),
            *jira_issue_models(
                "4", key="P1-4", **issue_kwargs, status_id="12", resolved=dt(2016, 2, 1),
            ),
            *jira_issue_models(
                "5", key="P1-5", **issue_kwargs, status_id="12", resolved=dt(2016, 2, 1),
            ),
        ]

        body = self._body()

        IN_PROGRESS = JIRAStatusCategory.IN_PROGRESS.value
        DONE = JIRAStatusCategory.DONE.value
        TO_DO = JIRAStatusCategory.TO_DO.value
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            ids = sorted(await self._fetch_ids(json=body))
            assert ids == ["P1-1", "P1-2", "P1-3", "P1-4", "P1-5"]

            body["filter"] = {"status_categories": [DONE]}
            ids = sorted(await self._fetch_ids(json=body))
            assert ids == ["P1-4", "P1-5"]

            body["filter"] = {"status_categories": [IN_PROGRESS, TO_DO]}
            ids = sorted(await self._fetch_ids(json=body))
            assert ids == ["P1-1", "P1-2", "P1-3"]

            body["filter"] = {"status_categories": [IN_PROGRESS, DONE]}
            ids = sorted(await self._fetch_ids(json=body))
            assert ids == ["P1-3", "P1-4", "P1-5"]


class TestSearchJIRAIssuesOrderBy(BaseSearchJIRAIssuesTest):
    async def test_life_time(self, mdb_rw: Database) -> None:
        issue_kwargs = {"created": dt(2016, 1, 1), "project_id": "1"}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="P"),
            *jira_issue_models("1", key="P-1", **issue_kwargs, resolved=dt(2016, 1, 4)),
            *jira_issue_models("2", key="P-2", **issue_kwargs, resolved=dt(2016, 1, 2)),
            *jira_issue_models("3", key="P-3", **issue_kwargs, resolved=dt(2016, 1, 3)),
            *jira_issue_models("4", key="P-4", **issue_kwargs, resolved=dt(2016, 1, 5)),
            *jira_issue_models("5", key="P-5", **issue_kwargs, resolved=None),
        ]

        body = self._body()
        metric = JIRAMetricID.JIRA_LIFE_TIME

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            body["order_by"] = [{"field": metric}]
            ids = await self._fetch_ids(json=body)
            assert ids == ("P-2", "P-3", "P-1", "P-4")

            body["order_by"] = [{"field": metric, "exclude_nulls": False}]
            ids = await self._fetch_ids(json=body)
            assert ids == ("P-2", "P-3", "P-1", "P-4", "P-5")

            body["order_by"] = [{"field": metric, "exclude_nulls": False, "nulls_first": True}]
            ids = await self._fetch_ids(json=body)
            assert ids == ("P-5", "P-2", "P-3", "P-1", "P-4")

            body["order_by"] = [{"field": metric, "direction": OrderByDirection.DESCENDING.value}]
            ids = await self._fetch_ids(json=body)
            assert ids == ("P-4", "P-1", "P-3", "P-2")


class TestSearchJIRAIssuesOrderByTrait(BaseSearchJIRAIssuesTest):
    async def test_created(self, mdb_rw: Database) -> None:
        issue_kwargs = {"project_id": "1"}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="P"),
            *jira_issue_models("1", key="P-1", **issue_kwargs, created=dt(2016, 1, 2)),
            *jira_issue_models("2", key="P-2", **issue_kwargs, created=dt(2016, 1, 4)),
            *jira_issue_models("3", key="P-3", **issue_kwargs, created=dt(2016, 1, 3)),
            *jira_issue_models("4", key="P-4", **issue_kwargs, created=dt(2016, 1, 1)),
        ]

        field = SearchJIRAIssuesOrderByIssueTrait.CREATED.value
        body = self._body()

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            body["order_by"] = [{"field": field}]
            ids = await self._fetch_ids(json=body)
            assert ids == ("P-4", "P-1", "P-3", "P-2")

            body["order_by"] = [{"field": field, "direction": OrderByDirection.DESCENDING.value}]
            ids = await self._fetch_ids(json=body)
            assert ids == ("P-2", "P-3", "P-1", "P-4")

    async def test_work_began(self, sdb: Database, mdb_rw: Database) -> None:
        issue_kwargs = {"project_id": "1", "created": dt(2016, 1, 2)}
        pr_kwargs = {"repository_full_name": "o/r"}
        mdb_models = [
            *pr_models(99, 4, 4, **pr_kwargs, created_at=dt(2016, 1, 2)),
            *pr_models(99, 5, 5, **pr_kwargs, created_at=dt(2016, 1, 8)),
            md_factory.JIRAProjectFactory(id="1", key="P"),
            *jira_issue_models("1", key="P-1", **issue_kwargs, work_began=dt(2016, 1, 4)),
            *jira_issue_models("2", key="P-2", **issue_kwargs, work_began=None),
            *jira_issue_models("3", key="P-3", **issue_kwargs, work_began=dt(2016, 1, 5)),
            *jira_issue_models("4", key="P-4", **issue_kwargs, work_began=dt(2016, 1, 7)),
            *jira_issue_models("5", key="P-5", **issue_kwargs, work_began=dt(2016, 1, 6)),
            *pr_jira_issue_mappings((4, "4"), (5, "5")),
        ]

        field = SearchJIRAIssuesOrderByIssueTrait.WORK_BEGAN.value
        body = self._body()

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=99, full_name="o/r")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            body["order_by"] = [{"field": field}]
            ids = await self._fetch_ids(json=body)
            # for P-4 PR is created before issue work_began, using that as work_began
            # for P-5 PR is created after, using issue work_began
            assert ids == ("P-4", "P-1", "P-3", "P-5")

            body["order_by"] = [{"field": field, "exclude_nulls": False}]
            ids = await self._fetch_ids(json=body)
            assert ids == ("P-4", "P-1", "P-3", "P-5", "P-2")

            body["order_by"] = [{"field": field, "exclude_nulls": False, "nulls_first": True}]
            ids = await self._fetch_ids(json=body)
            assert ids == ("P-2", "P-4", "P-1", "P-3", "P-5")

            body["order_by"] = [{"field": field, "direction": OrderByDirection.DESCENDING.value}]
            ids = await self._fetch_ids(json=body)
            assert ids == ("P-5", "P-3", "P-1", "P-4")

    async def test_updated(self, mdb_rw: Database) -> None:
        issue_kwargs = {"project_id": "1", "created": dt(2016, 1, 1)}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="P"),
            *jira_issue_models("1", key="P-1", **issue_kwargs, updated=dt(2016, 1, 2)),
            *jira_issue_models("2", key="P-2", **issue_kwargs, updated=dt(2016, 1, 4)),
            *jira_issue_models("3", key="P-3", **issue_kwargs, updated=dt(2016, 1, 3)),
        ]

        field = SearchJIRAIssuesOrderByIssueTrait.UPDATED.value
        body = self._body()

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            body["order_by"] = [{"field": field}]
            assert await self._fetch_ids(json=body) == ("P-1", "P-3", "P-2")

            body["order_by"] = [{"field": field, "direction": OrderByDirection.DESCENDING.value}]
            assert await self._fetch_ids(json=body) == ("P-2", "P-3", "P-1")

    async def test_multiple_order_by(self, mdb_rw: Database) -> None:
        i_kw = {"project_id": "1"}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="P"),
            *jira_issue_models(
                "1", key="P-1", **i_kw, created=dt(2016, 1, 2), updated=dt(2016, 1, 3),
            ),
            *jira_issue_models(
                "2", key="P-2", **i_kw, created=dt(2016, 1, 2), updated=dt(2016, 1, 4),
            ),
            *jira_issue_models(
                "3", key="P-3", **i_kw, created=dt(2016, 1, 3), updated=dt(2016, 1, 3),
            ),
        ]
        body = self._body()

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            UPDATED = SearchJIRAIssuesOrderByIssueTrait.UPDATED.value
            CREATED = SearchJIRAIssuesOrderByIssueTrait.CREATED.value

            body["order_by"] = [
                {"field": UPDATED},
                {"field": CREATED, "direction": OrderByDirection.DESCENDING.value},
            ]
            assert await self._fetch_ids(json=body) == ("P-3", "P-1", "P-2")

            body["order_by"] = [
                {"field": CREATED, "direction": OrderByDirection.DESCENDING.value},
                {"field": UPDATED},
            ]
            assert await self._fetch_ids(json=body) == ("P-3", "P-1", "P-2")
