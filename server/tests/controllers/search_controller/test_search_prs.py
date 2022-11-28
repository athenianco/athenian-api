from datetime import date
from operator import itemgetter
from typing import Any, Sequence

import pytest
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.models.state.models import AccountJiraInstallation
from athenian.api.models.web import (
    OrderByDirection,
    PullRequestMetricID,
    PullRequestStage,
    SearchPullRequestsOrderByPRTrait as PRTrait,
    SearchPullRequestsOrderByStageTiming as StageTiming,
    SearchPullRequestsRequest,
)
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_ACCOUNT_ID
from tests.testutils.factory.state import TeamFactory
from tests.testutils.factory.wizards import insert_repo, pr_models
from tests.testutils.requester import Requester
from tests.testutils.time import dt, freeze_time


class BaseSearchPRsTest(Requester):
    async def _request(self, assert_status: int = 200, **kwargs: Any) -> dict:
        path = "/private/search/pull_requests"
        client = self.client

        response = await client.request(method="POST", path=path, headers=self.headers, **kwargs)
        assert response.status == assert_status
        return await response.json()

    def _body(
        self,
        account: int = DEFAULT_ACCOUNT_ID,
        date_from: date = date(2019, 11, 1),
        date_to: date = date(2019, 12, 1),
        **kwargs: Any,
    ) -> dict:
        req = SearchPullRequestsRequest(
            account=account, date_from=date_from, date_to=date_to, **kwargs,
        )
        body = req.to_dict()
        body["date_from"] = body["date_from"].isoformat()
        body["date_to"] = body["date_to"].isoformat()
        return body

    async def _fetch_pr_numbers(self, **kwargs) -> Sequence[int]:
        res = await self._request(**kwargs)
        return tuple(pr["number"] for pr in res["pull_requests"])


class TestSearchPRsError(BaseSearchPRsTest):
    async def test_invalid_repositories(self, sdb: Database) -> None:
        body = self._body(repositories=["github.com/wrong/repo"])
        await self._request(403, json=body)

    async def test_invalid_team_in_participants(self, sdb: Database) -> None:
        body = self._body(participants={"author": ["{42}"]})
        res = await self._request(404, json=body)
        assert res["title"] == "Team not found"

    @pytest.mark.xfail
    async def test_empty_team_in_participants(self, sdb: Database) -> None:
        # TODO: this fails with unhandled exception
        await models_insert(sdb, TeamFactory(id=42, members=[]))
        body = self._body(participants={"author": ["{42}"]})
        await self._request(400, json=body)

    async def test_jira_not_installed(self, sdb: Database, mdb_rw: Database) -> None:
        await sdb.execute(
            sa.delete(AccountJiraInstallation).where(AccountJiraInstallation.account_id == 1),
        )
        body = self._body(jira={"projects": ["DEV"]})
        res = await self._request(422, json=body)
        assert res["detail"] == "JIRA has not been installed to the metadata yet."

    async def test_invalid_date(self, sdb: Database) -> None:
        body = self._body()
        body["date_from"] = "foo"
        await self._request(400, json=body)

        body.pop("date_from")
        await self._request(400, json=body)


class TestSearchPRs(BaseSearchPRsTest):
    async def test_smoke(self, sdb: Database) -> None:
        body = self._body()
        res = await self._request(json=body)
        assert len(res["pull_requests"]) == 7

    async def test_repositories(self, sdb: Database, mdb_rw: Database) -> None:
        times = {"created_at": dt(2019, 11, 15), "updated_at": dt(2019, 11, 20)}
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo0 = md_factory.RepositoryFactory(node_id=99, full_name="org0/repo0")
            await insert_repo(repo0, mdb_cleaner, mdb_rw, sdb)
            repo1 = md_factory.RepositoryFactory(node_id=98, full_name="org0/repo1")
            await insert_repo(repo1, mdb_cleaner, mdb_rw, sdb)
            models = [
                *pr_models(99, 10, 1010, repository_full_name="org0/repo0", **times),
                *pr_models(98, 11, 1011, repository_full_name="org0/repo1", **times),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            body = self._body()
            assert {1010, 1011} & set(await self._fetch_pr_numbers(json=body)) == {1010, 1011}

            body = self._body(repositories=["github.com/org0/repo0", "github.com/org0/repo1"])
            assert {1010, 1011} & set(await self._fetch_pr_numbers(json=body)) == {1010, 1011}

            body = self._body(repositories=["github.com/org0/repo0"])
            assert {1010, 1011} & set(await self._fetch_pr_numbers(json=body)) == {1010}

            body = self._body(repositories=["github.com/org0/repo1"])
            assert {1010, 1011} & set(await self._fetch_pr_numbers(json=body)) == {1011}

    async def test_participants(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10, members=[40390]))
        body = self._body(participants={"author": ["{10}"]})
        res = await self._request(json=body)
        assert res["pull_requests"] == [
            {"number": 1247, "repository": "github.com/src-d/go-git"},
        ]

        await models_insert(sdb, TeamFactory(id=11, parent_id=10, members=[39874]))
        # now also PR authored subteam are selected
        body = self._body(participants={"author": ["{10}"]})
        res = await self._request(json=body)
        assert sorted(res["pull_requests"], key=itemgetter("number")) == [
            {"number": 1247, "repository": "github.com/src-d/go-git"},
            {"number": 1248, "repository": "github.com/src-d/go-git"},
        ]

    async def test_jira(self, sdb: Database, mdb_rw: Database) -> None:
        times = {"created_at": dt(2019, 11, 15), "updated_at": dt(2019, 11, 20)}
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=99, full_name="org0/repo0")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
            models = [
                *pr_models(99, 10, 110, repository_full_name="org0/repo0", **times),
                *pr_models(99, 11, 111, repository_full_name="org0/repo0", **times),
                md_factory.JIRAProjectFactory(id="200", key="DD"),
                md_factory.JIRAProjectFactory(id="201", key="EE"),
                md_factory.JIRAIssueFactory(id="20", priority_name="extreme", project_id="200"),
                md_factory.JIRAIssueFactory(id="21", priority_name="medium", project_id="200"),
                md_factory.JIRAIssueFactory(id="22", project_id="201"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=10, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=11, jira_id="21"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=11, jira_id="22"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            body = self._body()
            assert {110, 111} & set(await self._fetch_pr_numbers(json=body)) == {110, 111}

            body = self._body(jira={"priorities": ["extreme"]})
            assert {110, 111} & set(await self._fetch_pr_numbers(json=body)) == {110}

            body = self._body(jira={"priorities": ["medium"]})
            assert {110, 111} & set(await self._fetch_pr_numbers(json=body)) == {111}

            body = self._body(jira={"projects": ["EE"]})
            assert {111} & set(await self._fetch_pr_numbers(json=body)) == {111}

    async def test_multiple_filters(self, sdb: Database, mdb_rw: Database) -> None:
        ts = {"created_at": dt(2022, 4, 2), "updated_at": dt(2022, 4, 15)}

        body = self._body(
            date_from=date(2022, 4, 1),
            date_to=date(2022, 4, 30),
        )

        t0 = dt(2022, 4, 4)
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo0 = md_factory.RepositoryFactory(node_id=20, full_name="org0/repo0")
            repo1 = md_factory.RepositoryFactory(node_id=21, full_name="org0/repo1")
            await insert_repo(repo0, mdb_cleaner, mdb_rw, sdb)
            await insert_repo(repo1, mdb_cleaner, mdb_rw, sdb)
            models = [
                *pr_models(20, 11, 1, repository_full_name="org0/repo0", **ts),
                *pr_models(21, 12, 2, repository_full_name="org0/repo1", **ts),
                *pr_models(
                    20, 13, 3, repository_full_name="org0/repo0", closed=True, closed_at=t0, **ts,
                ),
                *pr_models(
                    21, 14, 4, repository_full_name="org0/repo1", closed=True, closed_at=t0, **ts,
                ),
                *pr_models(
                    21, 15, 5, repository_full_name="org0/repo1", closed=True, closed_at=t0, **ts,
                ),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            ns = await self._fetch_pr_numbers(json=body)
            assert sorted(ns) == [1, 2, 3, 4, 5]

            body["repositories"] = ["github.com/org0/repo0"]
            ns = await self._fetch_pr_numbers(json=body)
            assert sorted(ns) == [1, 3]

            body["repositories"] = ["github.com/org0/repo1"]
            body["stages"] = [PullRequestStage.DONE]
            ns = await self._fetch_pr_numbers(json=body)
            assert sorted(ns) == [4, 5]

            body["stages"] = [PullRequestStage.WIP]
            ns = await self._fetch_pr_numbers(json=body)
            assert ns == (2,)


class TestSearchPRsOrderByMetric(BaseSearchPRsTest):
    _from: dict = {"date_from": date(2019, 10, 1), "date_to": date(2019, 12, 1)}

    async def test_int_metric(self, sdb: Database) -> None:
        metric = PullRequestMetricID.PR_SIZE
        body = self._body(order_by=[{"field": metric, "exclude_nulls": False}], **self._from)
        ns = await self._fetch_pr_numbers(json=body)
        assert sorted(ns[:3]) == [1226, 1235, 1238]  # all of size 2
        assert ns[3:] == (1246, 1247, 1248, 1231, 1225, 1243, 1153)

        # direction
        body["order_by"][0]["direction"] = OrderByDirection.DESCENDING.value
        ns = await self._fetch_pr_numbers(json=body)
        assert ns[:6] == (1243, 1225, 1231, 1248, 1247, 1246)
        assert sorted(ns[6:9]) == [1226, 1235, 1238]  # all of size 2
        assert ns[9] == 1153  # null size since out of time, last

        # nulls_first
        body = self._body(
            order_by=[{"field": metric, "nulls_first": True, "exclude_nulls": False}],
            **self._from,
        )
        ns = await self._fetch_pr_numbers(json=body)
        assert ns[0] == 1153  # null
        assert sorted(ns[1:4]) == [1226, 1235, 1238]  # all of size 2
        assert ns[4:] == (1246, 1247, 1248, 1231, 1225, 1243)

        body["order_by"][0]["direction"] = OrderByDirection.DESCENDING.value
        ns = await self._fetch_pr_numbers(json=body)
        assert ns[0] == 1153  # null
        assert ns[1:7] == (1243, 1225, 1231, 1248, 1247, 1246)
        assert sorted(ns[7:]) == [1226, 1235, 1238]

    async def test_timedelta_metric(self, sdb: Database) -> None:
        metric = PullRequestMetricID.PR_REVIEW_TIME
        body = self._body(order_by=[{"field": metric, "exclude_nulls": False}], **self._from)
        ns = await self._fetch_pr_numbers(json=body)
        assert ns[:2] == (1226, 1225)
        assert sorted(ns[2:]) == [1153, 1231, 1235, 1238, 1243, 1246, 1247, 1248]  # all nulls

        body["order_by"][0]["direction"] = OrderByDirection.DESCENDING.value
        ns = await self._fetch_pr_numbers(json=body)
        assert ns[:2] == (1225, 1226)
        assert sorted(ns[2:]) == [1153, 1231, 1235, 1238, 1243, 1246, 1247, 1248]

        body["order_by"][0]["nulls_first"] = True
        ns = await self._fetch_pr_numbers(json=body)
        assert sorted(ns[:8]) == [1153, 1231, 1235, 1238, 1243, 1246, 1247, 1248]
        assert ns[8:] == (1225, 1226)

        body = self._body(order_by=[{"field": metric, "exclude_nulls": True}], **self._from)
        ns = await self._fetch_pr_numbers(json=body)
        assert ns == (1226, 1225)

    async def test_float_metric(self, sdb: Database) -> None:
        metric = PullRequestMetricID.PR_PARTICIPANTS_PER
        body = self._body(order_by=[{"field": metric}], **self._from)
        ns = await self._fetch_pr_numbers(json=body)
        assert sorted(ns[:6]) == [1231, 1235, 1243, 1246, 1247, 1248]  # all 2 participants
        assert sorted(ns[6:8]) == [1226, 1238]  # 3 participants
        assert ns[8:] == (1225, 1153)  # 4 and 6 participants

    async def test_multiple_metrics(self, sdb: Database, mdb_rw: Database) -> None:
        times = {"created_at": dt(2022, 4, 2), "updated_at": dt(2022, 4, 15)}

        body = self._body(
            date_from=date(2022, 4, 1),
            date_to=date(2022, 4, 30),
            order_by=[
                {
                    "field": PullRequestMetricID.PR_SIZE,
                    "direction": OrderByDirection.DESCENDING.value,
                },
                {"field": PullRequestMetricID.PR_OPEN_TIME, "exclude_nulls": False},
            ],
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=99, full_name="org0/repo0")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
            pr_kwargs = {**times, "repository_full_name": "org0/repo0", "deletions": 0}
            models = [
                *pr_models(99, 10, 110, additions=10, **pr_kwargs),
                *pr_models(
                    99, 11, 111, closed=True, closed_at=dt(2022, 4, 5), additions=12, **pr_kwargs,
                ),
                *pr_models(
                    99, 12, 112, closed=True, closed_at=dt(2022, 4, 8), additions=12, **pr_kwargs,
                ),
                *pr_models(
                    99, 13, 113, closed=True, closed_at=dt(2022, 4, 7), additions=12, **pr_kwargs,
                ),
                *pr_models(99, 14, 114, **pr_kwargs, additions=13),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            ns = await self._fetch_pr_numbers(json=body)
            # 111 112 and 113 have the same size so they are ordered by ascending PR_OPEN_TIME
            assert ns == (114, 111, 113, 112, 110)

    async def test_lead_time(self, sdb: Database) -> None:
        metric = PullRequestMetricID.PR_LEAD_TIME
        body = self._body(order_by=[{"field": metric, "exclude_nulls": True}], **self._from)
        ns = await self._fetch_pr_numbers(json=body)
        assert ns == (1235, 1226, 1225, 1231)

        body["order_by"][0]["direction"] = OrderByDirection.DESCENDING.value
        ns = await self._fetch_pr_numbers(json=body)
        assert ns == (1231, 1225, 1226, 1235)


class TestSearchPRsOrderByStageTiming(BaseSearchPRsTest):
    async def test_review_stage_timing_smoke(self, sdb: Database) -> None:
        body = self._body(order_by=[{"field": StageTiming.PR_WIP_STAGE_TIMING.value}])
        ns = await self._fetch_pr_numbers(json=body)
        assert ns == (1238, 1247, 1231, 1235, 1248, 1246, 1243)

        body = self._body(order_by=[{"field": StageTiming.PR_MERGE_STAGE_TIMING.value}])
        ns = await self._fetch_pr_numbers(json=body)
        assert ns == (1235, 1231, 1238)

    async def test_wip_stage_timing(self, sdb: Database, mdb_rw: Database) -> None:
        body = self._body(
            date_from=date(2022, 4, 1),
            date_to=date(2022, 4, 30),
            repositories=["github.com/org0/repo0"],
            order_by=[{"field": StageTiming.PR_WIP_STAGE_TIMING.value}],
        )
        pr_kwargs = {"repository_full_name": "org0/repo0"}
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=99, full_name="org0/repo0")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)

            models = [
                *pr_models(99, 11, 1, created_at=dt(2022, 4, 3), **pr_kwargs),
                *pr_models(99, 12, 2, created_at=dt(2022, 4, 5), **pr_kwargs),
                *pr_models(99, 13, 3, created_at=dt(2022, 4, 4), **pr_kwargs),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            ns = await self._fetch_pr_numbers(json=body)
            assert ns == (2, 3, 1)

            body["order_by"][0]["direction"] = OrderByDirection.DESCENDING.value
            ns = await self._fetch_pr_numbers(json=body)
            assert ns == (1, 3, 2)

    @freeze_time("2022-04-25")
    async def test_multiple_stage_timings(self, sdb: Database, mdb_rw: Database) -> None:
        body = self._body(
            date_from=date(2022, 4, 1),
            date_to=date(2022, 4, 30),
            repositories=["github.com/org0/repo0"],
        )
        WIP = StageTiming.PR_WIP_STAGE_TIMING.value
        REVIEW = StageTiming.PR_REVIEW_STAGE_TIMING.value
        DESC = OrderByDirection.DESCENDING.value
        repo = md_factory.RepositoryFactory(node_id=99, full_name="org0/repo0")
        pr_kwargs = {"repository_full_name": "org0/repo0"}
        # wip days: 1 19 1 2
        # review days: 20 nat 19 18
        models = [
            *pr_models(
                99, 11, 1, created_at=dt(2022, 4, 3), review_request=dt(2022, 4, 4), **pr_kwargs,
            ),
            *pr_models(99, 12, 2, created_at=dt(2022, 4, 5), **pr_kwargs),
            *pr_models(
                99, 13, 3, created_at=dt(2022, 4, 4), review_request=dt(2022, 4, 5), **pr_kwargs,
            ),
            *pr_models(
                99, 14, 4, created_at=dt(2022, 4, 4), review_request=dt(2022, 4, 6), **pr_kwargs,
            ),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)

            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            body["order_by"] = [{"field": WIP}, {"field": REVIEW, "exclude_nulls": False}]
            assert await self._fetch_pr_numbers(json=body) == (3, 1, 4, 2)

            body["order_by"] = [{"field": WIP}, {"field": REVIEW}]
            # 2 filtered out, it has nat review timing
            assert await self._fetch_pr_numbers(json=body) == (3, 1, 4)

            body["order_by"] = [
                {"field": WIP, "direction": DESC},
                {"field": REVIEW, "exclude_nulls": False},
            ]
            assert await self._fetch_pr_numbers(json=body) == (2, 4, 3, 1)

            body["order_by"] = [
                {"field": WIP, "direction": DESC},
                {"field": REVIEW, "direction": DESC, "exclude_nulls": False},
            ]
            assert await self._fetch_pr_numbers(json=body) == (2, 4, 1, 3)

            body["order_by"] = [
                {"field": REVIEW, "direction": DESC, "exclude_nulls": False},
                {"field": WIP, "direction": DESC},
            ]
            assert await self._fetch_pr_numbers(json=body) == (1, 3, 4, 2)

            body["order_by"] = [
                {
                    "field": REVIEW,
                    "direction": DESC,
                    "exclude_nulls": False,
                    "nulls_first": True,
                },
                {"field": WIP},
            ]
            assert await self._fetch_pr_numbers(json=body) == (2, 1, 3, 4)

            body["order_by"] = [
                {"field": REVIEW, "exclude_nulls": False, "nulls_first": True},
                {"field": WIP},
            ]
            assert await self._fetch_pr_numbers(json=body) == (2, 4, 3, 1)


class TestSearchPRsOrderByTrait(BaseSearchPRsTest):
    async def test_work_began(self, sdb: Database, mdb_rw: Database) -> None:
        body = self._body(
            date_from=date(2022, 4, 1),
            date_to=date(2022, 4, 30),
            order_by=[{"field": PRTrait.WORK_BEGAN.value}],
        )
        pr_kwargs = {"repository_full_name": "org0/repo0"}
        repo = md_factory.RepositoryFactory(node_id=99, full_name="org0/repo0")
        models = [
            # work_began is min(created_at, first_commit)
            *pr_models(99, 11, 1, created_at=dt(2022, 4, 20), **pr_kwargs),
            *pr_models(
                99, 12, 2, created_at=dt(2022, 4, 20), commits=[dt(2022, 4, 10)], **pr_kwargs,
            ),
            *pr_models(
                99, 13, 3, created_at=dt(2022, 4, 19), commits=[dt(2022, 4, 23)], **pr_kwargs,
            ),
            *pr_models(99, 14, 4, created_at=dt(2022, 4, 12), **pr_kwargs),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            assert await self._fetch_pr_numbers(json=body) == (2, 4, 3, 1)

            body["order_by"][0]["direction"] = OrderByDirection.DESCENDING.value
            assert await self._fetch_pr_numbers(json=body) == (1, 3, 4, 2)

    @freeze_time("2022-04-28")
    async def test_first_review_request(self, sdb: Database, mdb_rw: Database) -> None:
        body = self._body(
            date_from=date(2022, 4, 1),
            date_to=date(2022, 4, 30),
            order_by=[
                {"field": PRTrait.FIRST_REVIEW_REQUEST.value, "exclude_nulls": False},
            ],
        )
        pr_kwargs = {"repository_full_name": "org0/repo0", "created_at": dt(2022, 4, 10)}
        repo = md_factory.RepositoryFactory(node_id=99, full_name="org0/repo0")
        models = [
            *pr_models(99, 11, 1, review_request=dt(2022, 4, 20), **pr_kwargs),
            *pr_models(99, 12, 2, review_request=dt(2022, 4, 23), **pr_kwargs),
            *pr_models(99, 13, 3, **pr_kwargs),
            *pr_models(99, 14, 4, review_request=dt(2022, 4, 22), **pr_kwargs),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            assert await self._fetch_pr_numbers(json=body) == (1, 4, 2, 3)

            body["order_by"][0]["nulls_first"] = True
            assert await self._fetch_pr_numbers(json=body) == (3, 1, 4, 2)

            body["order_by"][0]["nulls_first"] = False
            body["order_by"][0]["direction"] = OrderByDirection.DESCENDING.value
            assert await self._fetch_pr_numbers(json=body) == (2, 4, 1, 3)

            body["order_by"][0]["exclude_nulls"] = True
            assert await self._fetch_pr_numbers(json=body) == (2, 4, 1)


class TestSearchPRsStagesFilter(BaseSearchPRsTest):
    async def test_smoke(self, sdb: Database) -> None:
        body = self._body()
        ns = await self._fetch_pr_numbers(json=body)
        assert sorted(ns) == [1231, 1235, 1238, 1243, 1246, 1247, 1248]

        body = self._body(stages=[PullRequestStage.MERGING])
        ns = await self._fetch_pr_numbers(json=body)
        assert sorted(ns) == [1238]

        body = self._body(stages=[PullRequestStage.WIP])
        ns = await self._fetch_pr_numbers(json=body)
        assert sorted(ns) == [1243, 1246]

        body = self._body(stages=[PullRequestStage.DONE])
        ns = await self._fetch_pr_numbers(json=body)
        assert sorted(ns) == [1231, 1235, 1247, 1248]

        body = self._body(stages=[PullRequestStage.MERGING, PullRequestStage.WIP])
        ns = await self._fetch_pr_numbers(json=body)
        assert sorted(ns) == [1238, 1243, 1246]

        body = self._body(stages=[PullRequestStage.REVIEWING])
        ns = await self._fetch_pr_numbers(json=body)
        assert ns == ()

    async def test_created_data(self, sdb: Database, mdb_rw: Database) -> None:
        times = {"created_at": dt(2022, 4, 2), "updated_at": dt(2022, 4, 15)}

        body = self._body(
            date_from=date(2022, 4, 1),
            date_to=date(2022, 4, 30),
            repositories=["github.com/org0/repo0"],
        )

        t0 = dt(2022, 4, 4)
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=99, full_name="org0/repo0")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
            pr_kwargs = {**times, "repository_full_name": "org0/repo0"}
            models = [
                *pr_models(99, 11, 1, **pr_kwargs),
                *pr_models(99, 12, 2, **pr_kwargs),
                *pr_models(
                    99, 13, 3, closed=True, closed_at=t0, merged=True, merged_at=t0, **pr_kwargs,
                ),
                *pr_models(99, 14, 4, closed=True, closed_at=t0, **pr_kwargs),
                *pr_models(99, 15, 5, closed=True, closed_at=t0, **pr_kwargs),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            ns = await self._fetch_pr_numbers(json=body)
            assert sorted(ns) == [1, 2, 3, 4, 5]

            body["stages"] = [PullRequestStage.WIP]
            assert sorted(await self._fetch_pr_numbers(json=body)) == [1, 2]

            body["stages"] = [PullRequestStage.DONE]
            assert sorted(await self._fetch_pr_numbers(json=body)) == [4, 5]

            body["stages"] = [PullRequestStage.RELEASING]
            assert await self._fetch_pr_numbers(json=body) == (3,)

            body["stages"] = [PullRequestStage.WIP, PullRequestStage.RELEASE_IGNORED]
            assert sorted(await self._fetch_pr_numbers(json=body)) == [1, 2]

            body["stages"] = [PullRequestStage.DONE, PullRequestStage.RELEASING]
            assert sorted(await self._fetch_pr_numbers(json=body)) == [3, 4, 5]
