from functools import partial
from typing import Any

import pytest

from athenian.api.db import Database
from athenian.api.models.web import CalculatedReleaseMetric, ReleaseMetricID
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_ACCOUNT_ID
from tests.testutils.factory.wizards import insert_repo, pr_jira_issue_mappings, pr_models
from tests.testutils.requester import Requester
from tests.testutils.time import dt


class BaseCalcMetricsReleasesTest(Requester):
    async def _request(self, *, assert_status=200, **kwargs) -> dict | list:
        response = await self.client.request(
            method="POST", path="/v1/metrics/releases", headers=self.headers, **kwargs,
        )
        assert response.status == assert_status
        return await response.json()

    @classmethod
    def _body(self, **kwargs: Any) -> dict:
        kwargs.setdefault("account", DEFAULT_ACCOUNT_ID)
        kwargs.setdefault("granularities", ["all"])
        if "for" not in kwargs and "for_" in kwargs:
            kwargs["for"] = kwargs.pop("for_")
        if "with" not in kwargs and "with_" in kwargs:
            kwargs["with"] = kwargs.pop("with_")
        return kwargs


class TestCalcMetricsReleasesErrors(BaseCalcMetricsReleasesTest):
    async def test_empty_granularities(self):
        body = self._body(
            date_from="2018-01-12",
            date_to="2020-03-01",
            for_=[["{1}"]],
            jira={"epics": []},
            metrics=list(ReleaseMetricID),
            granularities=[],
        )
        res = await self._request(json=body, assert_status=400)
        assert "granularities" in res["detail"]


class TestCalcMetricsReleases(BaseCalcMetricsReleasesTest):
    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_smoke(self, no_jira):
        body = self._body(
            date_from="2018-01-12",
            date_to="2020-03-01",
            for_=[["{1}"]],
            jira={"epics": []},
            metrics=list(ReleaseMetricID),
            granularities=["all", "3 month"],
        )
        rbody = await self._request(json=body)
        models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
        assert len(models) == 2
        for model in models:
            assert model.for_ == ["{1}"]
            assert model.metrics == body["metrics"]
            assert model.matches == {
                "github.com/src-d/go-git": "tag",
                "github.com/src-d/gitbase": "branch",
            }
            assert model.granularity in body["granularities"]
            for mv in model.values:
                exist = mv.values[model.metrics.index(ReleaseMetricID.TAG_RELEASE_AGE)] is not None
                for metric, value in zip(model.metrics, mv.values):
                    if "branch" in metric:
                        if "avg" not in metric and metric != ReleaseMetricID.BRANCH_RELEASE_AGE:
                            assert value == 0, metric
                        else:
                            assert value is None, metric
                    elif exist:
                        assert value is not None, metric
            if model.granularity == "all":
                assert len(model.values) == 1
                assert any(v is not None for v in model.values[0].values)
            else:
                assert any(v is not None for values in model.values for v in values.values)
                assert len(model.values) == 9

    @pytest.mark.parametrize(
        "role, n",
        [("releaser", 20), ("pr_author", 10), ("commit_author", 21)],
    )
    async def test_participants_single(self, role, n):
        body = self._body(
            date_from="2018-01-12",
            date_to="2020-03-01",
            for_=[["github.com/src-d/go-git"]],
            with_=[{role: ["github.com/mcuadros"]}],
            metrics=[ReleaseMetricID.RELEASE_COUNT],
        )
        rbody = await self._request(json=body)
        models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
        assert len(models) == 1
        assert models[0].values[0].values[0] == n

    async def test_participants_multiple(self):
        body = self._body(
            date_from="2018-01-12",
            date_to="2020-03-01",
            for_=[["github.com/src-d/go-git"]],
            with_=[
                {
                    "releaser": ["github.com/smola"],
                    "pr_author": ["github.com/mcuadros"],
                    "commit_author": ["github.com/smola"],
                },
            ],
            metrics=[ReleaseMetricID.RELEASE_COUNT],
        )
        rbody = await self._request(json=body)
        models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
        assert len(models) == 1
        assert models[0].values[0].values[0] == 12

    async def test_participants_team(self, sample_team):
        body = self._body(
            date_from="2018-01-12",
            date_to="2020-03-01",
            for_=[["github.com/src-d/go-git"]],
            with_=[
                {
                    "releaser": ["{%d}" % sample_team],
                    "pr_author": ["{%d}" % sample_team],
                    "commit_author": ["{%d}" % sample_team],
                },
            ],
            metrics=[ReleaseMetricID.RELEASE_COUNT],
        )
        rbody = await self._request(json=body)
        models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
        assert len(models) == 1
        assert models[0].values[0].values[0] == 21

    async def test_participants_groups(self):
        body = self._body(
            date_from="2018-01-12",
            date_to="2020-03-01",
            for_=[["github.com/src-d/go-git"]],
            with_=[{"releaser": ["github.com/mcuadros"]}, {"pr_author": ["github.com/smola"]}],
            metrics=[ReleaseMetricID.RELEASE_COUNT],
        )
        rbody = await self._request(json=body)
        models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
        assert len(models) == 2
        assert models[0].values[0].values[0] == 20
        assert models[1].values[0].values[0] == 4

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    @pytest.mark.parametrize(
        "account, date_to, quantiles, extra_metrics, in_, code",
        [
            (3, "2020-02-22", [0, 1], [], "{1}", 404),
            (2, "2020-02-22", [0, 1], [], "github.com/src-d/go-git", 422),
            (10, "2020-02-22", [0, 1], [], "{1}", 404),
            (1, "2015-10-13", [0, 1], [], "{1}", 200),
            (1, "2015-10-13", [0, 1], ["whatever"], "{1}", 400),
            (1, "2010-01-11", [0, 1], [], "{1}", 400),
            (1, "2020-01-32", [0, 1], [], "{1}", 400),
            (1, "2020-01-01", [-1, 0.5], [], "{1}", 400),
            (1, "2020-01-01", [0, -1], [], "{1}", 400),
            (1, "2020-01-01", [10, 20], [], "{1}", 400),
            (1, "2020-01-01", [0.5, 0.25], [], "{1}", 400),
            (1, "2020-01-01", [0.5, 0.5], [], "{1}", 400),
            (1, "2015-10-13", [0, 1], [], "github.com/athenianco/athenian-api", 403),
        ],
    )
    async def test_nasty_input(self, account, date_to, quantiles, extra_metrics, in_, code):
        body = {
            "for": [[in_], [in_]],
            "metrics": [ReleaseMetricID.TAG_RELEASE_AGE] + extra_metrics,
            "date_from": "2015-10-13",
            "date_to": date_to,
            "granularities": ["4 month"],
            "quantiles": quantiles,
            "account": account,
        }
        await self._request(assert_status=code, json=body)

    @pytest.mark.parametrize("devid", ["whatever", ""])
    async def test_participants_invalid(self, devid):
        body = self._body(
            date_from="2018-01-12",
            date_to="2020-03-01",
            for_=[["github.com/src-d/go-git"]],
            with_=[{"releaser": [devid]}],
            metrics=[ReleaseMetricID.RELEASE_COUNT],
        )
        await self._request(assert_status=400, json=body)

    @pytest.mark.parametrize("q, value", ((0.95, "2687847s"), (1, "2687847s")))
    async def test_quantiles(self, q, value):
        body = self._body(
            date_from="2015-01-12",
            date_to="2020-03-01",
            for_=[["{1}"], ["github.com/src-d/go-git"]],
            metrics=[ReleaseMetricID.TAG_RELEASE_AGE],
            quantiles=[0, q],
        )
        rbody = await self._request(json=body)
        models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
        assert len(models) == 2
        assert models[0].values == models[1].values
        model = models[0]
        assert model.values[0].values == [value]

    async def test_jira(self):
        body = self._body(
            date_from="2018-01-01",
            date_to="2020-03-01",
            for_=[["{1}"]],
            metrics=[ReleaseMetricID.RELEASE_COUNT, ReleaseMetricID.RELEASE_PRS],
            jira={"labels_include": ["bug", "onboarding", "performance"]},
        )
        rbody = await self._request(json=body)
        models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
        assert len(models) == 1
        assert models[0].values[0].values == [8, 43]
        del body["jira"]

        rbody = await self._request(json=body)
        models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
        assert len(models) == 1
        assert models[0].values[0].values == [22, 235]

    async def test_empty_jira_object(self):
        body = self._body(
            date_from="2018-01-01",
            date_to="2020-03-01",
            for_=[["{1}"]],
            metrics=[ReleaseMetricID.RELEASE_COUNT],
        )
        rbody = await self._request(json=body)
        count = rbody[0]["values"][0]["values"][0]

        body["jira"] = {}
        rbody = await self._request(json=body)
        count_empty_jira = rbody[0]["values"][0]["values"][0]
        assert count == count_empty_jira

        body["jira"] = {"epics": [], "labels_include": []}
        rbody = await self._request(json=body)
        count_empty_jira_2 = rbody[0]["values"][0]["values"][0]
        assert count == count_empty_jira_2

    async def test_labels(self):
        body = self._body(
            date_from="2018-01-01",
            date_to="2020-03-01",
            for_=[["{1}"]],
            metrics=[ReleaseMetricID.RELEASE_COUNT, ReleaseMetricID.RELEASE_PRS],
            labels_include=["bug", "plumbing", "Enhancement"],
        )
        rbody = await self._request(json=body)
        models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
        assert len(models) == 1
        assert models[0].values[0].values == [3, 36]
        del body["labels_include"]

        rbody = await self._request(json=body)
        models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
        assert len(models) == 1
        assert models[0].values[0].values == [22, 235]

    @pytest.mark.parametrize(
        "second_repo, counts",
        [
            ("/beta", [44, 119]),
            ("", [44, 192]),
        ],
    )
    async def test_logical(
        self,
        logical_settings_db,
        release_match_setting_tag_logical_db,
        second_repo,
        counts,
    ):
        body = self._body(
            date_from="2018-01-01",
            date_to="2020-03-01",
            for_=[["github.com/src-d/go-git/alpha", "github.com/src-d/go-git" + second_repo]],
            metrics=[ReleaseMetricID.RELEASE_COUNT, ReleaseMetricID.RELEASE_PRS],
        )
        rbody = await self._request(json=body)
        models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
        assert len(models) == 1
        assert models[0].values[0].values == counts

    async def test_participants_many_participants(self):
        body = self._body(
            date_from="2018-01-12",
            date_to="2020-03-01",
            for_=[["github.com/src-d/go-git"]],
            with_=[
                {
                    "releaser": ["github.com/smola"],
                    "pr_author": ["github.com/mcuadros"],
                    "commit_author": ["github.com/smola"],
                },
                {"releaser": ["github.com/mcuadros"]},
            ],
            metrics=[ReleaseMetricID.RELEASE_COUNT],
        )
        rbody = await self._request(json=body)
        models = [CalculatedReleaseMetric.from_dict(i) for i in rbody]
        assert len(models) == 2
        assert models[0].values[0].values[0] == 12
        assert models[1].values[0].values[0] == 20

    async def test_no_repositories(self) -> None:
        body = self._body(
            date_from="2018-01-01",
            date_to="2020-03-01",
            metrics=[ReleaseMetricID.RELEASE_COUNT],
            for_=[None, ["{1}"]],
        )
        rbody = await self._request(assert_status=200, json=body)

        calculated = sorted(rbody, key=lambda calc: "for" in calc)

        assert "for" not in calculated[0]
        assert calculated[0]["values"] == [{"date": "2018-01-01", "values": [22]}]

        assert calculated[1]["for"] == ["{1}"]
        # all repositories is the same as ALL reposet repositories, metric values are the same
        assert calculated[1]["values"] == calculated[0]["values"]

    async def test_jiragroups(self, sdb: Database, mdb_rw: Database) -> None:
        body = self._body(
            date_from="2018-01-01",
            date_to="2018-02-01",
            for_=[["github.com/org/repo"]],
            metrics=[
                ReleaseMetricID.RELEASE_COUNT,
                ReleaseMetricID.RELEASE_PRS,
            ],
            jiragroups=[
                {"issue_types": ["bug"]},
                {"issue_types": ["task"]},
                {},
            ],
        )

        mk_release = partial(
            md_factory.ReleaseFactory,
            repository_full_name="org/repo",
            repository_node_id=99,
            published_at=dt(2018, 1, 5),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=99, full_name="org/repo")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)

            models = [
                mk_release(sha="A" * 40, name="r0"),
                mk_release(sha="B" * 40, name="r1"),
                mk_release(sha="C" * 40, name="r2"),
                mk_release(sha="D" * 40, name="r3"),
                md_factory.NodeCommitFactory(node_id=101, repository_id=99, sha="A" * 40),
                md_factory.NodeCommitFactory(node_id=102, repository_id=99, sha="B" * 40),
                md_factory.NodeCommitFactory(node_id=103, repository_id=99, sha="C" * 40),
                md_factory.NodeCommitFactory(node_id=104, repository_id=99, sha="D" * 40),
                *pr_models(99, 1, 1, merge_commit_id=101),
                *pr_models(99, 2, 2, merge_commit_id=102),
                *pr_models(99, 3, 3, merge_commit_id=103),
                *pr_models(99, 4, 4, merge_commit_id=104),
                md_factory.JIRAProjectFactory(id="1", key="DD"),
                md_factory.JIRAIssueTypeFactory(id="t", name="task"),
                md_factory.JIRAIssueTypeFactory(id="b", name="bug"),
                md_factory.JIRAIssueFactory(id="20", project_id="1", type_id="t", type="task"),
                md_factory.JIRAIssueFactory(id="30", project_id="1", type_id="b", type="bug"),
                *pr_jira_issue_mappings((1, "20"), (2, "30"), (3, "20")),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            rbody = await self._request(json=body)
            assert len(rbody) == 3

            res_all = next(r for r in rbody if not r.get("jira"))
            assert res_all["values"][0]["values"] == [4, 4]
            res_bug = next(r for r in rbody if r.get("jira") == {"issue_types": ["bug"]})
            assert res_bug["values"][0]["values"] == [1, 1]
            res_task = next(r for r in rbody if r.get("jira") == {"issue_types": ["task"]})
            assert res_task["values"][0]["values"] == [2, 2]

            body.pop("jiragroups")
            rbody = await self._request(json=body)
            assert len(rbody) == 1
            assert rbody[0]["values"][0]["values"] == [4, 4]

    async def test_jiragroups_labels(self, sdb: Database, mdb_rw: Database) -> None:
        body = self._body(
            date_from="2018-01-01",
            date_to="2018-02-01",
            for_=[["github.com/o/r"]],
            metrics=[ReleaseMetricID.RELEASE_COUNT, ReleaseMetricID.RELEASE_PRS],
            jiragroups=[{"labels_include": ["l0"]}, {"labels_include": ["l1", "l0"]}, {}],
        )

        mk_release = partial(
            md_factory.ReleaseFactory,
            repository_full_name="o/r",
            repository_node_id=99,
            published_at=dt(2018, 1, 5),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=99, full_name="o/r")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)

            models = [
                mk_release(sha="A" * 40, name="r0"),
                mk_release(sha="B" * 40, name="r1"),
                mk_release(sha="C" * 40, name="r2"),
                mk_release(sha="D" * 40, name="r3"),
                md_factory.NodeCommitFactory(node_id=101, repository_id=99, sha="A" * 40),
                md_factory.NodeCommitFactory(node_id=102, repository_id=99, sha="B" * 40),
                md_factory.NodeCommitFactory(node_id=103, repository_id=99, sha="C" * 40),
                md_factory.NodeCommitFactory(node_id=104, repository_id=99, sha="D" * 40),
                *pr_models(99, 1, 1, merge_commit_id=101),
                *pr_models(99, 2, 2, merge_commit_id=102),
                *pr_models(99, 3, 3, merge_commit_id=103),
                *pr_models(99, 4, 4, merge_commit_id=104),
                md_factory.JIRAProjectFactory(id="1", key="DD"),
                md_factory.JIRAIssueFactory(id="1", project_id="1", labels=["l0"]),
                md_factory.JIRAIssueFactory(id="2", project_id="1", labels=["l1"]),
                md_factory.JIRAIssueFactory(id="3", project_id="1", labels=["l0", "l1"]),
                md_factory.JIRAIssueFactory(id="4", project_id="1", labels=["l2"]),
                md_factory.JIRAIssueFactory(id="5", project_id="1", labels=[]),
                *pr_jira_issue_mappings((1, "1"), (2, "2"), (3, "3")),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            rbody = await self._request(json=body)
            assert len(rbody) == 3

            res_l0 = next(r for r in rbody if r["jira"] == {"labels_include": ["l0"]})
            assert res_l0["values"][0]["values"] == [2, 2]

            res_l1_l0 = next(r for r in rbody if r["jira"] == {"labels_include": ["l1", "l0"]})
            assert res_l1_l0["values"][0]["values"] == [3, 3]

            res_all = next(r for r in rbody if r["jira"] == {})
            assert res_all["values"][0]["values"] == [4, 4]
