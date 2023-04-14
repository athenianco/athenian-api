from datetime import date
from typing import Any

import pytest
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.internal.features.github.deployment_metrics import CHANGE_FAILURE_LABEL
from athenian.api.models.persistentdata.models import DeployedComponent, DeployedLabel
from athenian.api.models.web import CalculatedDeploymentMetric, DeploymentMetricID
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_ACCOUNT_ID
from tests.testutils.factory.persistentdata import (
    DeployedComponentFactory,
    DeployedLabelFactory,
    DeploymentNotificationFactory,
)
from tests.testutils.factory.wizards import commit_models, insert_repo
from tests.testutils.requester import Requester
from tests.testutils.time import dt


class BaseCalcMetricsDeploymentsTest(Requester):
    path = "v1/metrics/deployments"

    @classmethod
    def _body(self, **kwargs: Any) -> dict:
        kwargs.setdefault("account", DEFAULT_ACCOUNT_ID)
        kwargs.setdefault("granularities", ["all"])
        if "for" not in kwargs and "for_" in kwargs:
            kwargs["for"] = kwargs.pop("for_")
        return kwargs


class TestCalcMetricsDeployments(BaseCalcMetricsDeploymentsTest):
    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_deployment_metrics_smoke(self, sample_deployments):
        body = self._body(
            date_from="2018-01-12",
            date_to="2020-03-01",
            for_=[
                {
                    "repositories": ["{1}"],
                    "withgroups": [
                        {"releaser": ["github.com/mcuadros"]},
                        {"pr_author": ["github.com/mcuadros"]},
                    ],
                    "environments": ["staging", "production", "mirror"],
                },
            ],
            metrics=[
                DeploymentMetricID.DEP_SUCCESS_COUNT,
                DeploymentMetricID.DEP_DURATION_SUCCESSFUL,
            ],
        )
        res = await self.post_json(json=body)
        model = [CalculatedDeploymentMetric.from_dict(obj) for obj in res]
        assert [m.to_dict() for m in model] == [
            {
                "for": {
                    "repositories": ["{1}"],
                    "with": {"releaser": ["github.com/mcuadros"]},
                    "environments": ["staging"],
                },
                "metrics": ["dep-success-count", "dep-duration-successful"],
                "granularity": "all",
                "values": [
                    {
                        "date": date(2018, 1, 12),
                        "values": [3, "600s"],
                        "confidence_maxs": [None, "600s"],
                        "confidence_mins": [None, "600s"],
                        "confidence_scores": [None, 100],
                    },
                ],
            },
            {
                "for": {
                    "repositories": ["{1}"],
                    "with": {"releaser": ["github.com/mcuadros"]},
                    "environments": ["production"],
                },
                "metrics": ["dep-success-count", "dep-duration-successful"],
                "granularity": "all",
                "values": [
                    {
                        "date": date(2018, 1, 12),
                        "values": [3, "600s"],
                        "confidence_maxs": [None, "600s"],
                        "confidence_mins": [None, "600s"],
                        "confidence_scores": [None, 100],
                    },
                ],
            },
            {
                "for": {
                    "repositories": ["{1}"],
                    "with": {"releaser": ["github.com/mcuadros"]},
                    "environments": ["mirror"],
                },
                "metrics": ["dep-success-count", "dep-duration-successful"],
                "granularity": "all",
                "values": [{"date": date(2018, 1, 12), "values": [0, None]}],
            },
            {
                "for": {
                    "repositories": ["{1}"],
                    "with": {"pr_author": ["github.com/mcuadros"]},
                    "environments": ["staging"],
                },
                "metrics": ["dep-success-count", "dep-duration-successful"],
                "granularity": "all",
                "values": [
                    {
                        "date": date(2018, 1, 12),
                        "values": [3, "600s"],
                        "confidence_maxs": [None, "600s"],
                        "confidence_mins": [None, "600s"],
                        "confidence_scores": [None, 100],
                    },
                ],
            },
            {
                "for": {
                    "repositories": ["{1}"],
                    "with": {"pr_author": ["github.com/mcuadros"]},
                    "environments": ["production"],
                },
                "metrics": ["dep-success-count", "dep-duration-successful"],
                "granularity": "all",
                "values": [
                    {
                        "date": date(2018, 1, 12),
                        "values": [3, "600s"],
                        "confidence_maxs": [None, "600s"],
                        "confidence_mins": [None, "600s"],
                        "confidence_scores": [None, 100],
                    },
                ],
            },
            {
                "for": {
                    "repositories": ["{1}"],
                    "with": {"pr_author": ["github.com/mcuadros"]},
                    "environments": ["mirror"],
                },
                "metrics": ["dep-success-count", "dep-duration-successful"],
                "granularity": "all",
                "values": [{"date": date(2018, 1, 12), "values": [0, None]}],
            },
        ]

    async def test_empty_for(self, sample_deployments):
        body = self._body(
            date_from="2018-01-12",
            date_to="2020-03-01",
            for_=[{}],
            metrics=[DeploymentMetricID.DEP_SUCCESS_COUNT],
        )
        res = await self.post_json(json=body)
        model = [CalculatedDeploymentMetric.from_dict(obj) for obj in res]
        assert [m.to_dict() for m in model] == [
            {
                "for": {},
                "granularity": "all",
                "metrics": [DeploymentMetricID.DEP_SUCCESS_COUNT],
                "values": [{"date": date(2018, 1, 12), "values": [9]}],
            },
        ]

    @pytest.mark.parametrize(
        "account, date_from, date_to, repos, withgroups, metrics, code",
        [
            (1, "2018-01-12", "2020-01-12", ["{1}"], [], [DeploymentMetricID.DEP_PRS_COUNT], 200),
            (1, "2020-01-12", "2018-01-12", ["{1}"], [], [DeploymentMetricID.DEP_PRS_COUNT], 400),
            (
                2,
                "2018-01-12",
                "2020-01-12",
                ["github.com/src-d/go-git"],
                [],
                [DeploymentMetricID.DEP_PRS_COUNT],
                422,
            ),
            (
                3,
                "2018-01-12",
                "2020-01-12",
                ["github.com/src-d/go-git"],
                [],
                [DeploymentMetricID.DEP_PRS_COUNT],
                404,
            ),
            (1, "2018-01-12", "2020-01-12", ["{1}"], [], ["whatever"], 400),
            (
                1,
                "2018-01-12",
                "2020-01-12",
                ["github.com/athenianco/athenian-api"],
                [],
                [DeploymentMetricID.DEP_PRS_COUNT],
                403,
            ),
            (
                1,
                "2018-01-12",
                "2020-01-12",
                ["{1}"],
                [{"pr_author": ["github.com/akbarik"]}],
                [DeploymentMetricID.DEP_PRS_COUNT],
                400,
            ),
        ],
    )
    async def test_deployment_metrics_nasty_input(
        self,
        account,
        date_from,
        date_to,
        repos,
        withgroups,
        metrics,
        code,
    ):
        body = self._body(
            account=account,
            date_from=date_from,
            date_to=date_to,
            for_=[{"repositories": [*repos], "withgroups": [*withgroups]}],
            metrics=[*metrics],
        )
        await self.post_json(json=body, assert_status=code)

    async def test_deployment_metrics_filter_labels(
        self,
        precomputed_deployments,
        rdb,
        client_cache,
    ):
        body = self._body(
            date_from="2015-01-12",
            date_to="2020-03-01",
            for_=[{"pr_labels_include": ["bug", "plumbing", "enhancement"]}],
            metrics=[DeploymentMetricID.DEP_COUNT],
        )

        async def request():
            res = await self.post_json(json=body)
            return [CalculatedDeploymentMetric.from_dict(obj) for obj in res]

        model = await request()
        assert model[0].values[0].values[0] == 1

        del body["for"][0]["pr_labels_include"]
        body["for"][0]["pr_labels_exclude"] = ["bug", "plumbing", "enhancement"]
        model = await request()
        assert model[0].values[0].values[0] == 0

        del body["for"][0]["pr_labels_exclude"]
        await rdb.execute(
            sa.insert(DeployedLabel).values(
                {
                    DeployedLabel.account_id: 1,
                    DeployedLabel.deployment_name: "Dummy deployment",
                    DeployedLabel.key: "nginx",
                    DeployedLabel.value: 504,
                },
            ),
        )
        body["for"][0]["with_labels"] = {"nginx": 503}
        model = await request()
        assert model[0].values[0].values[0] == 0

        body["for"][0]["with_labels"] = {"nginx": 504}
        model = await request()
        assert model[0].values[0].values[0] == 1

        del body["for"][0]["with_labels"]
        body["for"][0]["without_labels"] = {"nginx": 503}
        model = await request()
        assert model[0].values[0].values[0] == 1

        body["for"][0]["without_labels"] = {"nginx": 504}
        model = await request()
        assert model[0].values[0].values[0] == 0

    async def test_deployment_metrics_environments(
        self,
        sample_deployments,
        rdb,
        client_cache,
    ):
        await rdb.execute(
            sa.delete(DeployedComponent).where(
                DeployedComponent.deployment_name == "staging_2018_12_02",
            ),
        )
        body = self._body(
            date_from="2018-01-12",
            date_to="2020-03-01",
            for_=[{"envgroups": [["production"]]}],
            metrics=[
                DeploymentMetricID.DEP_SUCCESS_COUNT,
                DeploymentMetricID.DEP_DURATION_SUCCESSFUL,
            ],
        )

        res = await self.post_json(json=body)
        model = [CalculatedDeploymentMetric.from_dict(obj) for obj in res]
        assert len(model) == 1
        assert model[0].values[0].values[0] == 3
        body["for"][0]["envgroups"] = [["staging"], ["production"]]
        res = await self.post_json(json=body)
        model = [CalculatedDeploymentMetric.from_dict(obj) for obj in res]
        assert len(model) == 2
        assert model[0].for_.environments == ["staging"]
        assert model[0].values[0].values[0] == 3
        assert model[1].for_.environments == ["production"]
        assert model[1].values[0].values[0] == 3

    async def test_deployment_metrics_with(self, sample_deployments):
        body = self._body(
            date_from="2018-01-12",
            date_to="2020-03-01",
            for_=[
                {"with": {"pr_author": ["github.com/mcuadros"]}, "environments": ["production"]},
            ],
            metrics=[
                DeploymentMetricID.DEP_SUCCESS_COUNT,
                DeploymentMetricID.DEP_DURATION_SUCCESSFUL,
            ],
        )
        res = await self.post_json(json=body)
        model = [CalculatedDeploymentMetric.from_dict(obj) for obj in res]
        assert len(model) == 1
        assert model[0].values[0].values[0] == 3

    @pytest.mark.app_validate_responses(False)
    async def test_deployment_metrics_team(self, sample_deployments, sample_team):
        team_str = "{%d}" % sample_team
        body = self._body(
            date_from="2018-01-12",
            date_to="2020-03-01",
            for_=[
                {
                    "repositories": ["{1}"],
                    "withgroups": [{"releaser": [team_str]}, {"pr_author": [team_str]}],
                    "environments": ["production"],
                },
            ],
            metrics=[
                DeploymentMetricID.DEP_SUCCESS_COUNT,
                DeploymentMetricID.DEP_DURATION_SUCCESSFUL,
            ],
        )
        res = await self.post_json(json=body)
        model = [CalculatedDeploymentMetric.from_dict(obj) for obj in res]
        assert [m.to_dict() for m in model] == [
            {
                "for": {
                    "repositories": ["{1}"],
                    "with": {"releaser": [team_str]},
                    "environments": ["production"],
                },
                "metrics": ["dep-success-count", "dep-duration-successful"],
                "granularity": "all",
                "values": [
                    {
                        "date": date(2018, 1, 12),
                        "values": [3, "600s"],
                        "confidence_maxs": [None, "600s"],
                        "confidence_mins": [None, "600s"],
                        "confidence_scores": [None, 100],
                    },
                ],
            },
            {
                "for": {
                    "repositories": ["{1}"],
                    "with": {"pr_author": [team_str]},
                    "environments": ["production"],
                },
                "metrics": ["dep-success-count", "dep-duration-successful"],
                "granularity": "all",
                "values": [
                    {
                        "date": date(2018, 1, 12),
                        "values": [3, "600s"],
                        "confidence_maxs": [None, "600s"],
                        "confidence_mins": [None, "600s"],
                        "confidence_scores": [None, 100],
                    },
                ],
            },
        ]

    async def test_deployment_metrics_empty_granularities(self, sample_deployments):
        body = self._body(
            date_from="2018-01-12",
            date_to="2020-03-01",
            for_=[{}],
            metrics=[DeploymentMetricID.DEP_SUCCESS_COUNT],
            granularities=[],
        )
        res = await self.post_json(json=body, assert_status=400)
        assert "granularities" in res["detail"]


class TestChangeFailureMetrics(BaseCalcMetricsDeploymentsTest):
    async def test_base(self, mdb_rw: Database, rdb: Database, sdb: Database) -> None:
        body = self._body(
            date_from="2005-01-01",
            date_to="2005-03-01",
            for_=[{}],
            metrics=[
                DeploymentMetricID.DEP_COUNT,
                DeploymentMetricID.DEP_CHANGE_FAILURE_COUNT,
                DeploymentMetricID.DEP_CHANGE_FAILURE_RATIO,
            ],
        )
        DepFact = DeploymentNotificationFactory
        DepCompFact = DeployedComponentFactory
        DepLabelFact = DeployedLabelFactory
        await models_insert(
            rdb,
            DepFact(name="d1", finished_at=dt(2005, 1, 10), started_at=dt(2005, 1, 1)),
            DepCompFact(deployment_name="d1", repository_node_id=9, resolved_commit_node_id=1),
            DepLabelFact(deployment_name="d1", key="foo", value=True),
            DepFact(name="d2", finished_at=dt(2005, 2, 1), started_at=dt(2005, 1, 1)),
            DepCompFact(deployment_name="d2", repository_node_id=9, resolved_commit_node_id=2),
            DepLabelFact(deployment_name="d2", key=CHANGE_FAILURE_LABEL, value=None),
            DepFact(name="d3", finished_at=dt(2005, 2, 1), started_at=dt(2005, 1, 1)),
            DepCompFact(deployment_name="d3", repository_node_id=9, resolved_commit_node_id=3),
            DepLabelFact(deployment_name="d3", key=CHANGE_FAILURE_LABEL, value=["DEV-123"]),
            DepFact(name="d4", finished_at=dt(2004, 12, 29), started_at=dt(2004, 12, 29)),
            DepCompFact(deployment_name="d4", repository_node_id=9, resolved_commit_node_id=4),
            DepLabelFact(deployment_name="d4", key=CHANGE_FAILURE_LABEL, value=True),
            DepFact(name="d5", finished_at=dt(2005, 2, 1), started_at=dt(2005, 1, 1)),
            DepCompFact(deployment_name="d5", repository_node_id=9, resolved_commit_node_id=5),
            DepFact(
                name="d6",
                finished_at=dt(2005, 2, 1),
                started_at=dt(2005, 1, 1),
                conclusion="FAILURE",
            ),
            DepCompFact(deployment_name="d6", repository_node_id=9, resolved_commit_node_id=6),
        )
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo0 = md_factory.RepositoryFactory(node_id=9, full_name="o/r")
            await insert_repo(repo0, mdb_cleaner, mdb_rw, sdb)
            commit_kwargs = {"repository_full_name": "o/r", "repository_id": 9}
            models = [
                *commit_models(node_id=1, oid="A" * 40, **commit_kwargs),
                *commit_models(node_id=2, oid="B" * 40, **commit_kwargs),
                *commit_models(node_id=3, oid="C" * 40, **commit_kwargs),
                *commit_models(node_id=4, oid="D" * 40, **commit_kwargs),
                *commit_models(node_id=5, oid="E" * 40, **commit_kwargs),
                *commit_models(node_id=6, oid="F" * 40, **commit_kwargs),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            res = await self.post_json(json=body)

        # d2, d3 and d4 are marked as change failure
        # d4 is out of interval
        # d6 is not successful, is excluded
        assert res[0]["values"][0]["values"][0] == 5
        assert res[0]["values"][0]["values"][1] == 2
        assert res[0]["values"][0]["values"][2] == pytest.approx(2 / 4)
