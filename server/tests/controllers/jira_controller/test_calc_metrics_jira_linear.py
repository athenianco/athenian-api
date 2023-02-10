from datetime import date
from typing import Any

import numpy as np
import pytest

from athenian.api.db import Database
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.settings import LogicalRepositorySettings
from athenian.api.models.web import (
    CalculatedJIRAMetricValues,
    CalculatedLinearMetricValues,
    JIRAMetricID,
)
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_ACCOUNT_ID
from tests.testutils.factory.state import MappedJIRAIdentityFactory, TeamFactory
from tests.testutils.factory.wizards import jira_issue_models
from tests.testutils.requester import Requester
from tests.testutils.time import dt


class BaseCalcMetricsJiraLinearTest(Requester):
    async def _request(self, *, assert_status=200, **kwargs) -> dict | list:
        response = await self.client.request(
            method="POST", path="/v1/metrics/jira", headers=self.headers, **kwargs,
        )
        assert response.status == assert_status
        return await response.json()

    @classmethod
    def _body(self, **kwargs: Any) -> dict:
        kwargs.setdefault("account", DEFAULT_ACCOUNT_ID)
        kwargs.setdefault("exclude_inactive", True)
        kwargs.setdefault("granularities", ["all"])
        kwargs.setdefault("timezone", 0)
        if "with" not in kwargs and "with_" in kwargs:
            kwargs["with"] = kwargs.pop("with_")
        if "for" not in kwargs and "for_" in kwargs:
            kwargs["for"] = kwargs.pop("for_")
        return kwargs


class TestCalcMetricsJiraLinearErrors(BaseCalcMetricsJiraLinearTest):
    async def test_empty_granularities(self) -> None:
        body = self._body(
            date_from="2020-09-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            granularities=[],
        )
        res = await self._request(json=body, assert_status=400)
        assert isinstance(res, dict)
        assert "granularities" in res["detail"]


class TestCalcMetricsJiraLinear(BaseCalcMetricsJiraLinearTest):
    @pytest.mark.parametrize("exclude_inactive", [False, True])
    async def test_smoke(self, exclude_inactive):
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-23",
            timezone=120,
            metrics=[JIRAMetricID.JIRA_RAISED, JIRAMetricID.JIRA_RESOLVED],
            exclude_inactive=exclude_inactive,
            granularities=["all", "2 month"],
        )
        res = await self._request(json=body)
        assert len(res) == 2
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert items[0].granularity == "all"
        assert items[0].with_ is None
        assert items[0].values == [
            CalculatedLinearMetricValues(
                date=date(2020, 1, 1),
                values=[1699, 1573],
                confidence_mins=[None] * 2,
                confidence_maxs=[None] * 2,
                confidence_scores=[None] * 2,
            ),
        ]
        assert items[1].granularity == "2 month"
        assert items[1].with_ is None
        assert len(items[1].values) == 5
        assert items[1].values[0] == CalculatedLinearMetricValues(
            date=date(2020, 1, 1),
            values=[160, 39],
            confidence_mins=[None] * 2,
            confidence_maxs=[None] * 2,
            confidence_scores=[None] * 2,
        )
        assert items[1].values[-1] == CalculatedLinearMetricValues(
            date=date(2020, 9, 1),
            values=[237, 238],
            confidence_mins=[None] * 2,
            confidence_maxs=[None] * 2,
            confidence_scores=[None] * 2,
        )

    @pytest.mark.parametrize(
        "account, metrics, date_from, date_to, timezone, granularities, status",
        [
            (1, [JIRAMetricID.JIRA_RAISED], "2020-01-01", "2020-04-01", 120, ["all"], 200),
            (2, [JIRAMetricID.JIRA_RAISED], "2020-01-01", "2020-04-01", 120, ["all"], 422),
            (3, [JIRAMetricID.JIRA_RAISED], "2020-01-01", "2020-04-01", 120, ["all"], 404),
            (1, [], "2020-01-01", "2020-04-01", 120, ["all"], 200),
            (1, None, "2020-01-01", "2020-04-01", 120, ["all"], 400),
            (1, [JIRAMetricID.JIRA_RAISED], "2020-05-01", "2020-04-01", 120, ["all"], 400),
            (1, [JIRAMetricID.JIRA_RAISED], "2020-01-01", "2020-04-01", 100500, ["all"], 400),
            (1, [JIRAMetricID.JIRA_RAISED], "2020-01-01", "2020-04-01", 120, ["whatever"], 400),
        ],
    )
    async def test_nasty_input(
        self,
        account,
        metrics,
        date_from,
        date_to,
        timezone,
        granularities,
        status,
    ):
        body = self._body(
            date_from=date_from,
            date_to=date_to,
            timezone=timezone,
            account=account,
            metrics=metrics,
            with_=[],
            granularities=granularities,
        )
        await self._request(json=body, assert_status=status)

    async def test_priorities(self) -> None:
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            priorities=["high"],
        )
        res = await self._request(json=body)
        assert len(res) == 1
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert items[0].granularity == "all"
        assert items[0].values[0].values == [392]

    async def test_types(self):
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            types=["tASK"],
        )
        res = await self._request(json=body)
        assert len(res) == 1
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert items[0].granularity == "all"
        assert items[0].values[0].values == [658]

    async def test_epics(self):
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            epics=["DEV-70", "DEV-843"],
        )
        res = await self._request(json=body)
        assert len(res) == 1
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert items[0].granularity == "all"
        assert items[0].values[0].values == [38]

    async def test_labels(self):
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            labels_include=["PERFORmance"],
            labels_exclude=["buG"],
        )
        res = await self._request(json=body)
        assert len(res) == 1
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert items[0].granularity == "all"
        # 148 without labels_exclude
        # 147 without cleaning
        assert items[0].values[0].values == [142]

    @pytest.mark.parametrize(
        "assignees, reporters, commenters, count",
        [
            (["Vadim markovtsev"], ["waren long"], ["lou Marvin caraig"], 1136),
            (["Vadim markovtsev"], [], [], 529),
            ([], [], ["{1}"], 236),
            ([None, "Vadim MARKOVTSEV"], [], [], 694),
            ([], ["waren long"], [], 539),
            ([], [], ["lou Marvin caraig"], 236),
            ([None], [], ["lou Marvin caraig"], 381),
        ],
    )
    async def test_with(
        self,
        assignees,
        reporters,
        commenters,
        count,
        sample_team,
        sdb,
    ):
        await models_insert(
            sdb,
            MappedJIRAIdentityFactory(github_user_id=51, jira_user_id="5ddec0b9be6c1f0d071ff82d"),
        )
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            with_=[{"assignees": assignees, "reporters": reporters, "commenters": commenters}],
        )
        res = await self._request(json=body)
        assert len(res) == 1
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert items[0].granularity == "all"
        with_ = {}
        for key, val in {
            "assignees": assignees,
            "reporters": reporters,
            "commenters": commenters,
        }.items():
            if val:
                with_[key] = val
        assert items[0].with_.to_dict() == with_
        assert items[0].values[0].values == [count]

    async def test_participants_as_team_id(self, sdb: Database, mdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1),
            TeamFactory(id=2, members=[40020]),
            TeamFactory(id=3, members=[29]),
            MappedJIRAIdentityFactory(
                github_user_id=40020, jira_user_id="5de5049e2c5dd20d0f9040c1",
            ),
            MappedJIRAIdentityFactory(github_user_id=29, jira_user_id="5dd58cb9c7ac480ee5674902"),
        )
        body = self._body(
            date_from="2020-08-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            with_=[
                {"assignees": ["vadim Markovtsev"]},
                {"assignees": ["{2}"]},
                {"reporters": ["waren Long"]},
                {"reporters": ["{3}"]},
            ],
        )
        res = await self._request(json=body)

        assert len(res) == 4
        items = sorted(res, key=lambda i: str(i["with"]))
        assert items[0]["with"] == {"assignees": ["vadim Markovtsev"]}
        assert items[1]["with"] == {"assignees": ["{2}"]}
        assert items[0]["values"][0]["values"] == [155]
        assert items[1]["values"][0]["values"] == [155]

        assert items[2]["with"] == {"reporters": ["waren Long"]}
        assert items[3]["with"] == {"reporters": ["{3}"]}
        assert items[2]["values"][0]["values"] == [116]

    async def test_teams(self):
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            with_=[{"assignees": ["vadim Markovtsev"]}, {"reporters": ["waren Long"]}],
        )
        res = await self._request(json=body)
        assert len(res) == 2
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert items[0].granularity == "all"
        assert items[0].values[0].values == [529]
        assert items[0].with_.to_dict() == {"assignees": ["vadim Markovtsev"]}
        assert items[1].values[0].values == [539]
        assert items[1].with_.to_dict() == {"reporters": ["waren Long"]}

    @pytest.mark.parametrize(
        "metric, exclude_inactive, n",
        [
            (JIRAMetricID.JIRA_OPEN, False, 197),
            (JIRAMetricID.JIRA_OPEN, True, 188),
            (JIRAMetricID.JIRA_RESOLVED, False, 795),
            (JIRAMetricID.JIRA_RESOLVED, True, 795),
            (JIRAMetricID.JIRA_ACKNOWLEDGED, False, 776),
            (JIRAMetricID.JIRA_ACKNOWLEDGED_Q, False, 776),
            (JIRAMetricID.JIRA_RESOLUTION_RATE, False, 0.9578820466995239),
        ],
    )
    async def test_jira_metrics_counts(self, metric, exclude_inactive, n):
        body = self._body(
            date_from="2020-06-01",
            date_to="2020-10-23",
            timezone=120,
            metrics=[metric],
            exclude_inactive=exclude_inactive,
            granularities=["all"],
        )
        res = await self._request(json=body)
        assert len(res) == 1
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert items[0].granularity == "all"
        assert items[0].values == [
            CalculatedLinearMetricValues(
                date=date(2020, 6, 1),
                values=[n],
                confidence_mins=[None],
                confidence_maxs=[None],
                confidence_scores=[None],
            ),
        ]

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    @pytest.mark.parametrize(
        "metric, value, score, cmin, cmax",
        [
            (JIRAMetricID.JIRA_LIFE_TIME, "3360382s", 51, "2557907s", "4227690s"),
            (JIRAMetricID.JIRA_LEAD_TIME, "2922288s", 43, "2114455s", "3789853s"),
            (JIRAMetricID.JIRA_ACKNOWLEDGE_TIME, "450250s", 66, "369809s", "527014s"),
            (JIRAMetricID.JIRA_PR_LAG_TIME, "0s", 100, "0s", "0s"),
            (JIRAMetricID.JIRA_BACKLOG_TIME, "450856s", 66, "370707s", "527781s"),
        ],
    )
    @with_defer
    async def test_jira_metrics_bug_times(
        self,
        metric,
        value,
        score,
        cmin,
        cmax,
        pr_facts_calculator_factory,
        release_match_setting_tag,
        prefixer,
        bots,
    ):
        pr_facts_calculator_no_cache = pr_facts_calculator_factory(1, (6366825,))
        time_from = dt(2018, 1, 1)
        time_to = dt(2020, 4, 1)
        args = (
            time_from,
            time_to,
            {"src-d/go-git"},
            {},
            LabelFilter.empty(),
            JIRAFilter.empty(),
            False,
            bots,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            prefixer,
            False,
            0,
        )
        await pr_facts_calculator_no_cache(*args)
        await wait_deferred()
        np.random.seed(7)
        body = self._body(
            date_from="2016-01-01",
            date_to="2020-10-23",
            timezone=120,
            metrics=[metric],
            types=["BUG"],
            exclude_inactive=False,
            granularities=["all", "1 year"],
        )
        res = await self._request(json=body)
        assert len(res) == 2
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert items[0].granularity == "all"
        assert items[0].values == [
            CalculatedLinearMetricValues(
                date=date(2016, 1, 1),
                values=[value],
                confidence_mins=[cmin],
                confidence_maxs=[cmax],
                confidence_scores=[score],
            ),
        ]

    async def test_disabled_projects(self, disabled_dev):
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-23",
            timezone=120,
            metrics=[JIRAMetricID.JIRA_RAISED, JIRAMetricID.JIRA_RESOLVED],
            exclude_inactive=False,
            granularities=["all", "2 month"],
        )
        res = await self._request(json=body)
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        _check_metrics_no_dev_project(items)

    async def test_selected_projects(self, client_cache):
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-23",
            timezone=120,
            metrics=[JIRAMetricID.JIRA_RAISED, JIRAMetricID.JIRA_RESOLVED],
            exclude_inactive=False,
            projects=["PRO", "OPS", "ENG", "GRW", "CS", "CON"],
            granularities=["all", "2 month"],
        )
        res = await self._request(json=body)
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        _check_metrics_no_dev_project(items)

        del body["projects"]
        res = await self._request(json=body)
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        with pytest.raises(AssertionError):
            _check_metrics_no_dev_project(items)

    async def test_group_by_label_smoke(self) -> None:
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            group_by_jira_label=True,
        )
        res = await self._request(json=body)
        assert len(res) == 49
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert items[0].granularity == "all"
        assert items[0].jira_label == "performance"
        assert items[0].values[0].values == [143]
        assert items[1].jira_label == "webapp"
        assert items[1].values[0].values == [142]
        assert items[-1].jira_label is None
        assert items[-1].values[0].values == [729]

        body["labels_include"] = ["performance"]
        body["labels_exclude"] = ["security"]
        res = await self._request(json=body)
        assert len(res) == 1
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert items[0].granularity == "all"
        assert items[0].jira_label == "performance"
        assert items[0].values[0].values == [142]

    async def test_group_by_label_only_exclude(self) -> None:
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            group_by_jira_label=True,
            labels_exclude=["security"],
        )

        body["labels_exclude"] = ["security"]
        res = await self._request(json=body)

        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert len(items) == 48

        assert not any(i["jira_label"] == "security" for i in items)

        rel_item = next(i for i in items if i["jira_label"] == "reliability")
        assert rel_item["values"][0]["values"][0] == 83

        enhancement_item = next(i for i in items if i["jira_label"] == "enhancement")
        assert enhancement_item["values"][0]["values"][0] == 36

    @pytest.mark.xfail
    async def test_group_by_label_comma(self) -> None:
        # TODO: fix grouping when the label is a list of comma separated labels
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            group_by_jira_label=True,
        )

        body["labels_include"] = ["performance"]
        res = await self._request(json=body)
        assert len(res) == 1
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert items[0].jira_label == "performance"
        assert items[0].values[0].values == [143]

        body["labels_include"] = ["performance,security"]
        res = await self._request(json=body)
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert items[0].jira_label == "performance"
        assert items[0].values[0].values == [1]

        assert items[1].jira_label == "security"
        assert items[1].values[0].values == [1]

    async def test_group_by_label_empty(self) -> None:
        body = self._body(
            date_from="2019-12-02",
            date_to="2019-12-03",
            metrics=[JIRAMetricID.JIRA_RAISED],
            group_by_jira_label=True,
        )
        res = await self._request(json=body)
        assert len(res) == 1
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert len(items) == 1
        assert items[0].granularity == "all"
        assert items[0].jira_label is None
        assert items[0].values[0].values == [9]

        body["labels_include"] = ["whatever"]
        res = await self._request(json=body)
        assert len(res) == 1
        items = [CalculatedJIRAMetricValues.from_dict(i) for i in res]
        assert len(items) == 1
        assert items[0].granularity == "all"
        assert items[0].jira_label is None
        assert items[0].values[0].values == [0]


class TestGroups(BaseCalcMetricsJiraLinearTest):
    async def test_single_group_fixture(self) -> None:
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            for_=[{"priorities": ["high"]}],
        )
        res = await self._request(json=body)
        assert len(res) == 1
        assert res[0]["values"][0]["values"][0] == 392
        assert res[0]["for"] == {"priorities": ["high"]}

        body.pop("for")
        body["priorities"] = ["high"]

        res = await self._request(json=body)
        assert len(res) == 1
        assert res[0]["values"][0]["values"][0] == 392
        assert "for" not in res[0]

    async def test_two_groups_fixture(self) -> None:
        body = self._body(
            date_from="2020-09-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED, JIRAMetricID.JIRA_RESOLVED],
            for_=[
                {"issue_types": ["task"]},
                {"issue_types": ["bug"]},
                {"issue_types": ["task", "bug"]},
            ],
        )
        res = await self._request(json=body)
        assert len(res) == 3

        task_res = next(r for r in res if r["for"] == {"issue_types": ["task"]})
        assert task_res["values"][0]["values"] == [119, 118]

        bug_res = next(r for r in res if r["for"] == {"issue_types": ["bug"]})
        assert bug_res["values"][0]["values"] == [78, 83]

        both_res = next(r for r in res if r["for"] == {"issue_types": ["task", "bug"]})
        assert both_res["values"][0]["values"] == [197, 201]

        body.pop("for")
        body["types"] = ["task"]
        task_global_res = await self._request(json=body)
        assert task_global_res[0]["values"][0]["values"] == task_res["values"][0]["values"]

        body["types"] = ["bug"]
        bug_global_res = await self._request(json=body)
        assert bug_global_res[0]["values"][0]["values"] == bug_res["values"][0]["values"]

        body["types"] = ["task", "bug"]
        both_global_res = await self._request(json=body)
        assert both_global_res[0]["values"][0]["values"] == both_res["values"][0]["values"]

    @pytest.mark.app_validate_responses(False)
    async def test_more_groups(self, sdb: Database, mdb_rw: Database) -> None:
        body = self._body(
            date_from="2020-05-01",
            date_to="2020-6-20",
            metrics=[JIRAMetricID.JIRA_RAISED, JIRAMetricID.JIRA_LIFE_TIME],
            for_=[
                {"priorities": ["P0"]},
                {"projects": ["PJ1", "PJ2"]},
                {"projects": ["PJ2"]},
                {"priorities": ["P0"], "projects": ["PJ2"]},
                {"priorities": ["P0"], "projects": ["PJ1"]},
                {"priorities": ["P1"]},
            ],
        )

        issue_kwargs = {"created": dt(2020, 5, 1)}
        prio0 = {"priority_id": "00", "priority_name": "p0"}
        prio1 = {"priority_id": "10", "priority_name": "p1"}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="PJ1"),
            md_factory.JIRAProjectFactory(id="2", key="PJ2"),
            md_factory.JIRAPriorityFactory(id="00", name="P0"),
            md_factory.JIRAPriorityFactory(id="10", name="P1"),
            *jira_issue_models(
                "1", resolved=dt(2020, 5, 1, 10), project_id="1", **prio0, **issue_kwargs,
            ),
            *jira_issue_models(
                "2", project_id="2", resolved=dt(2020, 5, 1, 8), **prio0, **issue_kwargs,
            ),
            *jira_issue_models(
                "3", project_id="1", resolved=dt(2020, 5, 1, 5), **prio1, **issue_kwargs,
            ),
            *jira_issue_models("4", project_id="1", **prio1, **issue_kwargs),
            *jira_issue_models("5", project_id="2", **prio0, **issue_kwargs),
            *jira_issue_models(
                "6", project_id="2", resolved=dt(2020, 5, 1, 4), **prio0, **issue_kwargs,
            ),
            *jira_issue_models("7", project_id="2", **prio0, **issue_kwargs),
            *jira_issue_models(
                "8", resolved=dt(2020, 5, 1, 4), project_id="2", **prio1, **issue_kwargs,
            ),
            *jira_issue_models("9", project_id="2", **prio0, created=dt(2021, 1, 1)),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            res = await self._request(json=body)
            res = sorted(res, key=lambda r: body["for"].index(r["for"]))

            for i, r in enumerate(res):
                assert r["for"] == body["for"][i]

            assert res[0]["values"][0]["values"] == [5, "26400s"]  # life times are 10h,8h,4h
            assert res[1]["values"][0]["values"] == [8, "22320s"]  # life times are 10,8,5,4,4
            assert res[2]["values"][0]["values"] == [5, "19200s"]  # life times are 8,4,4
            assert res[3]["values"][0]["values"] == [4, "21600s"]  # life times are 8,4
            assert res[4]["values"][0]["values"] == [1, "36000s"]  # life times are 10
            assert res[5]["values"][0]["values"] == [3, "16200s"]  # life times are 5,4

            assert len(res) == 6

    async def test_empty_groups(self, sdb: Database, mdb_rw: Database) -> None:
        body = self._body(
            date_from="2001-04-01",
            date_to="2001-05-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            for_=[
                {"projects": ["PJ1"]},
                {},
            ],
        )
        issue_kwargs = {"created": dt(2001, 5, 1)}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="PJ1"),
            md_factory.JIRAProjectFactory(id="2", key="PJ2"),
            *jira_issue_models("1", project_id="1", **issue_kwargs),
            *jira_issue_models("2", project_id="2", **issue_kwargs),
            *jira_issue_models("3", project_id="2", **issue_kwargs),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            res = await self._request(json=body)
            assert len(res) == 2
            res_p1 = next(r for r in res if r["for"] == {"projects": ["PJ1"]})
            assert res_p1["values"][0]["values"] == [1]
            res_all = next(r for r in res if r["for"] == {})
            assert res_all["values"][0]["values"] == [3]

    async def test_labels(self, sdb: Database, mdb_rw: Database) -> None:
        body = self._body(
            date_from="2012-05-01",
            date_to="2012-06-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            for_=[
                {"labels_include": ["l0"]},
                {"labels_include": ["l1"]},
                {"labels_include": ["l0", "l1"]},
                {"projects": ["PJ1"]},
            ],
        )

        issue_kwargs = {"created": dt(2012, 5, 1), "project_id": "1"}
        mdb_models = [
            md_factory.JIRAProjectFactory(id="1", key="PJ1"),
            *jira_issue_models("1", labels=["l0"], **issue_kwargs),
            *jira_issue_models("2", labels=["l1"], **issue_kwargs),
            *jira_issue_models("3", labels=["l2", "l1"], **issue_kwargs),
            *jira_issue_models("4", labels=["l2", "l0"], **issue_kwargs),
            *jira_issue_models("5", labels=[], **issue_kwargs),
            *jira_issue_models("6", labels=["l3", "l2"], **issue_kwargs),
            *jira_issue_models("7", labels=["l3", "l2", "l1"], **issue_kwargs),
            *jira_issue_models("8", labels=["l0", "l1"], **issue_kwargs),
            *jira_issue_models("9", labels=["l3", "l0"], **issue_kwargs),
            *jira_issue_models("10", labels=["l3", "l1"], **issue_kwargs),
            # out of interval
            *jira_issue_models("11", labels=["l3", "l0"], created=dt(2012, 7, 1), project_id="1"),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            res = await self._request(json=body)
        res = sorted(res, key=lambda r: body["for"].index(r["for"]))

        assert res[0]["for"] == {"labels_include": ["l0"]}
        assert res[0]["values"][0]["values"] == [4]

        assert res[1]["for"] == {"labels_include": ["l1"]}
        assert res[1]["values"][0]["values"] == [5]

        assert res[2]["for"] == {"labels_include": ["l0", "l1"]}
        assert res[2]["values"][0]["values"] == [8]

        assert res[3]["for"] == {"projects": ["PJ1"]}
        assert res[3]["values"][0]["values"] == [10]


class TestGroupsErrors(BaseCalcMetricsJiraLinearTest):
    async def test_both_groups_and_filters(self) -> None:
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            for_=[{"issue_types": ["bug", "task"]}],
            labels_include=["foolabel"],
            projects=["p1", "p2"],
        )
        res = await self._request(assert_status=400, json=body)
        assert isinstance(res, dict)
        assert "`for` cannot be used with" in res["detail"]
        assert "labels_include" in res["detail"]
        assert "projects" in res["detail"]

    async def test_both_groups_and_group_by_label(self) -> None:
        body = self._body(
            date_from="2020-01-01",
            date_to="2020-10-20",
            metrics=[JIRAMetricID.JIRA_RAISED],
            for_=[{"issue_types": ["bug", "task"]}],
            group_by_jira_label=True,
        )
        res = await self._request(assert_status=400, json=body)
        assert isinstance(res, dict)
        assert "`for` cannot be used with" in res["detail"]
        assert "group_by_jira_label" in res["detail"]


def _check_metrics_no_dev_project(items: list[CalculatedJIRAMetricValues]) -> None:
    assert items[0].values == [
        CalculatedLinearMetricValues(
            date=date(2020, 1, 1),
            values=[767, 829],
            confidence_mins=[None] * 2,
            confidence_maxs=[None] * 2,
            confidence_scores=[None] * 2,
        ),
    ]
