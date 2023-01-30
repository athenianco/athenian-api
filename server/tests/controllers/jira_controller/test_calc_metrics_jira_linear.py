from datetime import date
from typing import Any

import numpy as np
import pytest

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.settings import LogicalRepositorySettings
from athenian.api.models.web import (
    CalculatedJIRAMetricValues,
    CalculatedLinearMetricValues,
    JIRAMetricID,
)
from tests.testutils.db import models_insert
from tests.testutils.factory.common import DEFAULT_ACCOUNT_ID
from tests.testutils.factory.state import MappedJIRAIdentityFactory
from tests.testutils.requester import Requester
from tests.testutils.time import dt


class TestCalcMetricsJiraLinear(Requester):
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
        return kwargs

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
                values=[1699, 1628],
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
            values=[237, 243],
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
            (JIRAMetricID.JIRA_OPEN, False, 142),
            (JIRAMetricID.JIRA_OPEN, True, 133),
            (JIRAMetricID.JIRA_RESOLVED, False, 850),
            (JIRAMetricID.JIRA_RESOLVED, True, 850),
            (JIRAMetricID.JIRA_ACKNOWLEDGED, False, 776),
            (JIRAMetricID.JIRA_ACKNOWLEDGED_Q, False, 776),
            (JIRAMetricID.JIRA_RESOLUTION_RATE, False, 1.0240963697433472),
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
