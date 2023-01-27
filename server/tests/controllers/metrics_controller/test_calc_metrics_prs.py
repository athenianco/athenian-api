from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.release_mine import mine_releases, override_first_releases
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseMatch
from athenian.api.models.state.models import AccountJiraInstallation, Team
from athenian.api.models.web import (
    CalculatedPullRequestMetrics,
    PullRequestMetricID,
    PullRequestWith,
)
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_ACCOUNT_ID
from tests.testutils.factory.state import ReleaseSettingFactory
from tests.testutils.factory.wizards import insert_repo, pr_models
from tests.testutils.requester import Requester
from tests.testutils.time import dt


class BaseCalcMetricsPRsTest(Requester):
    """Tests for endpoint /v1/metrics/pull_requests"""

    async def _request(self, *, assert_status=200, **kwargs) -> dict:
        response = await self.client.request(
            method="POST", path="/v1/metrics/pull_requests", headers=self.headers, **kwargs,
        )
        assert response.status == assert_status, (await response.read()).decode()
        return await response.json()

    @classmethod
    def _body(self, **kwargs: Any) -> dict:
        kwargs.setdefault("account", DEFAULT_ACCOUNT_ID)
        kwargs.setdefault("exclude_inactive", False)
        kwargs.setdefault("granularities", ["all"])

        if "for" not in kwargs and "for_" in kwargs:
            kwargs["for"] = kwargs.pop("for_")
        return kwargs


class TestCalcMetricsPRs(BaseCalcMetricsPRsTest):
    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    @pytest.mark.parametrize(
        "metric, count",
        [
            (PullRequestMetricID.PR_WIP_TIME, 51),
            (PullRequestMetricID.PR_WIP_PENDING_COUNT, 0),
            (PullRequestMetricID.PR_WIP_COUNT, 51),
            (PullRequestMetricID.PR_WIP_COUNT_Q, 51),
            (PullRequestMetricID.PR_REVIEW_TIME, 46),
            (PullRequestMetricID.PR_REVIEW_PENDING_COUNT, 0),
            (PullRequestMetricID.PR_REVIEW_COUNT, 46),
            (PullRequestMetricID.PR_REVIEW_COUNT_Q, 46),
            (PullRequestMetricID.PR_MERGING_TIME, 50),
            (PullRequestMetricID.PR_MERGING_PENDING_COUNT, 0),
            (PullRequestMetricID.PR_MERGING_COUNT, 50),
            (PullRequestMetricID.PR_MERGING_COUNT_Q, 50),
            (PullRequestMetricID.PR_RELEASE_TIME, 19),
            (PullRequestMetricID.PR_RELEASE_PENDING_COUNT, 189),
            (PullRequestMetricID.PR_RELEASE_COUNT, 19),
            (PullRequestMetricID.PR_RELEASE_COUNT_Q, 19),
            (PullRequestMetricID.PR_OPEN_TIME, 51),
            (PullRequestMetricID.PR_OPEN_COUNT, 51),
            (PullRequestMetricID.PR_OPEN_COUNT_Q, 51),
            (PullRequestMetricID.PR_LEAD_TIME, 19),
            (PullRequestMetricID.PR_LEAD_COUNT, 19),
            (PullRequestMetricID.PR_LEAD_COUNT_Q, 19),
            (PullRequestMetricID.PR_LIVE_CYCLE_TIME, 71),
            (PullRequestMetricID.PR_LIVE_CYCLE_COUNT, 71),
            (PullRequestMetricID.PR_LIVE_CYCLE_COUNT_Q, 71),
            (PullRequestMetricID.PR_ALL_COUNT, 200),
            (PullRequestMetricID.PR_FLOW_RATIO, 224),
            (PullRequestMetricID.PR_OPENED, 51),
            (PullRequestMetricID.PR_REVIEWED, 45),
            (PullRequestMetricID.PR_NOT_REVIEWED, 19),
            (PullRequestMetricID.PR_MERGED, 50),
            (PullRequestMetricID.PR_REJECTED, 3),
            (PullRequestMetricID.PR_CLOSED, 51),
            (PullRequestMetricID.PR_DONE, 22),
            (PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME, 51),
            (PullRequestMetricID.PR_WAIT_FIRST_REVIEW_COUNT, 51),
            (PullRequestMetricID.PR_WAIT_FIRST_REVIEW_COUNT_Q, 51),
            (PullRequestMetricID.PR_SIZE, 51),
            (PullRequestMetricID.PR_DEPLOYMENT_TIME, 0),  # because pdb is empty
        ],
    )
    async def test_smoke(self, metric, count, app, client_cache):
        """Trivial test to prove that at least something is working."""
        needs_env = metric == PullRequestMetricID.PR_DEPLOYMENT_TIME
        req_body = self._body(
            metrics=[metric],
            date_from="2015-10-13",
            date_to="2020-01-23",
            granularities=["week"],
            for_=[
                {
                    "with": {"author": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                    "repositories": ["github.com/src-d/go-git"],
                    **({"environments": ["production"]} if needs_env else {}),
                },
            ],
        )

        for _ in range(2):
            body = await self._request(json=req_body)
            cm = CalculatedPullRequestMetrics.from_dict(body)
            assert len(cm.calculated) == 1
            assert len(cm.calculated[0].values) > 0
            s = 0
            is_int = "TIME" not in metric
            for val in cm.calculated[0].values:
                assert len(val.values) == 1
                m = val.values[0]
                if is_int:
                    s += m != 0 and m is not None
                else:
                    s += m is not None
            assert s == count

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_all_time(self, headers):
        """https://athenianco.atlassian.net/browse/ENG-116"""
        devs = ["github.com/vmarkovtsev", "github.com/mcuadros"]
        for_block = {
            "with": {
                "author": devs,
                "merger": devs,
                "releaser": devs,
                "commenter": devs,
                "reviewer": devs,
                "commit_author": devs,
                "commit_committer": devs,
            },
            "repositories": ["github.com/src-d/go-git"],
        }
        body = self._body(
            for_=[for_block, for_block],
            metrics=[
                PullRequestMetricID.PR_WIP_TIME,
                PullRequestMetricID.PR_REVIEW_TIME,
                PullRequestMetricID.PR_MERGING_TIME,
                PullRequestMetricID.PR_RELEASE_TIME,
                PullRequestMetricID.PR_OPEN_TIME,
                PullRequestMetricID.PR_LEAD_TIME,
                PullRequestMetricID.PR_CYCLE_TIME,
                PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME,
            ],
            date_from="2015-10-13",
            date_to="2019-03-15",
            timezone=60,
            granularities=["day", "week", "month"],
        )
        body = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(body)
        assert cm.calculated[0].values[0].date == date(year=2015, month=10, day=13)
        assert cm.calculated[0].values[-1].date <= date(year=2019, month=3, day=15)
        for i in range(len(cm.calculated[0].values) - 1):
            assert cm.calculated[0].values[i].date < cm.calculated[0].values[i + 1].date
        cmins = cmaxs = cscores = 0
        gcounts = defaultdict(int)
        assert len(cm.calculated) == 6
        for calc in cm.calculated:
            assert calc.for_.with_ == PullRequestWith(**for_block["with"])
            assert calc.for_.repositories == ["github.com/src-d/go-git"]
            gcounts[calc.granularity] += 1
            nonzero = defaultdict(int)
            for val in calc.values:
                for m, t in zip(cm.metrics, val.values):
                    if t is None:
                        continue
                    assert pd.to_timedelta(t) >= timedelta(0), "Metric: %s\nValues: %s" % (
                        m,
                        val.values,
                    )
                    nonzero[m] += pd.to_timedelta(t) > timedelta(0)
                if val.confidence_mins is not None:
                    cmins += 1
                    for t, v in zip(val.confidence_mins, val.values):
                        if t is None:
                            assert v is None
                            continue
                        assert pd.to_timedelta(t) >= timedelta(
                            0,
                        ), "Metric: %s\nConfidence mins: %s" % (m, val.confidence_mins)
                if val.confidence_maxs is not None:
                    cmaxs += 1
                    for t, v in zip(val.confidence_maxs, val.values):
                        if t is None:
                            assert v is None
                            continue
                        assert pd.to_timedelta(t) >= timedelta(
                            0,
                        ), "Metric: %s\nConfidence maxs: %s" % (m, val.confidence_maxs)
                if val.confidence_scores is not None:
                    cscores += 1
                    for s, v in zip(val.confidence_scores, val.values):
                        if s is None:
                            assert v is None
                            continue
                        assert 0 <= s <= 100, "Metric: %s\nConfidence scores: %s" % (
                            m,
                            val.confidence_scores,
                        )
            for k, v in nonzero.items():
                assert v > 0, k
        assert cmins > 0
        assert cmaxs > 0
        assert cscores > 0
        assert all((v == 2) for v in gcounts.values())

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    @pytest.mark.parametrize(
        ("devs", "date_from"),
        ([{"with": {}}, "2019-11-28"], [{}, "2018-09-28"]),
    )
    async def test_empty_devs_tight_date(self, devs, date_from):
        """https://athenianco.atlassian.net/browse/ENG-126"""
        repos = ["github.com/src-d/go-git"]
        body = self._body(
            date_from=date_from,
            date_to="2020-01-16",
            for_=[{**devs, "repositories": repos, "environments": ["production"]}],
            granularities=["month"],
            metrics=list(PullRequestMetricID),
        )
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert len(cm.calculated[0].values) > 0

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_reposet(self):
        """Substitute {id} with the real repos."""
        body = self._body(
            for_=[
                {
                    "with": {"author": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                    "repositories": ["{1}"],
                },
            ],
            metrics=[PullRequestMetricID.PR_LEAD_TIME],
            date_from="2015-10-13",
            date_to="2020-01-23",
        )
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert len(cm.calculated[0].values) > 0
        assert cm.calculated[0].for_.repositories == ["{1}"]

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    @pytest.mark.parametrize(
        "metric, count",
        [
            (PullRequestMetricID.PR_WIP_COUNT, 590),
            (PullRequestMetricID.PR_REVIEW_COUNT, 428),
            (PullRequestMetricID.PR_MERGING_COUNT, 532),
            (PullRequestMetricID.PR_RELEASE_COUNT, 407),
            (PullRequestMetricID.PR_OPEN_COUNT, 583),
            (PullRequestMetricID.PR_LEAD_COUNT, 407),
            (PullRequestMetricID.PR_LIVE_CYCLE_COUNT, 904),
            (PullRequestMetricID.PR_OPENED, 590),
            (PullRequestMetricID.PR_REVIEWED, 368),
            (PullRequestMetricID.PR_NOT_REVIEWED, 275),
            (PullRequestMetricID.PR_CLOSED, 583),
            (PullRequestMetricID.PR_MERGED, 532),
            (PullRequestMetricID.PR_REJECTED, 51),
            (PullRequestMetricID.PR_DONE, 462),
            (PullRequestMetricID.PR_WIP_PENDING_COUNT, 0),
            (PullRequestMetricID.PR_REVIEW_PENDING_COUNT, 86),
            (PullRequestMetricID.PR_MERGING_PENDING_COUNT, 21),
            (PullRequestMetricID.PR_RELEASE_PENDING_COUNT, 4395),
        ],
    )
    async def test_counts_sums(self, metric, count):
        body = self._body(
            for_=[
                {
                    "with": {
                        k: ["github.com/vmarkovtsev", "github.com/mcuadros"]
                        for k in PullRequestWith().attribute_types
                    },
                    "repositories": ["{1}"],
                },
            ],
            metrics=[metric],
            date_from="2015-10-13",
            date_to="2020-01-23",
            granularities=["month"],
        )
        res = await self._request(json=body)
        s = 0
        for item in res["calculated"][0]["values"]:
            assert "confidence_mins" not in item
            assert "confidence_maxs" not in item
            assert "confidence_scores" not in item
            val = item["values"][0]
            if val is not None:
                s += val
        assert s == count

    # fmt: off
    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    @pytest.mark.parametrize(
        "with_bots, values", [
            (
                False,
                [
                    [2.461538553237915, None, None, None],
                    [2.8328359127044678, 2.452173948287964, 7.439024448394775, 8.300812721252441],
                    [3.009493589401245, 2.5473685264587402, 5.605769157409668, 6.509615421295166],
                    [2.930656909942627, 2.5, 6.036144733428955, 6.831325531005859],
                    [2.6624202728271484, 2.53125, 7.846153736114502, 8.269230842590332],
                ],
            ), (
                True,
                [
                    [1.4807692766189575, None, None, None],
                    [1.9402985572814941, 2.2058823108673096, 7.699999809265137, 8.050000190734863],
                    [2.129746913909912, 2.3416149616241455, 5.641025543212891, 6.115384578704834],
                    [2.0547444820404053, 2.2945735454559326, 6.262295246124268, 6.7868852615356445],  # noqa
                    [1.8216561079025269, 2.4556961059570312, 7.925000190734863, 8.0],
                ],
            ),
        ],
    )
    # fmt: on
    async def test_averages(self, with_bots, values, sdb):
        if with_bots:
            await sdb.execute(
                sa.insert(Team).values(
                    Team(owner_id=1, name=Team.BOTS, members=[39789]).create_defaults().explode(),
                ),
            )
        body = self._body(
            for_=[{"with": {}, "repositories": ["{1}"]}],
            metrics=[
                PullRequestMetricID.PR_PARTICIPANTS_PER,
                PullRequestMetricID.PR_REVIEWS_PER,
                PullRequestMetricID.PR_REVIEW_COMMENTS_PER,
                PullRequestMetricID.PR_COMMENTS_PER,
            ],
            date_from="2015-10-13",
            date_to="2020-01-23",
            granularities=["year"],
        )
        res = await self._request(json=body)
        assert [v["values"] for v in res["calculated"][0]["values"]] == values

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_sizes(self):
        body = self._body(
            for_=[{"with": {}, "repositories": ["{1}"]}],
            metrics=[PullRequestMetricID.PR_SIZE, PullRequestMetricID.PR_MEDIAN_SIZE],
            date_from="2015-10-13",
            date_to="2020-01-23",
        )
        rbody = await self._request(json=body)
        values = [v["values"] for v in rbody["calculated"][0]["values"]]
        assert values == [[296, 54]]
        for ts in rbody["calculated"][0]["values"]:
            for v, cmin, cmax in zip(ts["values"], ts["confidence_mins"], ts["confidence_maxs"]):
                assert cmin < v < cmax

        body["quantiles"] = [0, 0.9]
        rbody = await self._request(json=body)
        values = [v["values"] for v in rbody["calculated"][0]["values"]]
        assert values == [[177, 54]]

        body["granularities"].append("month")
        rbody = await self._request(json=body)
        values = [v["values"] for v in rbody["calculated"][0]["values"]]
        assert values == [[177, 54]]

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_index_error(self):
        body = self._body(
            for_=[{"with": {}, "repositories": ["github.com/src-d/go-git"]}],
            metrics=[
                PullRequestMetricID.PR_WIP_TIME,
                PullRequestMetricID.PR_REVIEW_TIME,
                PullRequestMetricID.PR_MERGING_TIME,
                PullRequestMetricID.PR_RELEASE_TIME,
                PullRequestMetricID.PR_LEAD_TIME,
            ],
            date_from="2019-02-25",
            date_to="2019-02-28",
            granularities=["week"],
        )
        await self._request(json=body)

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_ratio_flow(self, headers):
        """https://athenianco.atlassian.net/browse/ENG-411"""
        body = self._body(
            date_from="2016-01-01",
            date_to="2020-01-16",
            for_=[{"repositories": ["github.com/src-d/go-git"]}],
            granularities=["month"],
            metrics=[
                PullRequestMetricID.PR_FLOW_RATIO,
                PullRequestMetricID.PR_OPENED,
                PullRequestMetricID.PR_CLOSED,
            ],
        )
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert len(cm.calculated[0].values) > 0
        for v in cm.calculated[0].values:
            flow, opened, closed = v.values
            if opened is not None:
                assert flow is not None
            else:
                opened = 0
            if flow is None:
                assert closed is None
                continue
            assert flow == np.float32((opened + 1) / (closed + 1)), "%.3f != %d / %d" % (
                flow,
                opened,
                closed,
            )

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_exclude_inactive_full_span(self):
        body = self._body(
            date_from="2017-01-01",
            date_to="2017-01-11",
            for_=[{"repositories": ["github.com/src-d/go-git"]}],
            metrics=[PullRequestMetricID.PR_ALL_COUNT],
            exclude_inactive=True,
        )
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert cm.calculated[0].values[0].values[0] == 6

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_exclude_inactive_split(self):
        body = self._body(
            date_from="2016-12-21",
            date_to="2017-01-11",
            for_=[{"repositories": ["github.com/src-d/go-git"]}],
            granularities=["11 day"],
            metrics=[PullRequestMetricID.PR_ALL_COUNT],
            exclude_inactive=True,
        )
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert cm.calculated[0].values[0].values[0] == 1
        assert cm.calculated[0].values[1].date == date(2017, 1, 1)
        assert cm.calculated[0].values[1].values[0] == 6

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_filter_authors(self):
        body = self._body(
            date_from="2017-01-01",
            date_to="2017-01-11",
            for_=[
                {
                    "repositories": ["github.com/src-d/go-git"],
                    "with": {"author": ["github.com/mcuadros"]},
                },
            ],
            granularities=["all"],
            metrics=[PullRequestMetricID.PR_ALL_COUNT],
        )
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert cm.calculated[0].values[0].values[0] == 1

    @pytest.mark.app_validate_responses(False)
    async def test_filter_team(self, sample_team):
        body = {
            "date_from": "2017-01-01",
            "date_to": "2017-01-11",
            "for": [
                {
                    "repositories": ["github.com/src-d/go-git"],
                    "with": {"author": ["{%d}" % sample_team]},
                },
            ],
            "granularities": ["all"],
            "account": 1,
            "metrics": [PullRequestMetricID.PR_ALL_COUNT],
            "exclude_inactive": False,
        }
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert cm.calculated[0].values[0].values[0] == 4

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_group_authors(self):
        body = {
            "date_from": "2017-01-01",
            "date_to": "2017-04-11",
            "for": [
                {
                    "repositories": ["github.com/src-d/go-git"],
                    "withgroups": [
                        {"author": ["github.com/mcuadros"]},
                        {"merger": ["github.com/mcuadros"]},
                    ],
                },
            ],
            "granularities": ["all"],
            "account": 1,
            "metrics": [PullRequestMetricID.PR_ALL_COUNT],
            "exclude_inactive": False,
        }
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert cm.calculated[0].values[0].values[0] == 13
        assert cm.calculated[0].for_.with_.author == ["github.com/mcuadros"]
        assert not cm.calculated[0].for_.with_.merger
        assert cm.calculated[1].values[0].values[0] == 49
        assert cm.calculated[1].for_.with_.merger == ["github.com/mcuadros"]
        assert not cm.calculated[1].for_.with_.author

    @pytest.mark.app_validate_responses(False)
    async def test_group_team(self, sample_team):
        team_str = "{%d}" % sample_team
        body = {
            "date_from": "2017-01-01",
            "date_to": "2017-04-11",
            "for": [
                {
                    "repositories": ["github.com/src-d/go-git"],
                    "withgroups": [{"author": [team_str]}, {"merger": [team_str]}],
                },
            ],
            "granularities": ["all"],
            "account": 1,
            "metrics": [PullRequestMetricID.PR_ALL_COUNT],
            "exclude_inactive": False,
        }
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert cm.calculated[0].values[0].values[0] == 21
        assert cm.calculated[0].for_.with_.author == [team_str]
        assert not cm.calculated[0].for_.with_.merger
        assert cm.calculated[1].values[0].values[0] == 61
        assert cm.calculated[1].for_.with_.merger == [team_str]
        assert not cm.calculated[1].for_.with_.author

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_labels_include(self):
        body = {
            "date_from": "2018-09-01",
            "date_to": "2018-11-18",
            "for": [
                {
                    "repositories": ["github.com/src-d/go-git"],
                    "labels_include": ["bug", "enhancement"],
                },
            ],
            "granularities": ["all"],
            "account": 1,
            "metrics": [PullRequestMetricID.PR_ALL_COUNT],
            "exclude_inactive": False,
        }
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert cm.calculated[0].values[0].values[0] == 6

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_quantiles(self):
        body = {
            "date_from": "2018-06-01",
            "date_to": "2018-11-18",
            "for": [
                {
                    "repositories": [
                        "github.com/src-d/go-git",
                    ],
                    "labels_include": [
                        "bug",
                        "enhancement",
                    ],
                },
            ],
            "granularities": ["all"],
            "account": 1,
            "metrics": [PullRequestMetricID.PR_WIP_TIME],
            "exclude_inactive": False,
        }
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        wip1 = cm.calculated[0].values[0].values[0]

        body["quantiles"] = [0, 0.5]
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        wip2 = cm.calculated[0].values[0].values[0]
        assert int(wip1[:-1]) < int(wip2[:-1])  # yes, not >, here is why:
        # array([[['NaT', 'NaT', 'NaT'],
        #         ['NaT', 'NaT', 'NaT'],
        #         ['NaT', 'NaT', 'NaT'],
        #         [496338, 'NaT', 'NaT'],
        #         ['NaT', 'NaT', 'NaT'],
        #         [250, 'NaT', 'NaT'],
        #         ['NaT', 'NaT', 'NaT'],
        #         [1191, 0, 293],
        #         [3955, 'NaT', 'NaT'],
        #         ['NaT', 'NaT', 'NaT'],
        #         ['NaT', 'NaT', 'NaT'],
        #         ['NaT', 'NaT', 'NaT'],
        #         ['NaT', 'NaT', 'NaT']]], dtype='timedelta64[s]')
        # We discard 1191 and the overall average becomes bigger.

        body["granularities"] = ["week", "month"]
        await self._request(json=body)

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_jira(self):
        """Metrics over PRs filtered by JIRA properties."""
        body = {
            "for": [
                {
                    "repositories": ["{1}"],
                    "jira": {
                        "epics": ["DEV-149", "DEV-776", "DEV-737", "DEV-667", "DEV-140"],
                        "labels_include": ["performance", "enhancement"],
                        "labels_exclude": ["security"],
                        "issue_types": ["Task"],
                    },
                },
            ],
            "metrics": [PullRequestMetricID.PR_LEAD_TIME],
            "date_from": "2015-10-13",
            "date_to": "2020-01-23",
            "granularities": ["all"],
            "exclude_inactive": False,
            "account": 1,
        }
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert len(cm.calculated[0].values) > 0
        assert cm.calculated[0].values[0].values[0] == "478544s"

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_jira_disabled_projects(self, disabled_dev):
        body = {
            "for": [
                {
                    "repositories": ["{1}"],
                    "jira": {
                        "epics": ["DEV-149", "DEV-776", "DEV-737", "DEV-667", "DEV-140"],
                    },
                },
            ],
            "metrics": [PullRequestMetricID.PR_LEAD_TIME],
            "date_from": "2015-10-13",
            "date_to": "2020-01-23",
            "granularities": ["all"],
            "exclude_inactive": False,
            "account": 1,
        }
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert len(cm.calculated[0].values) > 0
        assert cm.calculated[0].values[0].values[0] is None

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_jira_custom_projects(self):
        body = {
            "for": [
                {
                    "repositories": ["{1}"],
                    "jira": {
                        "epics": ["DEV-149", "DEV-776", "DEV-737", "DEV-667", "DEV-140"],
                        "projects": ["ENG"],
                    },
                },
            ],
            "metrics": [PullRequestMetricID.PR_LEAD_TIME],
            "date_from": "2015-10-13",
            "date_to": "2020-01-23",
            "granularities": ["all"],
            "exclude_inactive": False,
            "account": 1,
        }
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert len(cm.calculated[0].values) > 0
        assert cm.calculated[0].values[0].values[0] is None

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_jira_only_custom_projects(self):
        body = {
            "for": [{"repositories": ["{1}"], "jira": {"projects": ["DEV"]}}],
            "metrics": [PullRequestMetricID.PR_MERGED],
            "date_from": "2015-10-13",
            "date_to": "2020-01-23",
            "granularities": ["all"],
            "exclude_inactive": False,
            "account": 1,
        }
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert len(cm.calculated[0].values) > 0
        assert cm.calculated[0].values[0].values[0] == 45  # > 400 without JIRA projects

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_jira_not_installed(self, sdb: Database):
        await sdb.execute(
            sa.delete(AccountJiraInstallation).where(AccountJiraInstallation.account_id == 1),
        )
        body = {
            "for": [
                {"jiragroups": [{"issue_types": ["Task"]}, {"projects": ["ENG"]}]},
            ],
            "metrics": [PullRequestMetricID.PR_LEAD_TIME],
            "date_from": "2015-10-13",
            "date_to": "2020-01-23",
            "granularities": ["all"],
            "exclude_inactive": False,
            "account": 1,
        }
        res = await self._request(json=body)
        assert len(calcs := res["calculated"]) == 2

        # jiragroups are preserved in response "for" but are ignored when computing the metric
        calc_issue = next(c for c in calcs if c["for"]["jira"] == {"issue_types": ["Task"]})
        calc_projs = next(c for c in calcs if c["for"]["jira"] == {"projects": ["ENG"]})

        assert calc_issue["values"][0]["values"] == calc_projs["values"][0]["values"]

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_groups_smoke(self):
        """Two repository groups."""
        body = self._body(
            for_=[
                {
                    "with": {"author": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                    "repositories": ["{1}", "github.com/src-d/go-git"],
                    "repogroups": [[0], [0]],
                },
            ],
            metrics=[PullRequestMetricID.PR_LEAD_TIME],
            date_from="2017-10-13",
            date_to="2018-01-23",
        )
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert len(cm.calculated) == 2
        assert cm.calculated[0] == cm.calculated[1]
        assert cm.calculated[0].values[0].values[0] == "3667053s"
        assert cm.calculated[0].for_.repositories == ["{1}"]

    @pytest.mark.parametrize("repogroups", [[[0, 0]], [[0, -1]], [[0, 1]]])
    async def test_groups_nasty(self, repogroups):
        """Two repository groups."""
        body = self._body(
            for_=[
                {
                    "with": {"author": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                    "repositories": ["{1}"],
                    "repogroups": repogroups,
                },
            ],
            metrics=[PullRequestMetricID.PR_LEAD_TIME],
            date_from="2017-10-13",
            date_to="2018-01-23",
        )
        await self._request(assert_status=400, json=body)

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_lines_smoke(self):
        """Two repository groups."""
        body = self._body(
            for_=[
                {
                    "with": {"author": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                    "repositories": ["{1}", "github.com/src-d/go-git"],
                    "lines": [50, 200, 100000, 100500],
                },
            ],
            metrics=[PullRequestMetricID.PR_OPENED],
            date_from="2017-10-13",
            date_to="2018-03-23",
        )
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert len(cm.calculated) == 3
        assert cm.calculated[0].values[0].values[0] == 3
        assert cm.calculated[0].for_.lines == [50, 200]
        assert cm.calculated[1].values[0].values[0] == 3
        assert cm.calculated[1].for_.lines == [200, 100000]
        assert cm.calculated[2].values[0].values[0] == 0
        assert cm.calculated[2].for_.lines == [100000, 100500]

        body["for"][0]["lines"] = [50, 100500]
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)
        assert len(cm.calculated) == 1
        assert cm.calculated[0].values[0].values[0] == 6

    @pytest.mark.parametrize(
        "metric",
        [
            PullRequestMetricID.PR_DEPLOYMENT_TIME,
            PullRequestMetricID.PR_DEPLOYMENT_COUNT,
            PullRequestMetricID.PR_DEPLOYMENT_COUNT_Q,
            PullRequestMetricID.PR_CYCLE_DEPLOYMENT_TIME,
            PullRequestMetricID.PR_CYCLE_DEPLOYMENT_COUNT,
            PullRequestMetricID.PR_CYCLE_DEPLOYMENT_COUNT_Q,
            PullRequestMetricID.PR_LEAD_DEPLOYMENT_TIME,
            PullRequestMetricID.PR_LEAD_DEPLOYMENT_COUNT,
            PullRequestMetricID.PR_LEAD_DEPLOYMENT_COUNT_Q,
        ],
    )
    async def test_deployments_no_env(self, metric):
        envs = {"environments": []} if "time" in metric else {}
        body = self._body(
            for_=[{"with": {}, "repositories": ["{1}"], **envs}],
            metrics=[metric],
            date_from="2015-10-13",
            date_to="2020-01-23",
            granularities=["year"],
        )
        await self._request(assert_status=400, json=body)

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_deployments_smoke(self, precomputed_deployments):
        body = self._body(
            for_=[
                {"repositories": ["{1}"], "environments": ["staging", "production"]},
            ],
            metrics=[
                PullRequestMetricID.PR_DEPLOYMENT_TIME,
                PullRequestMetricID.PR_LEAD_DEPLOYMENT_TIME,
                PullRequestMetricID.PR_CYCLE_DEPLOYMENT_TIME,
                PullRequestMetricID.PR_DEPLOYMENT_COUNT,
            ],
            date_from="2015-10-13",
            date_to="2020-01-23",
        )
        res = await self._request(json=body)
        values = [v["values"] for v in res["calculated"][0]["values"]]
        assert values == [
            [[None, "62533037s"], [None, "65158487s"], [None, "66004050s"], [0, 513]],
        ]

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    @pytest.mark.parametrize(
        "second_repo, counts",
        [
            ("/beta", [266, 52, 204, 201]),
            ("", [463, 81, 372, 352]),
        ],
    )
    async def test_logical_smoke(
        self,
        logical_settings_db,
        release_match_setting_tag_logical_db,
        second_repo,
        counts,
    ):
        body = self._body(
            for_=[
                {
                    "repositories": [
                        "github.com/src-d/go-git/alpha",
                        "github.com/src-d/go-git" + second_repo,
                    ],
                },
            ],
            metrics=[
                PullRequestMetricID.PR_MERGED,
                PullRequestMetricID.PR_REJECTED,
                PullRequestMetricID.PR_REVIEW_COUNT,
                PullRequestMetricID.PR_RELEASE_COUNT,
            ],
            date_from="2015-10-13",
            date_to="2020-01-23",
        )
        res = await self._request(json=body)
        values = [v["values"] for v in res["calculated"][0]["values"]]
        assert values == [counts]

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_logical_dupes(self, logical_settings_db, sdb):
        await models_insert(
            sdb,
            ReleaseSettingFactory(
                branches="master", match=ReleaseMatch.tag, repo_id=40550, logical_name="alpha",
            ),
            ReleaseSettingFactory(
                branches="master", match=ReleaseMatch.tag, repo_id=40550, logical_name="beta",
            ),
        )

        body = self._body(
            for_=[
                {
                    "repositories": [
                        "github.com/src-d/go-git/alpha",
                        "github.com/src-d/go-git/beta",
                    ],
                },
            ],
            metrics=[
                PullRequestMetricID.PR_MERGED,
                PullRequestMetricID.PR_REJECTED,
                PullRequestMetricID.PR_REVIEW_COUNT,
                PullRequestMetricID.PR_RELEASE_COUNT,
            ],
            date_from="2015-10-13",
            date_to="2020-01-23",
        )
        res = await self._request(json=body)
        values = [v["values"] for v in res["calculated"][0]["values"]]
        assert values == [[250, 49, 194, 189]]

    @with_defer
    async def test_calc_metrics_prs_release_ignored(
        self,
        mdb,
        pdb,
        rdb,
        release_match_setting_tag,
        pr_miner,
        prefixer,
        branches,
        default_branches,
    ):
        body = {
            "for": [{"repositories": ["{1}"]}],
            "metrics": [
                PullRequestMetricID.PR_RELEASE_TIME,
                PullRequestMetricID.PR_RELEASE_COUNT,
                PullRequestMetricID.PR_RELEASE_PENDING_COUNT,
                PullRequestMetricID.PR_REJECTED,
                PullRequestMetricID.PR_DONE,
            ],
            "date_from": "2017-06-01",
            "date_to": "2018-01-01",
            "granularities": ["all"],
            "exclude_inactive": True,
            "account": 1,
        }
        result = await self._request(json=body)
        assert result["calculated"][0]["values"][0]["values"] == ["763080s", 79, 61, 21, 102]
        time_from = datetime(year=2017, month=6, day=1, tzinfo=timezone.utc)
        time_to = datetime(year=2017, month=12, day=31, tzinfo=timezone.utc)
        releases, _, _, _ = await mine_releases(
            ["src-d/go-git"],
            {},
            None,
            default_branches,
            time_from,
            time_to,
            LabelFilter.empty(),
            JIRAFilter.empty(),
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            prefixer,
            1,
            (6366825,),
            mdb,
            pdb,
            rdb,
            None,
            with_deployments=False,
        )
        await wait_deferred()
        ignored = await override_first_releases(
            releases, {}, release_match_setting_tag, 1, pdb, threshold_factor=0,
        )
        assert ignored == 1
        result = await self._request(json=body)
        assert result["calculated"][0]["values"][0]["values"] == ["779385s", 65, 61, 21, 102]

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    async def test_pr_reviewed_ratio(self) -> None:
        body = self._body(
            date_from="2016-01-01",
            date_to="2020-01-16",
            for_=[{"repositories": ["github.com/src-d/go-git"]}],
            granularities=["month"],
            metrics=[
                PullRequestMetricID.PR_REVIEWED_RATIO,
                PullRequestMetricID.PR_REVIEWED,
                PullRequestMetricID.PR_NOT_REVIEWED,
            ],
        )
        res = await self._request(json=body)
        cm = CalculatedPullRequestMetrics.from_dict(res)

        for v in cm.calculated[0].values:
            ratio, reviewed, not_reviewed = v.values
            if reviewed == 0 and not_reviewed == 0:
                assert ratio is None
            elif reviewed == 0:
                assert ratio == 0
            else:
                assert ratio == pytest.approx(reviewed / (reviewed + not_reviewed), rel=0.001)

    async def test_no_repositories(self) -> None:
        body = self._body(
            date_from="2016-01-01",
            date_to="2016-03-01",
            for_=[
                {},
                {"with": {"author": ["github.com/vmarkovtsev", "github.com/mcuadros"]}},
            ],
            granularities=["all"],
            metrics=[PullRequestMetricID.PR_SIZE],
        )
        res = await self._request(json=body)
        calculated = sorted(res["calculated"], key=lambda c: "with" in c["for"])

        assert calculated[0]["for"] == {}
        assert calculated[0]["values"][0]["values"] == [179]
        assert calculated[1]["for"] == body["for"][1]
        assert calculated[1]["values"][0]["values"] == [386]

    @pytest.mark.app_validate_responses(False)
    async def test_multiple_jiragroups(self, sdb: Database, mdb_rw: Database) -> None:
        jiragroups = [
            {"issue_types": ["task"]},
            {"issue_types": ["bug"]},
            {"issue_types": ["task", "bug"]},
        ]
        body = self._body(
            date_from="2022-02-01",
            date_to="2022-03-01",
            for_=[{"repositories": ["github.com/o/r"], "jiragroups": jiragroups}],
            granularities=["all"],
            metrics=[PullRequestMetricID.PR_ALL_COUNT, PullRequestMetricID.PR_SIZE],
        )
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=99, full_name="o/r")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
            pr_kwargs = {"repository_full_name": "o/r", "created_at": dt(2022, 2, 3)}
            models = [
                *pr_models(99, 11, 1, additions=10, **pr_kwargs),
                *pr_models(99, 12, 2, additions=20, **pr_kwargs),
                *pr_models(99, 13, 3, additions=30, **pr_kwargs),
                *pr_models(99, 14, 4, additions=40, **pr_kwargs),
                md_factory.JIRAProjectFactory(id="1", name="PRJ"),
                md_factory.JIRAIssueTypeFactory(id="t", name="task"),
                md_factory.JIRAIssueTypeFactory(id="b", name="bug"),
                md_factory.JIRAIssueFactory(id="20", project_id="1", type_id="t", type="task"),
                md_factory.JIRAIssueFactory(id="30", project_id="1", type_id="b", type="bug"),
                md_factory.JIRAIssueFactory(id="40", project_id="1", type_id="b", type="task"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=11, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=12, jira_id="30"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=13, jira_id="40"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            res = await self._request(json=body)
            assert len(calcs := res["calculated"]) == 3
            group_0_res = next(c for c in calcs if c["for"]["jira"] == {"issue_types": ["bug"]})
            assert group_0_res["values"][0]["values"][0] == 2
            assert group_0_res["values"][0]["values"][1] == 25

            group_1_res = next(c for c in calcs if c["for"]["jira"] == {"issue_types": ["task"]})
            assert group_1_res["values"][0]["values"][0] == 1
            assert group_1_res["values"][0]["values"][1] == 10

            group_2_res = next(
                c for c in calcs if c["for"]["jira"] == {"issue_types": ["task", "bug"]}
            )
            assert group_2_res["values"][0]["values"][0] == 3
            assert group_2_res["values"][0]["values"][1] == 20


class TestCalcMetricsPRsErrors(BaseCalcMetricsPRsTest):
    async def test_access_denied(self):
        """https://athenianco.atlassian.net/browse/ENG-116"""
        body = self._body(
            for_=[
                {
                    "with": {
                        "commit_committer": ["github.com/vmarkovtsev", "github.com/mcuadros"],
                    },
                    "repositories": [
                        "github.com/src-d/go-git",
                        "github.com/athenianco/athenian-api",
                    ],
                },
            ],
            metrics=list(PullRequestMetricID),
            date_from="2015-10-13",
            date_to="2019-03-15",
            granularities=["month"],
        )
        await self._request(json=body, assert_status=403)

    # TODO: fix response validation against the schema
    @pytest.mark.app_validate_responses(False)
    @pytest.mark.parametrize(
        "account, date_to, quantiles, lines, in_, code",
        [
            (3, "2020-02-22", [0, 1], None, "{1}", 404),
            (2, "2020-02-22", [0, 1], None, "{1}", 422),
            (10, "2020-02-22", [0, 1], None, "{1}", 404),
            (1, "2015-10-13", [0, 1], None, "{1}", 200),
            (1, "2010-01-11", [0, 1], None, "{1}", 400),
            (1, "2020-01-32", [0, 1], None, "{1}", 400),
            (1, "2020-01-01", [-1, 0.5], None, "{1}", 400),
            (1, "2020-01-01", [0, -1], None, "{1}", 400),
            (1, "2020-01-01", [10, 20], None, "{1}", 400),
            (1, "2020-01-01", [0.5, 0.25], None, "{1}", 400),
            (1, "2020-01-01", [0.5, 0.5], None, "{1}", 400),
            (1, "2015-10-13", [0, 1], [], "{1}", 400),
            (1, "2015-10-13", [0, 1], [1], "{1}", 400),
            (1, "2015-10-13", [0, 1], [1, 1], "{1}", 400),
            (1, "2015-10-13", [0, 1], [-1, 1], "{1}", 400),
            (1, "2015-10-13", [0, 1], [1, 0], "{1}", 400),
            (1, "2015-10-13", [0, 1], None, "github.com/athenianco/api", 403),
            ("1", "2020-02-22", [0, 1], None, "{1}", 400),
            (1, "0015-10-13", [0, 1], None, "{1}", 400),
        ],
    )
    async def test_nasty_input(self, account, date_to, quantiles, lines, in_, code, mdb):
        """What if we specify a date that does not exist?"""
        body = self._body(
            for_=[
                {
                    "with": {"merger": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                    "repositories": [in_],
                    **({"lines": lines} if lines is not None else {}),
                },
                {
                    "with": {"releaser": ["github.com/vmarkovtsev", "github.com/mcuadros"]},
                    "repositories": ["github.com/src-d/go-git"],
                },
            ],
            metrics=[PullRequestMetricID.PR_LEAD_TIME],
            date_from="2015-10-13" if date_to != "0015-10-13" else "0015-10-13",
            date_to=date_to if date_to != "0015-10-13" else "2015-10-13",
            granularities=["week"],
            quantiles=quantiles,
            account=account,
        )
        await self._request(assert_status=code, json=body)

    async def test_repogroups_with_no_repositories(self) -> None:
        body = self._body(
            for_=[{"repogroups": [[0]]}],
            metrics=[PullRequestMetricID.PR_LEAD_TIME],
            date_from="2017-10-13",
            date_to="2018-01-23",
        )
        await self._request(assert_status=400, json=body)

    @pytest.mark.app_validate_responses(False)
    async def test_both_jira_and_jiragroups(self, sdb: Database) -> None:
        body = self._body(
            date_from="2022-02-01",
            date_to="2022-03-01",
            for_=[
                {
                    "jiragroups": [{"issue_types": ["task"]}, {"issue_types": ["bug"]}],
                    "jira": {"issue_types": ["task"]},
                },
            ],
            granularities=["all"],
            metrics=[PullRequestMetricID.PR_ALL_COUNT, PullRequestMetricID.PR_SIZE],
        )

        res = await self._request(assert_status=400, json=body)
        assert "jiragroups" in res["detail"]
        assert "jira" in res["detail"]
