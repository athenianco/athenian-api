from datetime import datetime, timedelta

import medvedi as md
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from athenian.api.internal.features.jira.issue_metrics import (
    AcknowledgeTimeBelowThresholdRatio,
    AcknowledgeTimeCalculator,
    LeadTimeBelowThresholdRatio,
    LeadTimeCalculator,
    LifeTimeBelowThresholdRatio,
    LifeTimeCalculator,
)
from athenian.api.internal.miners.jira.issue import ISSUE_PRS_BEGAN, ISSUE_PRS_RELEASED
from athenian.api.models.metadata.jira import AthenianIssue, Issue, Status
from tests.testutils.time import dt, dt64arr_us


class TestLeadTimeBelowThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 7, 1))
        issues = [
            [dt(2022, 1, 3, 1), dt(2022, 1, 3, 1, 5)],
            [dt(2022, 1, 3, 22), dt(2022, 1, 4, 2)],
            [dt(2022, 1, 4), dt(2022, 1, 10)],
        ]
        facts = self._gen_facts(*issues)
        groups_mask = np.full((1, len(issues)), True, bool)
        lead_time_calc = LeadTimeCalculator(quantiles=quantiles)
        calc = LeadTimeBelowThresholdRatio(lead_time_calc, quantiles=quantiles)

        lead_time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == pytest.approx(2 / 3)

        calc = LeadTimeBelowThresholdRatio(
            lead_time_calc, quantiles=quantiles, threshold=timedelta(hours=3),
        )
        calc(facts, min_times, max_times, None, groups_mask)
        assert calc.values[0][0].value == pytest.approx(1 / 3)

        calc = LeadTimeBelowThresholdRatio(
            lead_time_calc, quantiles=quantiles, threshold=timedelta(days=10),
        )
        calc(facts, min_times, max_times, None, groups_mask)
        assert calc.values[0][0].value == 1

    @classmethod
    def _gen_facts(cls, *values: list) -> md.DataFrame:
        column_names = [
            AthenianIssue.work_began.name,
            ISSUE_PRS_BEGAN,
            Issue.resolved.name,
            ISSUE_PRS_RELEASED,
        ]
        columns = {k: [] for k in column_names}
        for v in values:
            for i, r in enumerate((v[0], v[0], v[1], v[1])):
                columns[column_names[i]].append(r.replace(tzinfo=None))
        for k, v in columns.items():
            columns[k] = np.array(v, dtype="datetime64[us]")
        return md.DataFrame(columns)


class TestLifeTimeBelowThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 7, 1))
        issues = [
            [dt(2022, 1, 3), dt(2022, 1, 4)],
            [dt(2022, 1, 3), dt(2022, 1, 5)],
            [dt(2022, 1, 4), dt(2022, 1, 10)],
        ]
        facts = self._gen_facts(*issues)
        groups_mask = np.full((1, len(issues)), True, bool)

        life_time_calc = LifeTimeCalculator(quantiles=quantiles)
        life_time_calc(facts, min_times, max_times, None, groups_mask)
        calc = LifeTimeBelowThresholdRatio(
            life_time_calc, quantiles=quantiles, threshold=timedelta(days=1),
        )
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == pytest.approx(1 / 3)

        calc = LifeTimeBelowThresholdRatio(
            life_time_calc, quantiles=quantiles, threshold=timedelta(hours=12),
        )
        calc(facts, min_times, max_times, None, groups_mask)
        assert calc.values[0][0].value == 0

    @classmethod
    def _gen_facts(cls, *values: list) -> md.DataFrame:
        column_names = [
            ISSUE_PRS_BEGAN,
            Issue.created.name,
            Issue.resolved.name,
            ISSUE_PRS_RELEASED,
        ]
        columns = {k: [] for k in column_names}
        for v in values:
            for i, r in enumerate((v[0], v[0], v[1], v[1])):
                columns[column_names[i]].append(r.replace(tzinfo=None))
        for k, v in columns.items():
            columns[k] = np.array(v, dtype="datetime64[us]")
        return md.DataFrame(columns)


class TestAcknowledgeTimeCalculator:
    min_times = dt64arr_us(dt(2022, 1, 1))
    max_times = dt64arr_us(dt(2022, 7, 1))

    def test_base(self) -> None:
        issues = [
            [datetime(2022, 1, 3), datetime(2022, 1, 4)],
            [datetime(2022, 1, 3), datetime(2022, 1, 5)],
            [datetime(2022, 6, 20), datetime(2022, 7, 30)],  # out of focus
        ]
        calc = self._calc(issues)

        assert_array_equal(
            calc.peek[0], np.array([_np_td(1), _np_td(2), np.timedelta64("NaT", "s")]),
        )
        assert calc.values[0][0].value == timedelta(hours=36)

    def test_status(self) -> None:
        issues = [
            [datetime(2022, 1, 3), datetime(2022, 1, 4)],
            [datetime(2022, 1, 3), None, Status.CATEGORY_TODO],
        ]
        calc = self._calc(issues)

        assert_array_equal(calc.peek[0], np.array([_np_td(1), np.timedelta64("NaT", "s")]))
        assert calc.values[0][0].value == timedelta(hours=24)

    def test_work_began_before_creation(self) -> None:
        issues = [
            [datetime(2022, 1, 3), datetime(2022, 1, 4)],
            [datetime(2022, 1, 3), datetime(2022, 1, 1)],
        ]

        calc = self._calc(issues)

        assert_array_equal(calc.peek[0], np.array([_np_td(1), _np_td(0)]))
        assert calc.values[0][0].value == timedelta(hours=12)

    @classmethod
    def _calc(cls, issues: list[list]) -> AcknowledgeTimeCalculator:
        facts = _gen_ack_time_facts(*issues)
        groups_mask = np.full((1, len(facts)), True, bool)
        calc = AcknowledgeTimeCalculator(quantiles=(0, 1))
        calc(facts, cls.min_times, cls.max_times, None, groups_mask)
        return calc


class TestAcknowledgeTimeBelowThresholdRatio:
    def test_base(self) -> None:
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 7, 1))
        issues = [
            [datetime(2022, 1, 3), datetime(2022, 1, 4)],
            [datetime(2022, 1, 3), datetime(2022, 1, 5)],
            [datetime(2022, 1, 4), datetime(2022, 1, 10)],
            [datetime(2022, 1, 3), datetime(2022, 1, 6)],
            [datetime(2022, 1, 3), datetime(2022, 1, 8)],
        ]
        facts = _gen_ack_time_facts(*issues)
        groups_mask = np.full((1, len(issues)), True, bool)

        ack_time_calc = AcknowledgeTimeCalculator(quantiles=(0, 1))
        ack_time_calc(facts, min_times, max_times, None, groups_mask)
        calc = AcknowledgeTimeBelowThresholdRatio(
            ack_time_calc, quantiles=(0, 1), threshold=timedelta(days=4),
        )
        calc(facts, min_times, max_times, None, groups_mask)
        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == pytest.approx(3 / 5)


def _gen_ack_time_facts(*values: list) -> md.DataFrame:
    sanitized_values = [v if len(v) == 3 else [*v, Status.CATEGORY_IN_PROGRESS] for v in values]
    return md.DataFrame(
        {
            Issue.created.name: np.array([v[0] for v in sanitized_values]),
            ISSUE_PRS_BEGAN: np.array([v[1] for v in sanitized_values]),
            AthenianIssue.work_began.name: np.array([v[1] for v in sanitized_values]),
            Status.category_name.name: np.array([v[2] for v in sanitized_values]),
        },
    )


def _np_td(days: int) -> np.timedelta64:
    return np.timedelta64(days * 3600 * 24, "s")
