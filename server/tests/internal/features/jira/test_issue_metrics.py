from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from athenian.api.internal.features.jira.issue_metrics import (
    LeadTimeBelowThresholdRatio,
    LeadTimeCalculator,
)
from athenian.api.internal.miners.jira.issue import ISSUE_PRS_BEGAN, ISSUE_PRS_RELEASED
from athenian.api.models.metadata.jira import AthenianIssue, Issue
from tests.testutils.time import dt, dt64arr_ns


class TestLeadTimeBelowThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_ns(dt(2022, 1, 1))
        max_times = dt64arr_ns(dt(2022, 7, 1))
        issues = [
            [dt(2022, 1, 3, 1), dt(2022, 1, 3, 1, 5)],
            [dt(2022, 1, 3, 22), dt(2022, 1, 4, 2)],
            [dt(2022, 1, 4), dt(2022, 1, 7)],
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
            lead_time_calc, quantiles=quantiles, threshold=timedelta(days=5),
        )
        calc(facts, min_times, max_times, None, groups_mask)
        assert calc.values[0][0].value == 1

    @classmethod
    def _gen_facts(cls, *values: list) -> pd.DataFrame:
        filled_values = [[v[0], v[0], v[1], v[1]] for v in values]
        return pd.DataFrame.from_records(
            filled_values,
            columns=[
                AthenianIssue.work_began.name,
                ISSUE_PRS_BEGAN,
                Issue.resolved.name,
                ISSUE_PRS_RELEASED,
            ],
        )
