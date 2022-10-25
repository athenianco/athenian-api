from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from athenian.api.internal.features.github.pull_request_metrics import (
    ClosedCalculator,
    NotReviewedCalculator,
    ReviewedCalculator,
    ReviewedRatioCalculator,
    ReviewTimeBelowThresholdRatio,
    ReviewTimeCalculator,
    _ReviewedPlusNotReviewedCalculator,
    group_prs_by_participants,
)
from athenian.api.internal.miners.types import PRParticipationKind, PullRequestFacts
from athenian.api.typing_utils import df_from_structs
from tests.conftest import generate_pr_samples
from tests.controllers.features.github.test_pull_request_metrics import dt64arr_ns
from tests.testutils.factory.miners import PullRequestFactsFactory
from tests.testutils.time import dt


class TestGroupPRsByParticipants:
    def test_multiple_groups(self) -> None:
        items = pd.DataFrame({"author": [20, 30], "values": [100, 200]})

        participants = [
            {PRParticipationKind.AUTHOR: {10, 30}},
            {PRParticipationKind.AUTHOR: {20}},
        ]

        res = group_prs_by_participants(participants, items)
        # group {10, 30} has row 1, group {20} has row 0
        assert len(res) == 2
        assert np.array_equal(res[0], [1])
        assert np.array_equal(res[1], [0])

    def test_single_participant_groups(self) -> None:
        items = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        participants = [{PRParticipationKind.AUTHOR: {2}}]

        res = group_prs_by_participants(participants, items)
        # all rows are selected with a single group
        assert len(res) == 1
        assert np.array_equal(res[0], [0, 1])

    def test_no_participant_groups(self) -> None:
        items = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        res = group_prs_by_participants([], items)
        # all rows are selected with no groups
        assert len(res) == 1
        assert np.array_equal(res[0], [0, 1])

    def test_empty_items_multiple_participant_groups(self) -> None:
        items = pd.DataFrame()
        participants = [
            {PRParticipationKind.AUTHOR: {1, 3}},
            {PRParticipationKind.AUTHOR: {2}},
        ]

        res = group_prs_by_participants(participants, items)
        assert len(res) == 2
        assert np.array_equal(res[0], [])
        assert np.array_equal(res[1], [])


class TestReviewedCalculator:
    def test_base(self) -> None:
        prs = df_from_structs(generate_pr_samples(100))

        calc = ReviewedCalculator(quantiles=[0, 1])

        min_times = dt64arr_ns(dt(2001, 1, 1))
        max_times = dt64arr_ns(dt(2135, 1, 1))

        calc(prs, min_times, max_times, None, np.full((1, len(prs)), True))

        assert calc.values[0][0].value is not None
        assert 0 <= calc.values[0][0].value <= 100


class TestReviewedRatioCalculator:
    def test_base(self) -> None:
        prs = df_from_structs(generate_pr_samples(30))

        reviewed_calc = ReviewedCalculator(quantiles=(0, 1))
        closed_calc = ClosedCalculator(quantiles=(0, 1))
        not_reviewed_calc = NotReviewedCalculator(reviewed_calc, closed_calc, quantiles=(0, 1))
        rev_non_rev_calc = _ReviewedPlusNotReviewedCalculator(
            reviewed_calc, not_reviewed_calc, quantiles=(0, 1),
        )
        reviewed_ratio_calc = ReviewedRatioCalculator(
            reviewed_calc, rev_non_rev_calc, quantiles=(0, 1),
        )

        min_times = dt64arr_ns(dt(2001, 1, 1))
        max_times = dt64arr_ns(dt(2135, 1, 1))
        calc_args = (prs, min_times, max_times, None, np.full((1, len(prs)), True))

        reviewed_calc(*calc_args)
        closed_calc(*calc_args)
        not_reviewed_calc(*calc_args)
        rev_non_rev_calc(*calc_args)
        reviewed_ratio_calc(*calc_args)

        reviewed = reviewed_calc.values[0][0].value
        not_reviewed = not_reviewed_calc.values[0][0].value
        reviewed_ratio = reviewed_ratio_calc.values[0][0].value

        assert reviewed is not None
        assert not_reviewed is not None
        assert reviewed_ratio == pytest.approx(reviewed / (reviewed + not_reviewed), rel=0.001)


class TestReviewTimeCalculator:
    def test_base(self) -> None:
        calc = ReviewTimeCalculator(quantiles=(0, 1))
        min_times = dt64arr_ns(dt(2022, 1, 1))
        max_times = dt64arr_ns(dt(2022, 3, 1))

        prs = [
            PullRequestFactsFactory(
                first_review_request_exact=pd.Timestamp(dt(2022, 1, 1)),
                approved=pd.Timestamp(dt(2022, 1, 4)),
            ),
            PullRequestFactsFactory(
                first_review_request_exact=pd.Timestamp(dt(2022, 1, 1)),
                approved=pd.Timestamp(dt(2022, 1, 2)),
            ),
        ]
        facts = df_from_structs(prs)

        calc(facts, min_times, max_times, None, np.full((1, len(prs)), True, bool))
        assert calc.values[0][0].value == timedelta(days=2)


class TestReviewTimeBelowThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        threshold = timedelta(hours=5)
        min_times = dt64arr_ns(dt(2022, 1, 1))
        max_times = dt64arr_ns(dt(2022, 4, 1))

        review_time_calc = ReviewTimeCalculator(quantiles=quantiles)
        calc = ReviewTimeBelowThresholdRatio(
            review_time_calc, quantiles=quantiles, threshold=threshold,
        )

        prs = [
            self._mk_pr(dt(2022, 1, 1, 5), dt(2022, 1, 1, 10)),
            self._mk_pr(dt(2022, 1, 1, 2), dt(2022, 1, 1, 8)),
            self._mk_pr(dt(2022, 1, 1, 12), dt(2022, 1, 1, 14)),
            self._mk_pr(dt(2022, 1, 1, 12), None),
        ]
        facts = df_from_structs(prs)

        groups_mask = np.full((1, len(prs)), True, bool)
        review_time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == pytest.approx(2 / 3)

    def test_complex_groups_mask(self) -> None:
        quantiles = (0, 1)
        threshold = timedelta(hours=3)
        min_times = dt64arr_ns(dt(2022, 1, 1))
        max_times = dt64arr_ns(dt(2022, 4, 1))

        review_time_calc = ReviewTimeCalculator(quantiles=quantiles)
        calc = ReviewTimeBelowThresholdRatio(
            review_time_calc, quantiles=quantiles, threshold=threshold,
        )

        prs = [
            self._mk_pr(dt(2022, 1, 1, 5), dt(2022, 1, 1, 10)),
            self._mk_pr(dt(2022, 1, 1, 2), dt(2022, 1, 1, 5)),
            self._mk_pr(dt(2022, 1, 1, 12), dt(2022, 1, 1, 14)),
            self._mk_pr(dt(2022, 1, 1, 3), dt(2022, 1, 1, 4)),
        ]
        facts = df_from_structs(prs)

        groups_mask = np.array(
            [[True, True, False, True], [True, False, False, True], [True, False, False, False]],
            dtype=bool,
        )

        review_time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 3
        assert all(len(v) == 1 for v in calc.values)

        assert calc.values[0][0].value == pytest.approx(2 / 3)
        assert calc.values[1][0].value == pytest.approx(1 / 2)
        assert calc.values[2][0].value == pytest.approx(0)

    @classmethod
    def _mk_pr(cls, review_request: datetime, approved: Optional[datetime]) -> PullRequestFacts:
        return PullRequestFactsFactory(
            first_review_request_exact=pd.Timestamp(review_request),
            approved=pd.Timestamp(approved) if approved else None,
        )
