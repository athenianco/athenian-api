from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from athenian.api.internal.features.github.pull_request_metrics import (
    AllCounter,
    AverageReviewCommentsCalculator,
    ClosedCalculator,
    NotReviewedCalculator,
    OpenTimeBelowThresholdRatio,
    OpenTimeCalculator,
    ReviewCommentsAboveThresholdRatio,
    ReviewedCalculator,
    ReviewedRatioCalculator,
    ReviewTimeBelowThresholdRatio,
    ReviewTimeCalculator,
    SizeBelowThresholdRatio,
    SizeCalculator,
    WaitFirstReviewTimeBelowThresholdRatio,
    WaitFirstReviewTimeCalculator,
    _ReviewedPlusNotReviewedCalculator,
    group_prs_by_participants,
)
from athenian.api.internal.miners.types import PRParticipationKind, PullRequestFacts
from athenian.api.typing_utils import df_from_structs
from tests.conftest import generate_pr_samples
from tests.testutils.factory.miners import PullRequestFactsFactory
from tests.testutils.time import dt, dt64arr_ns


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

    def test_empty_groups_in_the_middle(self) -> None:
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
            self._mk_pr(dt(2022, 1, 1, 3), dt(2022, 1, 1, 4)),
        ]
        facts = df_from_structs(prs)

        groups_mask = np.array(
            [
                [True, True, False],
                [True, False, False],
                [False, False, False],
                [False, True, True],
            ],
            dtype=bool,
        )

        review_time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 4
        assert all(len(v) == 1 for v in calc.values)

        assert calc.values[0][0].value == pytest.approx(1 / 2)
        assert calc.values[1][0].value == pytest.approx(0)
        assert calc.values[2][0].value == pytest.approx(0)
        assert calc.values[3][0].value == pytest.approx(1)

    @classmethod
    def _mk_pr(cls, review_request: datetime, approved: Optional[datetime]) -> PullRequestFacts:
        return PullRequestFactsFactory(
            first_review_request_exact=pd.Timestamp(review_request),
            approved=pd.Timestamp(approved) if approved else None,
        )


class TestWaitFirstReviewTimeBelowThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_ns(dt(2022, 1, 1))
        max_times = dt64arr_ns(dt(2022, 2, 1))

        wait_review_calc = WaitFirstReviewTimeCalculator(quantiles=quantiles)
        calc = WaitFirstReviewTimeBelowThresholdRatio(
            wait_review_calc, quantiles=quantiles, threshold=timedelta(hours=24),
        )

        prs = [
            self._mk_pr(dt(2022, 1, 1, 2), dt(2022, 1, 1, 4)),
            self._mk_pr(dt(2022, 1, 2), dt(2022, 1, 3)),
            self._mk_pr(dt(2022, 1, 2, 2), dt(2022, 1, 3, 1)),
            self._mk_pr(dt(2022, 1, 3, 1), dt(2022, 1, 4, 5)),
            self._mk_pr(dt(2022, 1, 3, 1), None),
            self._mk_pr(None, dt(2022, 1, 3, 1)),
        ]
        facts = df_from_structs(prs)
        groups_mask = np.full((1, len(prs)), True, bool)

        wait_review_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == pytest.approx(3 / 4)

        calc = WaitFirstReviewTimeBelowThresholdRatio(
            wait_review_calc, quantiles=quantiles, threshold=timedelta(hours=48),
        )
        calc(facts, min_times, max_times, None, groups_mask)
        assert calc.values[0][0].value == 1

        calc = WaitFirstReviewTimeBelowThresholdRatio(
            wait_review_calc, quantiles=quantiles, threshold=timedelta(hours=12),
        )
        calc(facts, min_times, max_times, None, groups_mask)
        assert calc.values[0][0].value == pytest.approx(1 / 4)

    @classmethod
    def _mk_pr(
        cls,
        review_request: Optional[datetime],
        first_comment: Optional[datetime],
    ) -> PullRequestFacts:
        return PullRequestFactsFactory(
            first_review_request_exact=review_request,
            first_comment_on_first_review=pd.Timestamp(first_comment) if first_comment else None,
        )


class TestSizeBelowThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_ns(dt(2022, 1, 1))
        max_times = dt64arr_ns(dt(2022, 7, 1))
        prs = [
            self._mk_pr(50),
            self._mk_pr(100),
            self._mk_pr(120),
            self._mk_pr(101),
            self._mk_pr(1, created=dt(2023, 1, 1)),  # out of interval
        ]
        facts = df_from_structs(prs)

        groups_mask = np.full((1, len(prs)), True, bool)

        size_calc = SizeCalculator(quantiles=quantiles)
        calc = SizeBelowThresholdRatio(size_calc, quantiles=quantiles)  # default threshold is 100

        size_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == pytest.approx(2 / 4)

    _DEFAULT_DT = dt(2022, 1, 15)

    @classmethod
    def _mk_pr(cls, size: int, created: datetime = _DEFAULT_DT) -> PullRequestFacts:
        return PullRequestFactsFactory(size=size, created=pd.Timestamp(created))


class TestReviewCommentsAboveThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_ns(dt(2022, 1, 1))
        max_times = dt64arr_ns(dt(2022, 7, 1))
        prs = [
            self._mk_pr(0),  # ignored, not reviewed
            self._mk_pr(1),
            self._mk_pr(2),
            self._mk_pr(3),
            self._mk_pr(4),
            self._mk_pr(1, created=dt(2022, 8, 1)),  # ignored, out of time
            self._mk_pr(5, created=dt(2022, 8, 1)),  # ignored, out of time
        ]
        groups_mask = np.full((1, len(prs)), True, bool)
        facts = df_from_structs(prs)

        all_calc = AllCounter(quantiles=quantiles)
        review_comments_calc = AverageReviewCommentsCalculator(all_calc, quantiles=quantiles)
        calc = ReviewCommentsAboveThresholdRatio(review_comments_calc, quantiles=quantiles)

        all_calc(facts, min_times, max_times, None, groups_mask)
        review_comments_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == pytest.approx(2 / 4)

        calc = ReviewCommentsAboveThresholdRatio(
            review_comments_calc, quantiles=quantiles, threshold=4,
        )
        calc(facts, min_times, max_times, None, groups_mask)
        assert calc.values[0][0].value == pytest.approx(1 / 4)

    _DEFAULT_CREATED = dt(2022, 2, 1)

    @classmethod
    def _mk_pr(cls, review_comments: int, created=_DEFAULT_CREATED) -> PullRequestFacts:
        return PullRequestFactsFactory(review_comments=review_comments, created=created)


class TestOpenTimeBelowThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_ns(dt(2022, 1, 1))
        max_times = dt64arr_ns(dt(2022, 7, 1))
        prs = [
            self._mk_pr(dt(2022, 1, 3), dt(2022, 1, 5)),
            self._mk_pr(dt(2022, 1, 3), dt(2022, 1, 8)),
            self._mk_pr(dt(2022, 1, 3), dt(2022, 1, 5)),
            self._mk_pr(dt(2022, 1, 3), None),
        ]
        groups_mask = np.full((1, len(prs)), True, bool)
        facts = df_from_structs(prs)

        open_time_calc = OpenTimeCalculator(quantiles=quantiles)
        calc = OpenTimeBelowThresholdRatio(open_time_calc, quantiles=quantiles)

        open_time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == pytest.approx(2 / 3)

        calc = OpenTimeBelowThresholdRatio(
            open_time_calc, quantiles=quantiles, threshold=timedelta(days=10),
        )
        calc(facts, min_times, max_times, None, groups_mask)
        assert calc.values[0][0].value == 1

    def test_more_groups(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_ns(dt(2022, 1, 1))
        max_times = dt64arr_ns(dt(2022, 7, 1))
        prs = [
            self._mk_pr(dt(2022, 1, 3), dt(2022, 1, 5)),
            self._mk_pr(dt(2022, 1, 3), dt(2022, 1, 8)),
            self._mk_pr(dt(2022, 1, 3), dt(2022, 1, 5)),
            self._mk_pr(dt(2022, 1, 3), None),
        ]
        groups_mask = np.array(
            [[True, True, True, False], [True, False, False, True], [False, True, True, True]],
            dtype=bool,
        )

        facts = df_from_structs(prs)

        open_time_calc = OpenTimeCalculator(quantiles=quantiles)
        calc = OpenTimeBelowThresholdRatio(open_time_calc, quantiles=quantiles)

        open_time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 3
        assert calc.values[0][0].value == pytest.approx(2 / 3)
        assert calc.values[1][0].value == pytest.approx(1 / 1)
        assert calc.values[2][0].value == pytest.approx(1 / 2)

    @classmethod
    def _mk_pr(cls, created: datetime, closed: Optional[datetime]) -> PullRequestFacts:
        return PullRequestFactsFactory(
            created=created, closed=None if closed is None else pd.Timestamp(closed),
        )
