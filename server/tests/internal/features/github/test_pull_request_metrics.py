import numpy as np
import pandas as pd

from athenian.api.internal.features.github.pull_request_metrics import group_prs_by_participants
from athenian.api.internal.miners.types import PRParticipationKind


class TestGroupPRsByParticipants:
    def test_multiple_groups(self) -> None:
        items = pd.DataFrame({"author": [20, 30], "values": [100, 200]})

        participants = [
            {PRParticipationKind.AUTHOR: {10, 30}}, {PRParticipationKind.AUTHOR: {20}},
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
            {PRParticipationKind.AUTHOR: {1, 3}}, {PRParticipationKind.AUTHOR: {2}},
        ]

        res = group_prs_by_participants(participants, items)
        assert len(res) == 2
        assert np.array_equal(res[0], [])
        assert np.array_equal(res[1], [])
