from datetime import datetime, timezone

import numpy as np
import pytest

from athenian.api.experiments.indexing.indexer import BaseIndexer


class MockIndexer(BaseIndexer):

    @classmethod
    async def _fetch_data(cls, _):
        mock_data = [
            {
                "timestamp": datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
                "values": ["a"],
            },
            {
                "timestamp": datetime(1970, 1, 1, 0, 45, 0).replace(tzinfo=timezone.utc),
                "values": ["b"],
            },
            {
                "timestamp": datetime(1970, 1, 1, 1, 15, 0).replace(tzinfo=timezone.utc),
                "values": ["c"],
            },
            {
                "timestamp": datetime(1970, 1, 1, 3, 0, 0).replace(tzinfo=timezone.utc),
                "values": ["d"],
            },
            {
                "timestamp": datetime(1970, 1, 1, 3, 15, 0).replace(tzinfo=timezone.utc),
                "values": ["e"],
            },
            {
                "timestamp": datetime(1970, 1, 1, 3, 30, 0).replace(tzinfo=timezone.utc),
                "values": ["f"],
            },
            {
                "timestamp": datetime(1970, 1, 1, 3, 45, 0).replace(tzinfo=timezone.utc),
                "values": ["g"],
            },
            {
                "timestamp": datetime(1970, 1, 1, 4, 0, 0).replace(tzinfo=timezone.utc),
                "values": ["h"],
            },
            {
                "timestamp": datetime(1970, 1, 1, 4, 45, 0).replace(tzinfo=timezone.utc),
                "values": ["i"],
            },
        ]

        return mock_data


@pytest.mark.parametrize("date_from, date_to, expected", [
    (
        datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        datetime(1970, 1, 1, 5, 0, 0).replace(tzinfo=timezone.utc),
        np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i"]),
    ),
    (
        datetime(1969, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        np.array([]),
    ),
    (
        datetime(1971, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        datetime(1972, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        np.array([]),
    ),
    (
        datetime(1970, 1, 1, 0, 10, 0).replace(tzinfo=timezone.utc),
        datetime(1970, 1, 1, 3, 40, 0).replace(tzinfo=timezone.utc),
        np.array(["b", "c", "d", "e", "f"]),
    ),
    (
        datetime(1970, 1, 1, 0, 40, 0).replace(tzinfo=timezone.utc),
        datetime(1970, 1, 1, 0, 50, 0).replace(tzinfo=timezone.utc),
        np.array(["b"]),
    ),
    (
        datetime(1970, 1, 1, 0, 0, 0).replace(tzinfo=timezone.utc),
        datetime(1970, 1, 1, 4, 45, 0).replace(tzinfo=timezone.utc),
        np.array(["a", "b", "c", "d", "e", "f", "g", "h"]),
    ),
])
async def test_mocked_simple_search(date_from, date_to, expected):
    indexer = await MockIndexer.create(None)
    actual = indexer.search(date_from, date_to)
    np.testing.assert_array_equal(actual, expected)
