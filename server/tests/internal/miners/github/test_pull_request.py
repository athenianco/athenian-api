import contextlib
from unittest import mock

import numpy as np
from numpy.testing import assert_array_equal

from athenian.api.async_utils import read_sql_query
from athenian.api.db import Database
from athenian.api.internal.miners.github.pull_request import fetch_prs_numbers
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory


class TestFetchPRsNumbers:
    async def test_single_meta_acc_id(self, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestFactory(acc_id=3, node_id=30, number=100),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=31, number=101),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=32, number=102),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            prs_numbers = await fetch_prs_numbers(np.array([30, 31, 32, 33]), (3,), mdb_rw)
            assert_array_equal(prs_numbers, np.array([100, 101, 102, 0]))

            prs_numbers = await fetch_prs_numbers(np.array([30, 33, 31]), (3,), mdb_rw)
            assert_array_equal(prs_numbers, np.array([100, 0, 101]))

            prs_numbers = await fetch_prs_numbers(np.array([33, 35, 30]), (3,), mdb_rw)
            assert_array_equal(prs_numbers, np.array([0, 0, 100]))

    async def test_multiple_meta_acc_ids(self, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestFactory(acc_id=3, node_id=30, number=1),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=31, number=2),
                md_factory.NodePullRequestFactory(acc_id=4, node_id=32, number=1),
                md_factory.NodePullRequestFactory(acc_id=4, node_id=33, number=3),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            prs_numbers = await fetch_prs_numbers(np.array([30, 31, 32, 33]), (3, 4), mdb_rw)
            assert_array_equal(prs_numbers, np.array([1, 2, 1, 3]))

            prs_numbers = await fetch_prs_numbers(np.array([30, 31, 32, 35]), (3, 4), mdb_rw)
            assert_array_equal(prs_numbers, np.array([1, 2, 1, 0]))

    async def test_a_whole_lot_of_node_ids(self, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                *[
                    md_factory.NodePullRequestFactory(acc_id=3, node_id=n, number=100 + n)
                    for n in range(1, 106)
                ],
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            prs_numbers = await fetch_prs_numbers(np.array(list(range(1, 111))), (3, 4), mdb_rw)
            assert_array_equal(prs_numbers[:105], np.arange(1, 106) + 100)
            assert_array_equal(prs_numbers[105:], np.zeros(5))

    async def test_unsorted_pr_from_db(self, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestFactory(acc_id=3, node_id=20, number=3),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=21, number=5),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=22, number=1),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=23, number=4),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=24, number=2),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=25, number=6),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            # DB will often return rows ordered by node_id anyway, mock is needed to have
            # a true chaotic order
            with self._shuffle_read_sql_query_result():
                prs_numbers = await fetch_prs_numbers(np.array([23, 25, 19, 21, 22]), (3,), mdb_rw)
            assert_array_equal(prs_numbers, np.array([4, 6, 0, 5, 1]))

    async def test_no_pr_found_pr_order_noise_meta_acc_id(self, mdb_rw: Database) -> None:
        prs_numbers = await fetch_prs_numbers(np.array([22, 23]), (3,), mdb_rw)
        assert_array_equal(prs_numbers, np.array([0, 0]))

    @contextlib.contextmanager
    def _shuffle_read_sql_query_result(self):
        mock_path = f"{fetch_prs_numbers.__module__}.read_sql_query"

        async def _read_sql_query(*args, **kwargs):
            res = await read_sql_query(*args, **kwargs)
            return res.sample(frac=1)

        with mock.patch(mock_path, new=_read_sql_query):
            yield
