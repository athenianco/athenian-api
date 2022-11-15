import numpy as np
from numpy.testing import assert_array_equal

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
