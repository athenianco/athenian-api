import sys

from morcilla import Database
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest

from athenian.api.defer import with_defer
from athenian.api.internal.features.entries import (
    MetricEntriesCalculator as OriginalMetricEntriesCalculator,
    MetricEntriesCalculator as TrueMetricEntriesCalculator,
    make_calculator,
)
from athenian.api.internal.features.metric_calculator import DEFAULT_QUANTILE_STRIDE, group_by_repo
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings
from tests.testutils.time import dt


@pytest.fixture
def current_module():
    return sys.modules[__name__].__name__


@pytest.fixture
def base_testing_module(current_module):
    return current_module[: current_module.rfind(".")]


class MetricEntriesCalculator:
    """Fake calculator for different metrics."""

    def __init__(self, *args):
        """Create a `MetricEntriesCalculator`."""
        pass

    def is_ready_for(self, account, meta_ids) -> bool:
        """Check whether the calculator is ready for the given account and meta ids."""
        return account == 1


async def test_get_calculator(base_testing_module, mdb, pdb, rdb, cache):
    calc = make_calculator(1, (1,), mdb, pdb, rdb, cache)
    assert isinstance(calc, OriginalMetricEntriesCalculator)


def test_group_by_repo_single_repos():
    df = pd.DataFrame(
        {
            "repo": [
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
            ],
        },
    )
    groups = group_by_repo("repo", [["one"], ["two"], ["one", "two"]], df)
    assert len(groups) == 3
    assert_array_equal(groups[0], [0, 2, 3, 4])
    assert_array_equal(groups[1], [1, 5])
    assert_array_equal(groups[2], np.arange(len(df)))


def test_group_by_repo_few_groups():
    df = pd.DataFrame(
        {
            "repo": [
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
            ],
        },
    )
    groups = group_by_repo("repo", [["one"]], df)
    assert len(groups) == 1
    assert_array_equal(groups[0], [0, 2, 3, 4])


class TestCalcPullRequestFactsGithub:
    @with_defer
    async def test_gaetano_bug(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
        release_match_setting_tag: ReleaseSettings,
    ) -> None:
        meta_ids = (6366825,)
        calculator = TrueMetricEntriesCalculator(
            1, meta_ids, DEFAULT_QUANTILE_STRIDE, mdb, pdb, rdb, None,
        )
        prefixer = await Prefixer.load(meta_ids, mdb, None)
        repos = release_match_setting_tag.native.keys()
        branches, default_branches = await BranchMiner.extract_branches(
            repos, prefixer, meta_ids, mdb, None,
        )
        base_kwargs = dict(
            repositories={"src-d/go-git"},
            participants={},
            labels=LabelFilter.empty(),
            jira=JIRAFilter.empty(),
            exclude_inactive=False,
            bots=set(),
            release_settings=release_match_setting_tag,
            logical_settings=LogicalRepositorySettings.empty(),
            prefixer=prefixer,
            fresh=False,
            with_jira_map=False,
            branches=branches,
            default_branches=default_branches,
        )
        facts = await calculator.calc_pull_request_facts_github(
            time_from=dt(2017, 8, 10), time_to=dt(2017, 9, 1), **base_kwargs,
        )
        # await wait_deferred()
        last_review = facts[facts.node_id == 163078].last_review.values[0]

        await calculator.calc_pull_request_facts_github(
            time_from=dt(2017, 8, 10), time_to=dt(2017, 8, 20), **base_kwargs,
        )
        facts = await calculator.calc_pull_request_facts_github(
            time_from=dt(2017, 8, 10), time_to=dt(2017, 9, 1), **base_kwargs,
        )
        assert facts[facts.node_id == 163078].last_review.values[0] == last_review

        facts = await calculator.calc_pull_request_facts_github(
            time_from=dt(2017, 8, 10), time_to=dt(2017, 9, 1), **base_kwargs,
        )
        assert facts[facts.node_id == 163078].last_review.values[0] == last_review
