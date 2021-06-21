import sys

import pytest
from sqlalchemy import insert

from athenian.api.controllers.calculator_selector import get_calculators_for_account, \
    METRIC_ENTRIES_VARIATIONS_PREFIX
from athenian.api.controllers.features.entries import \
    MetricEntriesCalculator as OriginalMetricEntriesCalculator
from athenian.api.models.state.models import AccountFeature, Feature, FeatureComponent


@pytest.fixture
def current_module():
    return sys.modules[__name__].__name__


@pytest.fixture
def base_testing_module(current_module):
    return current_module[: current_module.rfind(".")]


class MetricEntriesCalculator:
    """Fake calculator for different metrics."""

    def __init__(self, *args) -> "MetricEntriesCalculator":
        """Create a `MetricEntriesCalculator`."""
        pass

    def is_ready_for(self, account, meta_ids) -> bool:
        """Check whether the calculator is ready for the given account and meta ids."""
        return account == 1


async def test_get_calculator_for_user_no_global_feature(sdb, mdb, pdb, rdb, cache,
                                                         base_testing_module):
    await sdb.execute(
        insert(Feature).values(
            Feature(
                name="another-feature-flag",
                component=FeatureComponent.server,
                enabled=True,
            )
            .create_defaults()
            .explode(),
        ),
    )

    calc = (await get_calculators_for_account(
        ["github"], 1, (1, ), None, sdb, mdb, pdb, rdb, cache,
        base_module=base_testing_module,
    ))["github"]
    assert isinstance(calc, OriginalMetricEntriesCalculator)


async def test_get_calculator_for_user_disabled_global_feature(sdb, mdb, pdb, rdb, cache,
                                                               base_testing_module):
    await sdb.execute(
        insert(Feature).values(
            Feature(
                name=f"{METRIC_ENTRIES_VARIATIONS_PREFIX['github']}_test_entries",
                component=FeatureComponent.server,
                enabled=False,
            )
            .create_defaults()
            .explode(),
        ),
    )

    calc = (await get_calculators_for_account(
        ["github"], 1, (1, ), None, sdb, mdb, pdb, rdb, cache,
        base_module=base_testing_module,
    ))["github"]
    assert isinstance(calc, OriginalMetricEntriesCalculator)


async def test_get_calculator_for_user_no_feature_for_account(sdb, mdb, pdb, rdb, cache,
                                                              base_testing_module):
    await sdb.execute(
        insert(Feature).values(
            Feature(
                name=f"{METRIC_ENTRIES_VARIATIONS_PREFIX['github']}_test_entries",
                component=FeatureComponent.server,
                enabled=True,
            )
            .create_defaults()
            .explode(),
        ),
    )

    calc = (await get_calculators_for_account(
        ["github"], 1, (1, ), None,
        sdb, mdb, pdb, rdb, cache, base_module=base_testing_module,
    ))["github"]
    assert isinstance(calc, OriginalMetricEntriesCalculator)


async def test_get_calculator_for_user_with_feature(
        sdb, mdb, pdb, rdb, cache, base_testing_module, current_module,
):
    feature_id = await sdb.execute(
        insert(Feature).values(
            Feature(
                name=f"{METRIC_ENTRIES_VARIATIONS_PREFIX['github']}test_calculator_selector",
                component=FeatureComponent.server,
                enabled=True,
            )
            .create_defaults()
            .explode(),
        ),
    )
    await sdb.execute(
        insert(AccountFeature).values(
            AccountFeature(account_id=1, feature_id=feature_id, enabled=True)
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )

    calc = (await get_calculators_for_account(
        ["github"], 1, (1, ), None, sdb, mdb, pdb, rdb, cache,
        base_module=base_testing_module,
    ))["github"]
    assert isinstance(calc, MetricEntriesCalculator)

    calc = (await get_calculators_for_account(
        ["github"], 2, (1, ), None, sdb, mdb, pdb, rdb, cache,
        base_module=base_testing_module,
    ))["github"]
    assert isinstance(calc, OriginalMetricEntriesCalculator)


async def test_get_calculator_for_user_with_feature_multiple_services(
        sdb, mdb, pdb, rdb, cache, base_testing_module, current_module,
):
    feature_id = await sdb.execute(
        insert(Feature).values(
            Feature(
                name=f"{METRIC_ENTRIES_VARIATIONS_PREFIX['github']}test_calculator_selector",
                component=FeatureComponent.server,
                enabled=True,
            )
            .create_defaults()
            .explode(),
        ),
    )
    await sdb.execute(
        insert(AccountFeature).values(
            AccountFeature(account_id=1, feature_id=feature_id, enabled=True)
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )

    calcs = await get_calculators_for_account(
        ["github"], 1, (1, ), None, sdb, mdb, pdb, rdb, cache,
        base_module=base_testing_module,
    )
    assert len(calcs) == 1
    assert isinstance(calcs["github"], MetricEntriesCalculator)

    with pytest.raises(AssertionError):
        await get_calculators_for_account(
            ["github", "bitbucket"], 1, (1,), None, sdb, mdb, pdb, rdb, cache,
            base_module=base_testing_module,
        )


@pytest.mark.parametrize("is_god", (True, False))
async def test_get_calculator_for_user_with_feature_god_only(
        sdb, mdb, pdb, rdb, cache, is_god, base_testing_module, current_module,
):
    feature_id = await sdb.execute(
        insert(Feature).values(
            Feature(
                name=f"{METRIC_ENTRIES_VARIATIONS_PREFIX['github']}test_calculator_selector",
                component=FeatureComponent.server,
                enabled=True,
            )
            .create_defaults()
            .explode(),
        ),
    )
    await sdb.execute(
        insert(AccountFeature).values(
            AccountFeature(
                account_id=1,
                feature_id=feature_id,
                enabled=True,
                parameters={"god_only": True},
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )

    calc = (await get_calculators_for_account(
        ["github"], 1, (1, ), "1" if is_god else None, sdb, mdb, pdb, rdb, cache,
        base_module=base_testing_module,
    ))["github"]

    expected_cls = MetricEntriesCalculator if is_god else OriginalMetricEntriesCalculator
    assert isinstance(calc, expected_cls)
