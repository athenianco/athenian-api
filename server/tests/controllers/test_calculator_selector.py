import sys

import pytest
from sqlalchemy import insert

from athenian.api.controllers.calculator_selector import get_calculator_for_user, \
    METRIC_ENTRIES_VARIATIONS_PREFIX
from athenian.api.controllers.features.entries import \
    MetricEntriesCalculator as OriginalMetricEntriesCalculator
from athenian.api.models.state.models import AccountFeature, Feature, \
    FeatureComponent, God


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

    calc = await get_calculator_for_user(
        "github", 1, "1", sdb, mdb, pdb, rdb, cache, base_module=base_testing_module,
    )
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

    calc = await get_calculator_for_user(
        "github", 1, "1", sdb, mdb, pdb, rdb, cache, base_module=base_testing_module,
    )
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

    calc = await get_calculator_for_user(
        "github", 1, "1", sdb, mdb, pdb, rdb, cache, base_module=base_testing_module,
    )
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

    calc = await get_calculator_for_user(
        "github", 1, "1", sdb, mdb, pdb, rdb, cache, base_module=base_testing_module,
    )
    assert isinstance(calc, MetricEntriesCalculator)


@pytest.mark.parametrize("is_god", (True, False))
async def test_get_calculator_for_user_with_feature_god_only(
        sdb, mdb, pdb, rdb, cache, is_god, base_testing_module, current_module,
):
    if is_god:
        await sdb.execute(
            insert(God).values(
                God(user_id="1").create_defaults().explode(with_primary_keys=True),
            ),
        )

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

    calc = await get_calculator_for_user(
        "github", 1, "1", sdb, mdb, pdb, rdb, cache, base_module=base_testing_module,
    )

    expected_cls = MetricEntriesCalculator if is_god else OriginalMetricEntriesCalculator
    assert isinstance(calc, expected_cls)
