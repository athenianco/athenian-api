import sys

import pytest
from sqlalchemy import insert

from athenian.api.controllers.calculator_selector import get_calculator_for_user, \
    METRIC_ENTRIES_VARIATIONS_PREFIX
from athenian.api.models.state.models import AccountFeature, Feature, \
    FeatureComponent, God


@pytest.fixture
def current_module():
    return sys.modules[__name__].__name__


@pytest.fixture
def base_testing_module(current_module):
    return current_module[: current_module.rfind(".")]


def calc_pull_request_metrics_line_github():
    """This is a fake function for testing."""
    return True


async def test_get_calculator_for_user_no_global_feature(sdb, base_testing_module):
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
        "github", "prs_linear", 1, "1", sdb, base_module=base_testing_module,
    )
    expected = "athenian.api.controllers.features.entries:calc_pull_request_metrics_line_github"
    actual = f"{calc.__module__}:{calc.__name__}"
    assert actual == expected


async def test_get_calculator_for_user_disabled_global_feature(sdb, base_testing_module):
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
        "github", "prs_linear", 1, "1", sdb, base_module=base_testing_module,
    )
    expected = "athenian.api.controllers.features.entries:calc_pull_request_metrics_line_github"
    actual = f"{calc.__module__}:{calc.__name__}"
    assert actual == expected


async def test_get_calculator_for_user_no_feature_for_account(sdb, base_testing_module):
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
        "github", "prs_linear", 1, "1", sdb, base_module=base_testing_module,
    )
    expected = "athenian.api.controllers.features.entries:calc_pull_request_metrics_line_github"
    actual = f"{calc.__module__}:{calc.__name__}"
    assert actual == expected


async def test_get_calculator_for_user_with_feature(
    sdb, base_testing_module, current_module,
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
        "github", "prs_linear", 1, "1", sdb, base_module=base_testing_module,
    )
    expected = f"{current_module}:calc_pull_request_metrics_line_github"
    actual = f"{calc.__module__}:{calc.__name__}"
    assert actual == expected
    assert calc()


@pytest.mark.parametrize("is_god", (True, False))
async def test_get_calculator_for_user_with_feature_god_only(
    sdb, is_god, base_testing_module, current_module,
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
        "github", "prs_linear", 1, "1", sdb, base_module=base_testing_module,
    )

    if is_god:
        expected = f"{current_module}:calc_pull_request_metrics_line_github"
        actual = f"{calc.__module__}:{calc.__name__}"
    else:
        expected = (
            "athenian.api.controllers.features.entries:"
            "calc_pull_request_metrics_line_github"
        )
        actual = f"{calc.__module__}:{calc.__name__}"

    assert actual == expected
