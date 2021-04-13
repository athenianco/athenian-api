import sys

import pytest
from sqlalchemy import insert

from athenian.api.controllers.features.entries import \
    get_calculator, get_calculator_for_user, METRIC_ENTRIES_VARIATIONS_PREFIX
from athenian.api.models.state.models import \
    AccountFeature, Feature, FeatureComponent, God


@pytest.fixture
def current_module():
    return sys.modules[__name__].__name__


@pytest.fixture
def base_testing_module(current_module):
    return current_module[: current_module.rfind(".")]


def calc_pull_request_metrics_line_github():
    """This is a fake function for testing."""
    return True


def test_get_calculator_no_variation(base_testing_module):
    calc = get_calculator("github", "prs_linear", base_module=base_testing_module)
    expected = "athenian.api.controllers.features.entries:calc_pull_request_metrics_line_github"
    actual = f"{calc.__module__}:{calc.__name__}"
    assert actual == expected


def test_get_calculator_missing_module_no_error():
    calc = get_calculator(
        "github", "prs_linear", variation="test_entries", base_module="missing_module",
    )
    expected = "athenian.api.controllers.features.entries:calc_pull_request_metrics_line_github"
    actual = f"{calc.__module__}:{calc.__name__}"
    assert actual == expected


def test_get_calculator_missing_implementation_no_error(base_testing_module):
    calc = get_calculator(
        "github",
        "prs_histogram",
        variation="test_entries",
        base_module=base_testing_module,
    )
    expected = (
        "athenian.api.controllers.features.entries:calc_pull_request_histograms_github"
    )
    actual = f"{calc.__module__}:{calc.__name__}"
    assert actual == expected


def test_get_calculator_raise_error(base_testing_module):
    try:
        get_calculator(
            "github",
            "prs_linear",
            variation="test_entries",
            base_module="missing_module",
            raise_err=True,
        )
    except ModuleNotFoundError:
        assert True
    else:
        raise AssertionError("Expected ModuleNotFoundError not raised")

    try:
        get_calculator(
            "github",
            "prs_histogram",
            variation="test_entries",
            raise_err=True,
            base_module=base_testing_module,
        )
    except RuntimeError:
        assert True
    else:
        raise AssertionError("Expected RuntimeError not raised")


def test_get_calculator_variation_found(base_testing_module, current_module):
    calc = get_calculator(
        "github",
        "prs_linear",
        variation="test_entries",
        base_module=base_testing_module,
    )
    expected = f"{current_module}:calc_pull_request_metrics_line_github"
    actual = f"{calc.__module__}:{calc.__name__}"
    assert actual == expected
    assert calc()


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
                name=f"{METRIC_ENTRIES_VARIATIONS_PREFIX['github']}test_entries",
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
                name=f"{METRIC_ENTRIES_VARIATIONS_PREFIX['github']}test_entries",
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
