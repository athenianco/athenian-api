import sys

import pytest

from athenian.api.controllers.features.entries import get_calculator


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
