from athenian.api.controllers.features.entries import get_calculator


def test_get_calculator_variation_found():
    calc = get_calculator("github", "prs_linear", variation="preloading")
    expected = "athenian.api.experiments.preloading.entries:calc_pull_request_metrics_line_github"
    actual = f"{calc.__module__}:{calc.__name__}"
    assert actual == expected
    assert calc()
