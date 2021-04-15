from athenian.api.controllers.features.entries import get_calculator
from athenian.api.experiments.preloading import MetricEntriesCalculator


def test_get_calculator_variation_found(
    mdb, pdb, rdb, cache,
):
    calc = get_calculator("github", mdb, pdb, rdb, cache, variation="preloading")
    assert isinstance(calc, MetricEntriesCalculator)
