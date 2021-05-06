from athenian.api.controllers.features.entries import make_calculator
from athenian.api.experiments.preloading import MetricEntriesCalculator


def test_get_calculator_variation_found(
    mdb, pdb, rdb, cache,
):
    calc = make_calculator(1, (1,), mdb, pdb, rdb, cache, variation="preloading")
    assert isinstance(calc, MetricEntriesCalculator)
