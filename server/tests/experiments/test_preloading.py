import pytest

from athenian.api.controllers.features.entries import make_calculator
from athenian.api.experiments.preloading import MetricEntriesCalculator


# TODO: this fails because precomputed data are not loaded for unittest
@pytest.mark.xfail
def test_get_calculator_variation_found(
    mdb, pdb, rdb, cache, with_preloading,
):
    if with_preloading:
        calc = make_calculator(1, (6366825,), mdb, pdb, rdb, cache, variation="preloading")
        assert isinstance(calc, MetricEntriesCalculator)
