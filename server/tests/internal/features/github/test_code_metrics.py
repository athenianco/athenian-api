from datetime import datetime, timezone

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.github.commit import FilterCommitsProperty


@with_defer
async def test_calc_code_metrics_github_cache(metrics_calculator_factory, meta_ids, prefixer):
    calculator = metrics_calculator_factory(1, meta_ids, with_cache=True)
    stats1 = await calculator.calc_code_metrics_github(
        FilterCommitsProperty.BYPASSING_PRS,
        [datetime(2016, 6, 1, tzinfo=timezone.utc), datetime(2019, 6, 1, tzinfo=timezone.utc)],
        ["src-d/go-git"],
        None,
        None,
        True,
        prefixer,
    )
    await wait_deferred()
    calculator._mdb = calculator._pdb = None
    stats2 = await calculator.calc_code_metrics_github(
        FilterCommitsProperty.BYPASSING_PRS,
        [datetime(2016, 6, 1, tzinfo=timezone.utc), datetime(2019, 6, 1, tzinfo=timezone.utc)],
        ["src-d/go-git"],
        None,
        None,
        True,
        prefixer,
    )
    assert stats1 == stats2
