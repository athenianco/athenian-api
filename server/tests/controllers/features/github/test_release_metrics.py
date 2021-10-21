from datetime import datetime, timezone

from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.web import ReleaseMetricID


@with_defer
async def test_calc_release_metrics_line_github_jira_cache(
        metrics_calculator_factory, release_match_setting_tag, prefixer_promise,
        branches, default_branches, mdb, pdb, rdb, cache):
    metrics_calculator = metrics_calculator_factory(1, (6366825,), with_cache=True)
    metrics, _ = await metrics_calculator.calc_release_metrics_line_github(
        [ReleaseMetricID.RELEASE_PRS],
        [[datetime(2018, 6, 12, tzinfo=timezone.utc),
          datetime(2020, 11, 11, tzinfo=timezone.utc)]],
        [0, 1],
        [["src-d/go-git"]],
        [],
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag, prefixer_promise, branches, default_branches,
    )
    await wait_deferred()
    assert metrics[0][0][0][0][0].value == 130
    metrics, _ = await metrics_calculator.calc_release_metrics_line_github(
        [ReleaseMetricID.RELEASE_PRS],
        [[datetime(2018, 6, 12, tzinfo=timezone.utc),
          datetime(2020, 11, 11, tzinfo=timezone.utc)]],
        [0, 1],
        [["src-d/go-git"]],
        [],
        LabelFilter.empty(),
        JIRAFilter(1, ["10003", "10009"], LabelFilter({"performance", "bug"}, set()),
                   set(), set(), False),
        release_match_setting_tag, prefixer_promise, branches, default_branches,
    )
    assert metrics[0][0][0][0][0].value == 7
