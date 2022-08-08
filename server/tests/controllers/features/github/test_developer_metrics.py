from datetime import datetime, timezone

from athenian.api.defer import with_defer
from athenian.api.internal.features.metric import MetricInt
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.developer import DeveloperTopic
from athenian.api.internal.settings import LogicalRepositorySettings


@with_defer
async def test_developer_metrics_smoke(
    metrics_calculator_factory,
    pdb,
    mdb,
    rdb,
    release_match_setting_tag,
    prefixer,
):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    utc = timezone.utc
    requested_topics = {
        DeveloperTopic.commits_pushed,
        DeveloperTopic.lines_changed,
        DeveloperTopic.active,
        DeveloperTopic.prs_created,
        DeveloperTopic.prs_merged,
        DeveloperTopic.releases,
        DeveloperTopic.pr_comments,
        DeveloperTopic.prs_reviewed,
        DeveloperTopic.reviews,
        DeveloperTopic.review_approvals,
        DeveloperTopic.review_rejections,
        DeveloperTopic.review_neutrals,
    }
    metrics, topics = await metrics_calculator_no_cache.calc_developer_metrics_github(
        [["mcuadros"], ["smola"]],
        [["src-d/go-git"]],
        [
            [
                datetime(2017, 1, 1, tzinfo=utc),
                datetime(2018, 1, 1, tzinfo=utc),
                datetime(2019, 1, 1, tzinfo=utc),
            ],
            [datetime(2017, 1, 1, tzinfo=utc), datetime(2019, 1, 1, tzinfo=utc)],
        ],
        requested_topics,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
    )
    assert set(topics) == requested_topics
    metrics = metrics.swapaxes(1, 2)  # the test was written before the axes swap
    order = [
        DeveloperTopic.active,
        DeveloperTopic.lines_changed,
        DeveloperTopic.commits_pushed,
        DeveloperTopic.prs_created,
        DeveloperTopic.prs_merged,
        DeveloperTopic.releases,
        DeveloperTopic.pr_comments,
        DeveloperTopic.prs_reviewed,
        DeveloperTopic.reviews,
        DeveloperTopic.review_approvals,
        DeveloperTopic.review_rejections,
        DeveloperTopic.review_neutrals,
    ]
    if topics != order:
        order = [topics.index(t) for t in order]
        for x in metrics:
            for y in x:
                for ts in y:
                    for row in ts:
                        row[:] = [row[i] for i in order]
    assert metrics.shape == (1, 2, 2)
    metric = MetricInt.from_fields
    assert metrics[0, 0, 0] == [
        [
            metric(exists=True, value=1, confidence_min=None, confidence_max=None),
            metric(exists=True, value=64687, confidence_min=None, confidence_max=None),
            metric(exists=True, value=416, confidence_min=None, confidence_max=None),
            metric(exists=True, value=53, confidence_min=None, confidence_max=None),
            metric(exists=True, value=197, confidence_min=None, confidence_max=None),
            metric(exists=True, value=7, confidence_min=None, confidence_max=None),
            metric(exists=True, value=243, confidence_min=None, confidence_max=None),
            metric(exists=True, value=72, confidence_min=None, confidence_max=None),
            metric(exists=True, value=37 + 38 + 69, confidence_min=None, confidence_max=None),
            metric(exists=True, value=37, confidence_min=None, confidence_max=None),
            metric(exists=True, value=38, confidence_min=None, confidence_max=None),
            metric(exists=True, value=69, confidence_min=None, confidence_max=None),
        ],
        [
            metric(exists=True, value=1, confidence_min=None, confidence_max=None),
            metric(exists=True, value=25545, confidence_min=None, confidence_max=None),
            metric(exists=True, value=140, confidence_min=None, confidence_max=None),
            metric(exists=True, value=10, confidence_min=None, confidence_max=None),
            metric(exists=True, value=112, confidence_min=None, confidence_max=None),
            metric(exists=True, value=14, confidence_min=None, confidence_max=None),
            metric(exists=True, value=106, confidence_min=None, confidence_max=None),
            metric(exists=True, value=24, confidence_min=None, confidence_max=None),
            metric(exists=True, value=14 + 11 + 22, confidence_min=None, confidence_max=None),
            metric(exists=True, value=14, confidence_min=None, confidence_max=None),
            metric(exists=True, value=11, confidence_min=None, confidence_max=None),
            metric(exists=True, value=22, confidence_min=None, confidence_max=None),
        ],
    ]
    assert metrics[0, 0, 1] == [
        [
            metric(exists=True, value=1, confidence_min=None, confidence_max=None),
            metric(exists=True, value=90232, confidence_min=None, confidence_max=None),
            metric(exists=True, value=556, confidence_min=None, confidence_max=None),
            metric(exists=True, value=63, confidence_min=None, confidence_max=None),
            metric(exists=True, value=309, confidence_min=None, confidence_max=None),
            metric(exists=True, value=21, confidence_min=None, confidence_max=None),
            metric(exists=True, value=349, confidence_min=None, confidence_max=None),
            metric(exists=True, value=96, confidence_min=None, confidence_max=None),
            metric(exists=True, value=191, confidence_min=None, confidence_max=None),
            metric(exists=True, value=51, confidence_min=None, confidence_max=None),
            metric(exists=True, value=49, confidence_min=None, confidence_max=None),
            metric(exists=True, value=91, confidence_min=None, confidence_max=None),
        ],
    ]
    assert metrics[0, 1, 0] == [
        [
            metric(exists=True, value=0, confidence_min=None, confidence_max=None),
            metric(exists=True, value=15237, confidence_min=None, confidence_max=None),
            metric(exists=True, value=126, confidence_min=None, confidence_max=None),
            metric(exists=True, value=34, confidence_min=None, confidence_max=None),
            metric(exists=True, value=18, confidence_min=None, confidence_max=None),
            metric(exists=True, value=2, confidence_min=None, confidence_max=None),
            metric(exists=True, value=179, confidence_min=None, confidence_max=None),
            metric(exists=True, value=90, confidence_min=None, confidence_max=None),
            metric(exists=True, value=138, confidence_min=None, confidence_max=None),
            metric(exists=True, value=73, confidence_min=None, confidence_max=None),
            metric(exists=True, value=25, confidence_min=None, confidence_max=None),
            metric(exists=True, value=39, confidence_min=None, confidence_max=None),
        ],
        [
            metric(exists=True, value=0, confidence_min=None, confidence_max=None),
            metric(exists=True, value=480, confidence_min=None, confidence_max=None),
            metric(exists=True, value=7, confidence_min=None, confidence_max=None),
            metric(exists=True, value=8, confidence_min=None, confidence_max=None),
            metric(exists=True, value=0, confidence_min=None, confidence_max=None),
            metric(exists=True, value=0, confidence_min=None, confidence_max=None),
            metric(exists=True, value=70, confidence_min=None, confidence_max=None),
            metric(exists=True, value=36, confidence_min=None, confidence_max=None),
            metric(exists=True, value=55, confidence_min=None, confidence_max=None),
            metric(exists=True, value=29, confidence_min=None, confidence_max=None),
            metric(exists=True, value=10, confidence_min=None, confidence_max=None),
            metric(exists=True, value=16, confidence_min=None, confidence_max=None),
        ],
    ]
    assert metrics[0, 1, 1] == [
        [
            metric(exists=True, value=0, confidence_min=None, confidence_max=None),
            metric(exists=True, value=15717, confidence_min=None, confidence_max=None),
            metric(exists=True, value=133, confidence_min=None, confidence_max=None),
            metric(exists=True, value=42, confidence_min=None, confidence_max=None),
            metric(exists=True, value=18, confidence_min=None, confidence_max=None),
            metric(exists=True, value=2, confidence_min=None, confidence_max=None),
            metric(exists=True, value=249, confidence_min=None, confidence_max=None),
            metric(exists=True, value=126, confidence_min=None, confidence_max=None),
            metric(exists=True, value=193, confidence_min=None, confidence_max=None),
            metric(exists=True, value=102, confidence_min=None, confidence_max=None),
            metric(exists=True, value=35, confidence_min=None, confidence_max=None),
            metric(exists=True, value=55, confidence_min=None, confidence_max=None),
        ],
    ]
