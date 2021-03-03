from datetime import datetime, timezone

from athenian.api.controllers.features.entries import calc_developer_metrics_github
from athenian.api.controllers.features.metric import Metric
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.developer import DeveloperTopic


async def test_developer_metrics_commits_pushed(
        pdb, mdb, rdb, release_match_setting_tag):
    utc = timezone.utc
    requested_topics = {
        DeveloperTopic.commits_pushed,
        DeveloperTopic.lines_changed,
        DeveloperTopic.active,
    }
    metrics, topics = await calc_developer_metrics_github(
        [["mcuadros"], ["smola"]],
        [["src-d/go-git"]],
        [[datetime(2017, 1, 1, tzinfo=utc),
          datetime(2018, 1, 1, tzinfo=utc),
          datetime(2019, 1, 1, tzinfo=utc)],
         [datetime(2017, 1, 1, tzinfo=utc),
          datetime(2019, 1, 1, tzinfo=utc)]],
        requested_topics,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag,
        1,
        (6366825,), mdb, pdb, rdb, None)
    assert set(topics) == requested_topics
    order = [DeveloperTopic.active,
             DeveloperTopic.lines_changed,
             DeveloperTopic.commits_pushed]
    if topics != order:
        order = [topics.index(t) for t in order]
        for x in metrics:
            for y in x:
                for ts in y:
                    for row in ts:
                        row[:] = [row[i] for i in order]
    assert metrics.shape == (1, 2)
    assert metrics[0][0] == [
        [[Metric(exists=True, value=1, confidence_min=None, confidence_max=None),
          Metric(exists=True, value=64687, confidence_min=None, confidence_max=None),
          Metric(exists=True, value=416, confidence_min=None, confidence_max=None)],
         [Metric(exists=True, value=1, confidence_min=None, confidence_max=None),
          Metric(exists=True, value=25545, confidence_min=None, confidence_max=None),
          Metric(exists=True, value=140, confidence_min=None, confidence_max=None)]],
        [[Metric(exists=True, value=1, confidence_min=None, confidence_max=None),
          Metric(exists=True, value=90232, confidence_min=None, confidence_max=None),
          Metric(exists=True, value=556, confidence_min=None, confidence_max=None)]],
    ]
    assert metrics[0][1] == [
        [[Metric(exists=True, value=0, confidence_min=None, confidence_max=None),
          Metric(exists=True, value=15237, confidence_min=None, confidence_max=None),
          Metric(exists=True, value=126, confidence_min=None, confidence_max=None)],
         [Metric(exists=True, value=0, confidence_min=None, confidence_max=None),
          Metric(exists=True, value=480, confidence_min=None, confidence_max=None),
          Metric(exists=True, value=7, confidence_min=None, confidence_max=None)]],
        [[Metric(exists=True, value=0, confidence_min=None, confidence_max=None),
          Metric(exists=True, value=15717, confidence_min=None, confidence_max=None),
          Metric(exists=True, value=133, confidence_min=None, confidence_max=None)]],
    ]
