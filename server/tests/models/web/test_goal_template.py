import pytest

from athenian.api.models.web.goal_template import GoalTemplate
from athenian.api.models.web.jira_metric_id import JIRAMetricID
from athenian.api.models.web.pull_request_metric_id import PullRequestMetricID
from athenian.api.models.web.release_metric_id import ReleaseMetricID


class TestGoalTemplate:
    def test_successful_metric_validation(self) -> None:
        GoalTemplate(1, "foo", PullRequestMetricID.PR_ALL_MAPPED_TO_JIRA)
        GoalTemplate(1, "foo", JIRAMetricID.JIRA_ACKNOWLEDGED_Q)
        GoalTemplate(1, "foo", ReleaseMetricID.BRANCH_RELEASE_AVG_COMMITS)

    def test_failed_metric_validation(self) -> None:
        with pytest.raises(ValueError):
            GoalTemplate(1, "foo", "not a metric")

    def test_failed_metric_validation_with_setter(self) -> None:
        template = GoalTemplate(1, "foo", PullRequestMetricID.PR_CLOSED)
        with pytest.raises(ValueError):
            template.metric = "not a metric"
