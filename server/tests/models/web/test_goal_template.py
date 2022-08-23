from athenian.api.models.web.goal_template import GoalTemplate
from athenian.api.models.web.jira_metric_id import JIRAMetricID


class TestGoalTemplate:
    def test_to_dict(self) -> None:
        template = GoalTemplate(id=1, name="foo", metric=JIRAMetricID.JIRA_ACKNOWLEDGED_Q)
        serialization = template.to_dict()
        assert serialization["id"] == 1
        assert serialization["name"] == "foo"
        assert serialization["metric"] == JIRAMetricID.JIRA_ACKNOWLEDGED_Q
        assert "repositories" not in serialization
