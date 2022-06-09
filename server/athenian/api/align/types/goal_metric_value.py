from datetime import timedelta
from typing import Any

from ariadne import ObjectType
from graphql import GraphQLResolveInfo

from athenian.api.serialization import serialize_timedelta

goal_metric_value = ObjectType("GoalMetricValue")


@goal_metric_value.field("str")
def resolve_goal_metric_value_str(obj: Any, info: GraphQLResolveInfo) -> Any:
    """Resolve the str field of the GoalMetricValue type."""
    # resolver is called with the TeamMetricValue model dict serialization
    # str key can be missing since None-valued keys are removed from serialization
    value = obj.get("str")
    if isinstance(value, timedelta):
        return serialize_timedelta(value)
    return value
