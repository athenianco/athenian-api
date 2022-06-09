from datetime import timedelta
from typing import Any
from unittest import mock

from ariadne import make_executable_schema, QueryType
from graphql import graphql_sync, GraphQLResolveInfo

from athenian.api.align.types.goal_metric_value import goal_metric_value, \
    resolve_goal_metric_value_str


class TestResolveGoalMetricValueStr:
    def test_already_str(self) -> None:
        assert self._resolve({"str": "foo"}) == "foo"

    def test_key_missing(self) -> None:
        assert self._resolve({"int": 2}) is None

    def test_timedelta(self) -> None:
        assert self._resolve({"str": timedelta(hours=2)}) == "7200s"

    def _resolve(self, obj: Any) -> Any:
        gql_info = mock.Mock(spec=GraphQLResolveInfo)
        return resolve_goal_metric_value_str(obj, gql_info)


class TestGoalMetricValueType:
    def test_base(self) -> None:
        type_defs = """
        type GoalMetricValue {
          str: String
          float: Float
        }
        type Query {
          one: GoalMetricValue
          two: GoalMetricValue
          three: GoalMetricValue
        }
        """
        query = QueryType()
        query.set_field("one", lambda *_: {"str": "bar"})
        query.set_field("two", lambda *_: {"str": timedelta(seconds=35, milliseconds=200)})
        query.set_field("three", lambda *_: {"float": 42.2})

        schema = make_executable_schema(type_defs, [query, goal_metric_value])

        q = "{ one {str, float}, two {str, float}, three {str, float} }"

        result = graphql_sync(schema, q)
        data = result.data
        assert data is not None
        assert data["one"] == {"float": None, "str": "bar"}
        assert data["two"] == {"float": None, "str": "35s"}
        assert data["three"] == {"float": 42.2, "str": None}
