from datetime import date, timedelta

from ariadne import MutationType, QueryType, make_executable_schema
from freezegun import freeze_time
from graphql import graphql_sync

from athenian.api.align.scalars.date import date_scalar


@freeze_time("2022-04-01")
def test_on_field_definition() -> None:
    type_defs = """
        scalar Date
        type Query {
          today: Date!
        }
    """
    query = QueryType()
    query.set_field("today", lambda *_: date.today())
    schema = make_executable_schema(type_defs, [query, date_scalar])
    result = graphql_sync(schema, "{today}")
    assert result.data == {"today": "2022-04-01"}
    assert result.errors is None


def test_on_input_field_definition() -> None:
    type_defs = """
        scalar Date
        type Query {hello: String!}
        type Mutation {
          travel(input: TravelInput!): Boolean
        }
        input TravelInput {
          when: Date!
        }
    """
    mutation = MutationType()
    check_date = date.today() - timedelta(days=365)

    @mutation.field("travel")
    def travel(*_, input):
        # "when" string has been converted to a date object
        assert input == {"when": check_date}
        return True

    schema = make_executable_schema(type_defs, [mutation, date_scalar])

    q = "mutation m($travelInput: TravelInput!) { travel(input: $travelInput) }"
    result = graphql_sync(schema, q, variable_values={"travelInput": {"when": str(check_date)}})
    assert result.data == {"travel": True}
    assert result.errors is None
