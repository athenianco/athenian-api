from datetime import date

from ariadne import make_executable_schema, MutationType, QueryType
from freezegun import freeze_time
from graphql import graphql_sync

from athenian.api.align.directives.date import DateDirective


@freeze_time("2022-04-01")
def test_on_field_definition() -> None:
    type_defs = """
        directive @date on FIELD_DEFINITION
        type Query {
          today: String @date
        }
    """
    query = QueryType()
    query.set_field("today", lambda *_: date.today())

    schema = make_executable_schema(type_defs, [query], directives={"date": DateDirective})

    result = graphql_sync(schema, "{today}")
    assert result.data == {"today": "2022-04-01"}
    assert result.errors is None


def test_on_input_field_definition() -> None:
    type_defs = """
        directive @date on INPUT_FIELD_DEFINITION
        type Query {hello: String!}
        type Mutation {
          travel(input: TravelInput!): Boolean
        }
        input TravelInput {
          when: String! @date
        }
    """
    mutation = MutationType()

    @mutation.field("travel")
    def travel(*_, input):
        # "when" string has been converted to a date object
        assert input == {"when": date(1985, 10, 26)}
        return True

    schema = make_executable_schema(type_defs, [mutation], directives={"date": DateDirective})

    q = "mutation m($travelInput: TravelInput!) { travel(input: $travelInput) }"
    result = graphql_sync(schema, q, variable_values={"travelInput": {"when": "1985-10-26"}})
    assert result.data == {"travel": True}
    assert result.errors is None
