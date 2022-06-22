from datetime import date, timedelta

from ariadne import MutationType, QueryType, make_executable_schema
from freezegun import freeze_time
from graphql import graphql_sync

from athenian.api.align.scalars.date import date_scalar


class TestDateScalarTypeField:
    def test_base(self) -> None:
        type_defs = """
            scalar Date
            type Query {
              birthday: Date!
            }
        """
        query = QueryType()
        query.set_field("birthday", lambda *_: date(2022, 4, 1))
        schema = make_executable_schema(type_defs, [query, date_scalar])
        result = graphql_sync(schema, "{birthday}")
        assert result.data == {"birthday": "2022-04-01"}
        assert result.errors is None


class TestDateScalarOnInputField:
    _TYPE_DEFS = """
    scalar Date
    type Query {hello: String!}
    type Mutation {
        travel(input: TravelInput!): Boolean
    }
    input TravelInput {
        when: Date!
    }
    """

    def test_base(self) -> None:
        mutation = MutationType()
        check_date = date.today() - timedelta(days=365)

        @mutation.field("travel")
        def travel(*_, input):
            # "when" string has been converted to a date object
            assert input == {"when": check_date}
            return True

        schema = make_executable_schema(self._TYPE_DEFS, [mutation, date_scalar])

        q = "mutation m($travelInput: TravelInput!) { travel(input: $travelInput) }"
        result = graphql_sync(
            schema, q, variable_values={"travelInput": {"when": check_date.isoformat()}},
        )
        assert result.data == {"travel": True}
        assert result.errors is None

    @freeze_time("2022-04-01")
    def test_next_year_date(self) -> None:
        mutation = MutationType()

        check_date = date(2024, 1, 1)

        @mutation.field("travel")
        def travel(*_, input):
            assert input["when"] == check_date
            return True

        schema = make_executable_schema(self._TYPE_DEFS, [mutation, date_scalar])

        q = "mutation m($travelInput: TravelInput!) { travel(input: $travelInput) }"
        result = graphql_sync(
            schema, q, variable_values={"travelInput": {"when": check_date.isoformat()}},
        )
        assert result.errors is None
