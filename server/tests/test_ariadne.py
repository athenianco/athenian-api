from dataclasses import dataclass

from ariadne import gql, make_executable_schema, QueryType
from graphql import graphql_sync

from athenian.api.ariadne import ariadne_disable_default_user


class TestGraphiQL:
    async def test(self, client) -> None:
        response = await client.request(method="GET", path="/align/ui")
        html = (await response.read()).decode("utf-8")

        assert "<html>" in html
        assert "graphiql.min.js" in html


class TestAriadnDisableDefaultUser:

    @dataclass
    class FakeContext:
        is_default_user: bool
        uid: str = "u00"

    def test(self) -> None:
        query = QueryType()
        type_defs = gql("""
        type Query {
          hello: String!
        }
        """)

        @query.field("hello")
        @ariadne_disable_default_user
        def travel(_, info):
            return "world"

        schema = make_executable_schema(type_defs, [query])
        q = "query { hello }"

        result = graphql_sync(schema, q, context_value=self.FakeContext(False))
        assert result.data == {"hello": "world"}
        assert result.errors is None

        result = graphql_sync(schema, q, context_value=self.FakeContext(True))
        assert result.data is None
        assert result.errors is not None
        assert "u00 is the default user" in result.errors[0].message
