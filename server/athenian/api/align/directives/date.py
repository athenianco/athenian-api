from typing import Union

from ariadne import SchemaDirectiveVisitor
from graphql import default_field_resolver, GraphQLString, is_non_null_type
from graphql.type import get_named_type, GraphQLField, GraphQLInputField, GraphQLInputObjectType, \
    GraphQLInterfaceType, GraphQLNonNull, GraphQLObjectType, GraphQLScalarType

from athenian.api.serialization import deserialize_date, serialize_date


class DateDirective(SchemaDirectiveVisitor):
    """GraphQL directive for a string field expressing a date."""

    def visit_field_definition(
        self,
        field: GraphQLField,
        object_type: Union[GraphQLObjectType, GraphQLInterfaceType],
    ) -> GraphQLField:
        """Wrap field resolver to convert resolved date value to a string."""
        original_resolver = field.resolve or default_field_resolver

        def resolve_formatted_date(obj, info, **kwargs):
            result = original_resolver(obj, info, **kwargs)
            if result is None:
                return None
            return serialize_date(result)

        field.resolve = resolve_formatted_date
        return field

    def visit_input_field_definition(
        self, field: GraphQLInputField, object_type: GraphQLInputObjectType,
    ) -> GraphQLInputField:
        """Wrap field type to serialize/parse date values."""
        is_not_null = is_non_null_type(field.type)
        orig_named_type = get_named_type(field.type)
        assert orig_named_type.name == "String"

        field.type = GraphQLDateType
        # re-wrap with GraphQLNonNull if needed
        if is_not_null:
            field.type = GraphQLNonNull(field.type)
        return field


GraphQLDateType = GraphQLScalarType(
    name="Date",
    description="ISO 8601 date.",
    serialize=serialize_date,
    parse_value=deserialize_date,
    extensions={"()": GraphQLString},
)


directives = {"date": DateDirective}
