from typing import Any, Union

from ariadne import SchemaDirectiveVisitor
from graphql import default_field_resolver, GraphQLArgument
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
        orig_type = field.type
        orig_named_type = get_named_type(field.type)
        assert orig_named_type.name == "String"

        field.type = GraphQLDateType()
        # re-wrap with GraphQLNonNull if needed
        if isinstance(orig_type, GraphQLNonNull):
            field.type = GraphQLNonNull(field.type)
        return field

    def visit_argument_definition(self,
                                  argument: GraphQLArgument,
                                  field: GraphQLField,
                                  object_type: GraphQLInputObjectType):
        """Wrap input argument to serialize/parse date values."""
        # TODO(vmarkovtsev): implement this
        return argument


class GraphQLDateType(GraphQLScalarType):
    """GraphQL type for a string containing a date."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Init the type."""
        kwargs.setdefault("name", "Date")
        super().__init__(*args, **kwargs)

    @staticmethod
    def serialize(value: Any) -> Any:
        """Serialize the date value to a string."""
        return serialize_date(value)

    @staticmethod
    def parse_value(value: Any) -> Any:
        """Parse the date object from a string value."""
        return deserialize_date(value, min_=None, max_future_delta=None)


directives = {"date": DateDirective}
