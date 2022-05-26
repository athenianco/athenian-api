from pathlib import Path
import sys
from typing import Any, cast, Dict, Optional

from ariadne import gql, make_executable_schema, SchemaBindable
from graphql import GraphQLInputType, GraphQLNonNull, GraphQLScalarType, GraphQLSchema, \
    GraphQLType, is_leaf_type, Undefined, ValueNode, VariableNode
from graphql.utilities import type_comparators, value_from_ast as original_value_from_ast
from graphql.validation.rules import variables_in_allowed_position

from athenian.api import metadata


def create_graphql_schema() -> GraphQLSchema:
    """Dynamically import all the children modules and compile a GraphQL schema."""
    patch_graphql()

    self_path = Path(__file__)
    amalgamation = "\n".join(
        p.read_text() for p in Path(self_path.with_name("spec")).glob("**/*.graphql")
    )
    spec = gql(amalgamation)
    bindables = []
    for pkg in ("queries", "mutations", "scalars"):
        for file_path in (self_path.parent / pkg).glob("*.py"):
            if file_path.stem == "__init__":
                continue
            __import__(package := f"{metadata.__package__}.align.{pkg}.{file_path.stem}")
            for var in sys.modules[package].__dict__.values():
                if isinstance(var, SchemaBindable) and not isinstance(var, type):
                    bindables.append(var)
    directives = {}
    for file_path in (self_path.parent / "directives").glob("*.py"):
        if file_path.stem == "__init__":
            continue
        __import__(package := f"{metadata.__package__}.align.directives.{file_path.stem}")
        directives.update(sys.modules[package].directives)
    return make_executable_schema(spec, *bindables, directives=directives)


def patch_graphql():
    """Add support for scalar type inheritance."""
    original_is_type_sub_type_of = type_comparators.is_type_sub_type_of

    def is_type_sub_type_of_patched(
            schema: GraphQLSchema, maybe_subtype: GraphQLType, super_type: GraphQLType,
    ) -> bool:
        if is_not_null := type_comparators.is_non_null_type(super_type):
            super_type = type_comparators.cast(GraphQLNonNull, super_type).of_type
        while True:
            try:
                super_type = super_type.extensions["()"]
            except (KeyError, TypeError, AttributeError):
                break
        if is_not_null:
            super_type = GraphQLNonNull(super_type)
        return original_is_type_sub_type_of(schema, maybe_subtype, super_type)

    type_comparators.is_type_sub_type_of = is_type_sub_type_of_patched
    variables_in_allowed_position.is_type_sub_type_of = is_type_sub_type_of_patched

    def value_from_ast_patched(
            value_node: Optional[ValueNode],
            type_: GraphQLInputType,
            variables: Optional[Dict[str, Any]] = None,
    ) -> Any:
        result = original_value_from_ast(value_node, type_, variables)
        if type_comparators.is_non_null_type(type_):
            type_ = type_.of_type
        if isinstance(value_node, VariableNode) and is_leaf_type(type_):
            type_ = cast(GraphQLScalarType, type_)
            try:
                result = type_.parse_value(result)
            except Exception:
                result = Undefined
        return result

    sys.modules["graphql"].value_from_ast = value_from_ast_patched
    sys.modules["graphql.utilities"].value_from_ast = value_from_ast_patched
    sys.modules["graphql.utilities.value_from_ast"].value_from_ast = value_from_ast_patched
