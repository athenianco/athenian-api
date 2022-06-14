from pathlib import Path
import sys
from typing import Any, Dict, Optional, cast

from ariadne import SchemaBindable, gql, make_executable_schema
from graphql import (
    GraphQLInputType,
    GraphQLNonNull,
    GraphQLScalarType,
    GraphQLSchema,
    GraphQLType,
    Undefined,
    ValueNode,
    VariableNode,
    is_leaf_type,
)
from graphql.utilities import type_comparators, value_from_ast as original_value_from_ast
from graphql.validation.rules import variables_in_allowed_position

from athenian.api import metadata


def create_graphql_schema() -> GraphQLSchema:
    """Dynamically import all the children modules and compile a GraphQL schema."""
    self_path = Path(__file__)
    amalgamation = "\n".join(
        p.read_text() for p in Path(self_path.with_name("spec")).glob("**/*.graphql")
    )
    spec = gql(amalgamation)
    bindables = []
    for pkg in ("queries", "mutations", "scalars", "types"):
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
