from pathlib import Path
import sys

from ariadne import gql, make_executable_schema
from graphql import GraphQLSchema

from athenian.api import metadata


def create_graphql_schema() -> GraphQLSchema:
    """Dynamically import all the children modules and compile a GraphQL schema."""
    self_path = Path(__file__)
    spec = gql(self_path.with_name("spec.gql").read_text())
    bindables = []
    for pkg in ("queries", "scalars"):
        for file_path in (self_path.parent / pkg).glob("*.py"):
            if file_path.stem == "__init__":
                continue
            __import__(package := f"{metadata.__package__}.align.{pkg}.{file_path.stem}")
            for var in sys.modules[package].__dict__.values():
                # isinstance(var, SchemaBindable)
                # https://github.com/mirumee/ariadne/pull/828
                if hasattr(var, "bind_to_schema") and not isinstance(var, type):
                    bindables.append(var)
    directives = {}
    for file_path in (self_path.parent / "directives").glob("*.py"):
        if file_path.stem == "__init__":
            continue
        __import__(package := f"{metadata.__package__}.align.directives.{file_path.stem}")
        directives.update(sys.modules[package].directives)
    return make_executable_schema(spec, *bindables, directives=directives)
