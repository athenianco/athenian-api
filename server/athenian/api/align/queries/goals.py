from typing import Any

from ariadne import ObjectType
from graphql import GraphQLResolveInfo

from athenian.api.tracing import sentry_span

query = ObjectType("Query")


@query.field("goals")
@sentry_span
async def resolve_goals(obj: Any, info: GraphQLResolveInfo, **kwargs) -> Any:
    """Serve goals()."""
    return []
