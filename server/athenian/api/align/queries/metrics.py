from typing import Any, Mapping

from ariadne import ObjectType
from graphql import GraphQLResolveInfo

from athenian.api.tracing import sentry_span

query = ObjectType("Query")


@query.field("metricsCurrentValues")
@sentry_span
async def resolve_metrics_current_values(obj: Any,
                                         info: GraphQLResolveInfo,
                                         accountId: int,
                                         params: Mapping[str, Any]) -> Any:
    """Serve metricsCurrentValues()."""
    pass
    """
    sdb, mdb, pdb, rdb, cache = \
        info.context.sdb, info.context.mdb, info.context.pdb, info.context.rdb, info.context.cache
    """
    raise NotImplementedError
