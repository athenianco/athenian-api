import argparse
from typing import Any

from athenian.api.controllers.events_controller.deployments import (
    resolve_deployed_component_references,
)
from athenian.api.internal.refetcher import Refetcher
from athenian.api.precompute.context import PrecomputeContext


async def main(context: PrecomputeContext, args: argparse.Namespace) -> Any:
    """Fill missing commit references in the deployed components."""
    await resolve_deployed_component_references(
        Refetcher(args.refetch_topic), context.sdb, context.mdb, context.rdb, context.cache,
    )
