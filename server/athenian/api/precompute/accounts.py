import argparse

from athenian.api.precompute.context import PrecomputeContext


async def main(context: PrecomputeContext, args: argparse.Namespace) -> None:
    """Precompute several accounts."""
