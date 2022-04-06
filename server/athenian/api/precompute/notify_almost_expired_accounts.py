import argparse

from athenian.api.precompute.context import PrecomputeContext


async def main(context: PrecomputeContext, args: argparse.Namespace) -> None:
    """Find accounts which will expire within 1 hour and report them to Slack."""
