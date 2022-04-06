import argparse
from typing import Any

from athenian.api.precompute.context import PrecomputeContext


async def main(context: PrecomputeContext, args: argparse.Namespace) -> Any:
    """Send Slack messages about accounts which are about to expire."""
