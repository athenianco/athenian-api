import argparse
from typing import List

from athenian.api.precompute.context import PrecomputeContext


async def main(context: PrecomputeContext, args: argparse.Namespace) -> List[int]:
    """Load all accounts, find which must be precomputed, and return their IDs."""
    return []
