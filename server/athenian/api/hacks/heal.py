import asyncio
import sys

from flogging import flogging

from athenian.api.precompute.refetcher import Refetcher


async def main():
    """Send a refetch message to pubsub."""
    refetcher = Refetcher(sys.argv[1], tuple(int(s) for s in sys.argv[2].split(",")))
    try:
        await refetcher.submit_commits([int(s) for s in sys.argv[3].split(",")])
    finally:
        await refetcher.close()


if __name__ == "__main__":
    flogging.setup("INFO")
    sys.exit(asyncio.run(main()))
