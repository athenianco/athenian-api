import asyncio
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
import logging  # noqa
import os

import databases

from athenian.api import create_memcached, setup_cache_metrics
from athenian.api.controllers.features.entries import calc_pull_request_metrics_line_github  # noqa
from athenian.api.controllers.features.github.pull_request_filter import filter_pull_requests
from athenian.api.controllers.miners.types import Property
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting
from athenian.api.models.web import PullRequestMetricID  # noqa


async def main():
    """Go away linter."""
    if False:
        cache = create_memcached("0.0.0.0:7001", logging.getLogger())
        setup_cache_metrics(cache, {}, None)
        for v in cache.metrics["context"].values():
            v.set(defaultdict(int))
    else:
        cache = None
    password = os.getenv("POSTGRES_PASSWORD")
    addr = "production-cloud-sql:%s@0.0.0.0:5432" % password
    # addr = "postgres:postgres@0.0.0.0:5433"
    mdb = databases.Database("postgresql://%s/metadata" % addr)
    await mdb.connect()
    if os.getenv("DISABLE_PDB"):
        pdb = databases.Database("sqlite:///tests/pdb.sqlite")
    else:
        pdb = databases.Database("postgresql://%s/precomputed" % addr)
    await pdb.connect()
    pdb.metrics = {
        "hits": ContextVar("pdb_hits", default=defaultdict(int)),
        "misses": ContextVar("pdb_misses", default=defaultdict(int)),
    }

    time_from = datetime(2019, 6, 5, tzinfo=timezone.utc)
    time_to = datetime(2020, 6, 30, tzinfo=timezone.utc)
    repos = {"classified"}
    # TODO(vmarkovtsev): load these from the settings
    settings = {
        "github.com/" + r: ReleaseMatchSetting("{{default}}", ".*", ReleaseMatch.tag_or_branch)
        for r in repos
    }
    # """
    prs = list(await filter_pull_requests(
        set(Property),
        time_from,
        time_to,
        repos,
        {},
        set(),
        False,
        settings,
        mdb,
        pdb,
        cache,
    ))
    for pr in prs:
        print("https://%s/pull/%d" % (pr.repository, pr.number))
    """
    metrics = (await calc_pull_request_metrics_line_github(
        [PullRequestMetricID.PR_REVIEW_TIME],
        [[time_from, time_to]], repos, {}, False, settings, mdb, pdb, cache,
    ))
    print(metrics)
    # """
    # breakpoint()
    print()


if __name__ == "__main__":
    exit(asyncio.run(main()))
