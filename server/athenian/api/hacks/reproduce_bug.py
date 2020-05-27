import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import logging  # noqa
import os

import databases

from athenian.api import create_memcached, setup_cache_metrics
from athenian.api.controllers.features.entries import calc_pull_request_metrics_line_github  # noqa
from athenian.api.controllers.features.github.pull_request_filter import filter_pull_requests
from athenian.api.controllers.miners.pull_request_list_item import Property
from athenian.api.controllers.settings import Match, ReleaseMatchSetting
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
    mdb = databases.Database(
        "postgresql://production-cloud-sql:%s@0.0.0.0:5432/metadata" % password)
    await mdb.connect()
    if os.getenv("DISABLE_PDB"):
        pdb = databases.Database("sqlite:///tests/pdb.sqlite")
    else:
        pdb = databases.Database(
            "postgresql://production-cloud-sql:%s@0.0.0.0:5432/precomputed" % password)
    await pdb.connect()

    time_from = datetime(2020, 5, 13, tzinfo=timezone.utc)
    time_to = datetime(2020, 5, 27, tzinfo=timezone.utc)
    repos = ["classified"]
    # TODO(vmarkovtsev): load these from the settings
    settings = {
        "github.com/" + r: ReleaseMatchSetting("{{default}}", ".*", Match.tag_or_branch)
        for r in repos
    }
    # """
    prs = list(await filter_pull_requests(
        [Property.RELEASE_HAPPENED],
        time_from,
        time_to,
        repos,
        {},
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
        [[time_from, time_to]], repos, [], settings, mdb, pdb, cache,
    ))
    print(metrics)
    """
    # breakpoint()
    print()


if __name__ == "__main__":
    exit(asyncio.run(main()))
