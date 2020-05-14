import asyncio
from datetime import datetime, timezone
import logging  # noqa
import os

import databases

from athenian.api import create_memcached, setup_cache_metrics  # noqa
from athenian.api.controllers.features.entries import calc_pull_request_metrics_line_github  # noqa
from athenian.api.controllers.features.github.pull_request_filter import filter_pull_requests
from athenian.api.controllers.miners.pull_request_list_item import Property
from athenian.api.controllers.settings import Match, ReleaseMatchSetting
from athenian.api.models.web import PullRequestMetricID  # noqa


async def main():
    """Go away linter."""
    cache = None  # create_memcached("0.0.0.0:7001", logging.getLogger())
    # setup_cache_metrics(cache, None)
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

    time_from = datetime(2020, 4, 30, tzinfo=timezone.utc)
    time_to = datetime(2020, 5, 15, tzinfo=timezone.utc)
    repos = ["censored"]
    # TODO(vmarkovtsev): load these from the settings
    settings = {
        "github.com/" + r: ReleaseMatchSetting("{{default}}", ".*", Match.tag_or_branch)
        for r in repos
    }
    # """
    prs = list(await filter_pull_requests(
        [Property.APPROVE_HAPPENED, Property.MERGING,
         Property.RELEASING, Property.RELEASE_HAPPENED],
        time_from,
        time_to,
        repos,
        settings,
        {},
        mdb,
        cache,
    ))
    for pr in prs:
        print("https://%s/pull/%d" % (pr.repository, pr.number), pr.stage_timings["review"])
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
