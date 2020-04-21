import asyncio
from collections import defaultdict
from datetime import date, datetime, timedelta
from itertools import chain, product
import logging
import pickle
from typing import Collection, Dict, Iterable, List, Optional, Sequence, Tuple

import aiomcache
from dateutil.rrule import DAILY, rrule

from athenian.api import metadata
from athenian.api.cache import gen_cache_key, max_exptime
from athenian.api.controllers.miners.github.pull_request import MinedPullRequest, PullRequestTimes
from athenian.api.models.metadata.github import PullRequest


async def load_cached_released_times(date_from: date,
                                     date_to: date,
                                     repos: Collection[str],
                                     cache: Optional[aiomcache.Client],
                                     ) -> Dict[str, PullRequestTimes]:
    """Fetch PullRequestTimes of the cached released PRs."""
    if cache is None:
        return {}
    log = logging.getLogger("%s.cached_released_times" % metadata.__package__)
    batch_size = 32

    async def fetch_repo_days(repo_days: List[Tuple[str, datetime]],
                              ) -> Iterable[Tuple[str, PullRequestTimes]]:
        cache_keys = [gen_cache_key("cached_released_times|%s|%d", repo, day.date().toordinal())
                      for repo, day in repo_days]
        try:
            buffers = await cache.multi_get(*cache_keys)
        except aiomcache.exceptions.ClientException:
            log.exception("failed to fetch cached %s", repo_days)
            return []
        return chain.from_iterable(pickle.loads(b) for b in buffers if b is not None)

    date_to = max(date_from, date_to - timedelta(days=1))
    tasks = list(product(repos, rrule(DAILY, dtstart=date_from, until=date_to)))
    batches = [fetch_repo_days(tasks[i:i + batch_size]) for i in range(0, len(tasks), batch_size)]
    parallel_retrievals = 16
    results = []
    for i in range(0, len(batches), parallel_retrievals):
        results.extend(chain.from_iterable(await asyncio.gather(
            *batches[i:i + parallel_retrievals], return_exceptions=True)))
    return dict(r for r in results if not isinstance(r, Exception))


async def store_cached_released_times(prs: Sequence[MinedPullRequest],
                                      times: Sequence[PullRequestTimes],
                                      cache: Optional[aiomcache.Client],
                                      ) -> None:
    """Put the PullRequestTimes belonging to released PRs to the cache."""
    if cache is None:
        return
    log = logging.getLogger("%s.cached_released_times" % metadata.__package__)
    repodays = defaultdict(lambda: defaultdict(list))
    rfnkey = PullRequest.repository_full_name.key
    for pr, t in zip(prs, times):
        if t.released:
            repodays[pr.pr[rfnkey]][t.released.best.date()].append(
                (pr.pr[PullRequest.node_id.key], t))

    async def store_repo_day(repo: str,
                             day: date,
                             items: List[Tuple[str, PullRequestTimes]],
                             ) -> None:
        cache_key = gen_cache_key("cached_released_times|%s|%d", repo, day.toordinal())
        payload = pickle.dumps(items)
        try:
            await cache.set(cache_key, payload, exptime=max_exptime)
        except aiomcache.exceptions.ClientException:
            log.exception("Failed to put %d bytes in memcached", len(payload))

    tasks = [store_repo_day(r, day, rdt)
             for (r, days) in repodays.items()
             for (day, rdt) in days.items()]
    chunk_size = 128
    for i in range(0, len(tasks), chunk_size):
        await asyncio.gather(*tasks[i:i + chunk_size], return_exceptions=True)
