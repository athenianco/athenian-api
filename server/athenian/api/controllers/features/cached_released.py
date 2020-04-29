import asyncio
from collections import defaultdict
from datetime import date, datetime, timedelta
from itertools import chain, product
import logging
import pickle
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Tuple

import aiomcache
from dateutil.rrule import DAILY, rrule

from athenian.api import metadata
from athenian.api.cache import gen_cache_key, max_exptime
from athenian.api.controllers.miners.github.pull_request import PullRequestTimes
from athenian.api.controllers.settings import ReleaseMatchSetting
from athenian.api.models.metadata.github import PullRequest


def _gen_released_times_cache_key(repo: str,
                                  day: date,
                                  release_settings: Dict[str, ReleaseMatchSetting],
                                  ) -> bytes:
    return gen_cache_key("cached_released_times|3|%s|%d|%s",
                         repo,
                         day.toordinal(),
                         release_settings["github.com/" + repo])


async def load_cached_released_times(date_from: date,
                                     date_to: date,
                                     repos: Collection[str],
                                     developers: Collection[str],
                                     release_settings: Dict[str, ReleaseMatchSetting],
                                     cache: Optional[aiomcache.Client],
                                     ) -> Dict[str, PullRequestTimes]:
    """Fetch PullRequestTimes of the cached released PRs."""
    assert isinstance(date_from, date) and not isinstance(date_from, datetime)
    assert isinstance(date_to, date) and not isinstance(date_to, datetime)
    if cache is None:
        return {}
    log = logging.getLogger("%s.cached_released_times" % metadata.__package__)
    batch_size = 32

    async def fetch_repo_days(repo_days: List[Tuple[str, datetime]],
                              ) -> Iterable[Tuple[str, str, PullRequestTimes]]:
        cache_keys = [_gen_released_times_cache_key(repo, day.date(), release_settings)
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
    if len(developers) > 0:
        developers = set(developers)
        result = dict((r[0], r[2]) for r in results
                      if not isinstance(r, Exception) and r[1] in developers)
    else:
        result = dict((r[0], r[2]) for r in results if not isinstance(r, Exception))
    return result


async def store_cached_released_times(prs: Sequence[Tuple[Dict[str, Any], PullRequestTimes]],
                                      release_settings: Dict[str, ReleaseMatchSetting],
                                      cache: Optional[aiomcache.Client],
                                      ) -> None:
    """Put the PullRequestTimes belonging to released PRs to the cache."""
    if cache is None:
        return
    log = logging.getLogger("%s.cached_released_times" % metadata.__package__)
    repodays = defaultdict(lambda: defaultdict(list))
    rfnkey = PullRequest.repository_full_name.key
    nidkey = PullRequest.node_id.key
    ulkey = PullRequest.user_login.key
    for pr, times in prs:
        if times.released:
            repodays[pr[rfnkey]][times.released.best.date()].append((pr[nidkey], pr[ulkey], times))

    async def store_repo_day(repo: str,
                             day: date,
                             items: List[Tuple[str, str, PullRequestTimes]],
                             ) -> None:
        cache_key = _gen_released_times_cache_key(repo, day, release_settings)
        payload = pickle.dumps(items)
        try:
            if not await cache.touch(cache_key, exptime=max_exptime):
                await cache.set(cache_key, payload, exptime=max_exptime)
        except aiomcache.exceptions.ClientException:
            log.exception("Failed to put %d bytes in memcached", len(payload))

    tasks = [store_repo_day(r, day, rdt)
             for (r, days) in repodays.items()
             for (day, rdt) in days.items()]
    chunk_size = 128
    for i in range(0, len(tasks), chunk_size):
        errors = await asyncio.gather(*tasks[i:i + chunk_size], return_exceptions=True)
        if any(errors):
            for e in errors:
                if e is not None:
                    raise e
