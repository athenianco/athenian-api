import dataclasses
from itertools import takewhile
from typing import Sequence
import uuid

import pytest

from athenian.api.controllers.features.cached_released import load_cached_released_times, \
    store_cached_released_times
from athenian.api.controllers.miners.github.pull_request import Fallback, MinedPullRequest, \
    PullRequestTimes
from athenian.api.models.metadata.github import PullRequest


@pytest.mark.parametrize("size", [100, 200, 1200])
async def test_load_store_cached_released_times(cache, pr_samples, size):
    samples = pr_samples(size + 5)  # type: Sequence[PullRequestTimes]
    names = ["one", "two", "three"]
    prs = [MinedPullRequest({PullRequest.repository_full_name.key: names[i % len(names)],
                             PullRequest.node_id.key: str(uuid.uuid4())},
                            None, None, None, None, None, None)
           for i in range(len(samples))]
    for i in range(1, 6):
        kwargs = dataclasses.asdict(samples[-i])
        kwargs["released"] = Fallback(None, None)
        samples[-i] = PullRequestTimes(**kwargs)
    await store_cached_released_times(prs, samples, cache)
    assert size / 2 < len(cache.mem) < size
    released_ats = sorted((t.released.best.date(), i) for i, t in enumerate(samples[:-5]))
    date_from = released_ats[0][0]
    date_to = released_ats[size // 2][0]
    n = size // 2 + 1 - sum(1 for _ in takewhile(lambda p: p[0] == date_to,
                                                 released_ats[size // 2::-1]))
    loaded_prs = await load_cached_released_times(date_from, date_to, names, cache)
    assert len(loaded_prs) == n
    true_prs = {prs[i].pr[PullRequest.node_id.key]: samples[i] for _, i in released_ats[:n]}
    diff_keys = set(loaded_prs) - set(true_prs)
    assert not diff_keys
    for k, load_value in loaded_prs.items():
        assert load_value == true_prs[k], k
