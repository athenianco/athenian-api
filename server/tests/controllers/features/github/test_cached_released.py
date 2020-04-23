import dataclasses
from datetime import timedelta
from itertools import takewhile
from typing import Sequence
import uuid

import pytest

from athenian.api.controllers.features.cached_released import load_cached_released_times, \
    store_cached_released_times
from athenian.api.controllers.miners.github.pull_request import Fallback, PullRequestTimes
from athenian.api.models.metadata.github import PullRequest


@pytest.mark.parametrize("size", [100, 200, 1200])
async def test_load_store_cached_released_times(cache, pr_samples, size):
    samples = pr_samples(size + 5)  # type: Sequence[PullRequestTimes]
    for i in range(1, 6):
        kwargs = dataclasses.asdict(samples[-i])
        kwargs["released"] = Fallback(None, None)
        samples[-i] = PullRequestTimes(**kwargs)
    names = ["one", "two", "three"]
    prs = [({PullRequest.repository_full_name.key: names[i % len(names)],
             PullRequest.user_login.key: "xxx",
             PullRequest.node_id.key: str(uuid.uuid4())}, s)
           for i, s in enumerate(samples)]
    await store_cached_released_times(prs, cache)
    assert size / 2 < len(cache.mem) <= size
    released_ats = sorted((t.released.best.date(), i) for i, t in enumerate(samples[:-5]))
    date_from = released_ats[0][0]
    date_to = released_ats[size // 2][0]
    n = size // 2 + 1 - sum(1 for _ in takewhile(lambda p: p[0] == date_to,
                                                 released_ats[size // 2::-1]))
    loaded_prs = await load_cached_released_times(date_from, date_to, names, [], cache)
    assert len(loaded_prs) == n
    true_prs = {prs[i][0][PullRequest.node_id.key]: prs[i][1] for _, i in released_ats[:n]}
    diff_keys = set(loaded_prs) - set(true_prs)
    assert not diff_keys
    for k, load_value in loaded_prs.items():
        assert load_value == true_prs[k], k


async def test_load_store_cached_released_filters(pr_samples, cache):
    samples = pr_samples(100)
    names = ["one", "two", "three"]
    devs = ["xxx", "yyy"]
    prs = [({PullRequest.repository_full_name.key: names[i % len(names)],
             PullRequest.user_login.key: devs[i % 2],
             PullRequest.node_id.key: str(uuid.uuid4())}, s)
           for i, s in enumerate(samples)]
    prs_by_key = {p[0][PullRequest.node_id.key]: p for p in prs}
    await store_cached_released_times(prs, cache)
    released_ats = sorted((t.released.best.date(), i) for i, t in enumerate(samples[:-5]))
    date_from = released_ats[0][0]
    date_to = released_ats[-1][0] + timedelta(days=1)
    loaded_prs = await load_cached_released_times(date_from, date_to, ["one"], ["xxx"], cache)
    for k, load_value in loaded_prs.items():
        pr = prs_by_key[k]
        assert pr[0][PullRequest.repository_full_name.key] == "one"
        assert pr[0][PullRequest.user_login.key] == "xxx"
        assert load_value == pr[1]
    assert len(loaded_prs) == 17
