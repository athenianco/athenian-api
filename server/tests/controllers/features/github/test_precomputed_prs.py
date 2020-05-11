import dataclasses
from typing import Sequence
import uuid

import pandas as pd

from athenian.api.controllers.features.github.precomputed_prs import load_precomputed_done_times, \
    store_precomputed_done_times
from athenian.api.controllers.miners.github.pull_request import Fallback, MinedPullRequest, \
    PullRequestTimes
from athenian.api.controllers.miners.github.release import matched_by_column
from athenian.api.controllers.settings import Match, ReleaseMatchSetting
from athenian.api.models.metadata.github import PullRequest, PullRequestCommit, Release


async def test_load_store_precomputed_done_smoke(pdb, pr_samples):
    samples = pr_samples(200)  # type: Sequence[PullRequestTimes]
    for i in range(1, 6):
        # merged but unreleased
        kwargs = dataclasses.asdict(samples[-i])
        kwargs["released"] = Fallback(None, None)
        samples[-i] = PullRequestTimes(**kwargs)
    for i in range(6, 11):
        # rejected
        kwargs = dataclasses.asdict(samples[-i])
        kwargs["released"] = kwargs["merged"] = Fallback(None, None)
        samples[-i] = PullRequestTimes(**kwargs)
    names = ["one", "two", "three"]
    settings = {"github.com/" + k: ReleaseMatchSetting("{{default}}", ".*", Match(i))
                for i, k in enumerate(names)}
    dummy_df = pd.DataFrame(columns=["user_login"])
    prs = [MinedPullRequest(
        pr={PullRequest.repository_full_name.key: names[i % len(names)],
            PullRequest.user_login.key: "xxx",
            PullRequest.merged_by_login.key: "yyy",
            PullRequest.node_id.key: uuid.uuid4().hex},
        release={matched_by_column: settings["github.com/" + names[i % len(names)]].match % 2,
                 Release.published_at.key: s.released.best,
                 Release.author.key: "zzz"},
        comments=dummy_df,
        commits=pd.DataFrame(columns=[
            PullRequestCommit.committer_login.key,
            PullRequestCommit.author_login.key,
        ]),
        reviews=dummy_df,
        review_comments=dummy_df,
        review_requests=dummy_df) for i, s in enumerate(samples)]
    await store_precomputed_done_times(prs, samples, settings, pdb)
    released_ats = sorted((t.released.best, i) for i, t in enumerate(samples[:-10]))
    time_from = released_ats[len(released_ats) // 2][0]
    time_to = released_ats[-1][0]
    n = len(released_ats) - len(released_ats) // 2 + \
        sum(1 for s in samples[-10:-5] if s.closed.best >= time_from)
    loaded_prs = await load_precomputed_done_times(time_from, time_to, names, [], settings, pdb)
    assert len(loaded_prs) == n
    true_prs = {prs[i].pr[PullRequest.node_id.key]: samples[i] for _, i in released_ats[-n:]}
    for i, s in enumerate(samples[-10:-5]):
        if s.closed.best >= time_from:
            true_prs[prs[-10 + i].pr[PullRequest.node_id.key]] = s
    diff_keys = set(loaded_prs) - set(true_prs)
    assert not diff_keys
    for k, load_value in loaded_prs.items():
        assert load_value == true_prs[k], k


async def test_load_store_precomputed_done_filters(pr_samples, pdb):
    samples = pr_samples(102)  # type: Sequence[PullRequestTimes]
    names = ["one", "two", "three"]
    settings = {"github.com/" + k: ReleaseMatchSetting("{{default}}", ".*", Match(i))
                for i, k in enumerate(names)}
    dummy_df = pd.DataFrame(columns=["user_login"])
    prs = [MinedPullRequest(
        pr={PullRequest.repository_full_name.key: names[i % len(names)],
            PullRequest.user_login.key: ["xxx", "wow"][i % 2],
            PullRequest.merged_by_login.key: "yyy",
            PullRequest.node_id.key: uuid.uuid4().hex},
        release={matched_by_column: settings["github.com/" + names[i % len(names)]].match % 2,
                 Release.published_at.key: s.released.best,
                 Release.author.key: ["foo", "zzz"][i % 2]},
        comments=dummy_df,
        commits=pd.DataFrame(columns=[
            PullRequestCommit.committer_login.key,
            PullRequestCommit.author_login.key,
        ]),
        reviews=dummy_df,
        review_comments=dummy_df,
        review_requests=dummy_df) for i, s in enumerate(samples)]
    await store_precomputed_done_times(prs, samples, settings, pdb)
    time_from = min(s.created.best for s in samples)
    time_to = max(s.max_timestamp() for s in samples)
    loaded_prs = await load_precomputed_done_times(time_from, time_to, ["one"], [], settings, pdb)
    assert set(loaded_prs) == {pr.pr[PullRequest.node_id.key] for pr in prs[::3]}
    loaded_prs = await load_precomputed_done_times(
        time_from, time_to, names, ["wow", "zzz"], settings, pdb)
    assert set(loaded_prs) == {pr.pr[PullRequest.node_id.key] for pr in prs[1::2]}
