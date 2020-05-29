import dataclasses
from datetime import datetime, timedelta
from typing import Sequence
import uuid

import pandas as pd

from athenian.api.controllers.miners.github.precomputed_prs import \
    load_precomputed_done_candidates, load_precomputed_done_times, store_precomputed_done_times
from athenian.api.controllers.miners.github.pull_request import Fallback, MinedPullRequest, \
    PullRequestTimes
from athenian.api.controllers.miners.github.release import matched_by_column
from athenian.api.controllers.miners.pull_request_list_item import ParticipationKind
from athenian.api.controllers.settings import Match, ReleaseMatchSetting
from athenian.api.models.metadata.github import PullRequest, PullRequestCommit, Release


def gen_dummy_df(dt: datetime) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [["xxx", dt, dt]], columns=["user_login", "created_at", "submitted_at"])


async def test_load_store_precomputed_done_smoke(mdb, pdb, pr_samples):
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
    prs = [MinedPullRequest(
        pr={PullRequest.repository_full_name.key: names[i % len(names)],
            PullRequest.user_login.key: "xxx",
            PullRequest.merged_by_login.key: "yyy",
            PullRequest.node_id.key: uuid.uuid4().hex},
        release={matched_by_column: settings["github.com/" + names[i % len(names)]].match % 2,
                 Release.author.key: "zzz"},
        comments=gen_dummy_df(s.first_comment_on_first_review.best),
        commits=pd.DataFrame.from_records(
            [["zzz", "zzz", s.first_commit.best]],
            columns=[
                PullRequestCommit.committer_login.key,
                PullRequestCommit.author_login.key,
                PullRequestCommit.committed_date.key,
            ],
        ),
        reviews=gen_dummy_df(s.first_comment_on_first_review.best),
        review_comments=gen_dummy_df(s.first_comment_on_first_review.best),
        review_requests=gen_dummy_df(s.first_review_request.best)) for i, s in enumerate(samples)]
    await store_precomputed_done_times(prs, samples, settings, mdb, pdb, None)
    # we should not crash on repeat
    await store_precomputed_done_times(prs, samples, settings, mdb, pdb, None)
    released_ats = sorted((t.released.best, i) for i, t in enumerate(samples[:-10]))
    time_from = released_ats[len(released_ats) // 2][0]
    time_to = released_ats[-1][0]
    n = len(released_ats) - len(released_ats) // 2 + \
        sum(1 for s in samples[-10:-5] if s.closed.best >= time_from)
    loaded_prs = await load_precomputed_done_times(
        time_from, time_to, names, {}, False, settings, mdb, pdb, None)
    assert len(loaded_prs) == n
    true_prs = {prs[i].pr[PullRequest.node_id.key]: samples[i] for _, i in released_ats[-n:]}
    for i, s in enumerate(samples[-10:-5]):
        if s.closed.best >= time_from:
            true_prs[prs[-10 + i].pr[PullRequest.node_id.key]] = s
    diff_keys = set(loaded_prs) - set(true_prs)
    assert not diff_keys
    for k, load_value in loaded_prs.items():
        assert load_value == true_prs[k], k


async def test_load_store_precomputed_done_filters(pr_samples, mdb, pdb, cache):
    samples = pr_samples(102)  # type: Sequence[PullRequestTimes]
    names = ["one", "two", "three"]
    settings = {"github.com/" + k: ReleaseMatchSetting("{{default}}", ".*", Match(i))
                for i, k in enumerate(names)}
    prs = [MinedPullRequest(
        pr={PullRequest.repository_full_name.key: names[i % len(names)],
            PullRequest.user_login.key: ["xxx", "wow"][i % 2],
            PullRequest.merged_by_login.key: "yyy",
            PullRequest.node_id.key: uuid.uuid4().hex},
        release={matched_by_column: settings["github.com/" + names[i % len(names)]].match % 2,
                 Release.author.key: ["foo", "zzz"][i % 2]},
        comments=gen_dummy_df(s.first_comment_on_first_review.best),
        commits=pd.DataFrame.from_records(
            [["yyy", "yyy", s.first_commit.best]],
            columns=[
                PullRequestCommit.committer_login.key,
                PullRequestCommit.author_login.key,
                PullRequestCommit.committed_date.key,
            ],
        ),
        reviews=gen_dummy_df(s.first_comment_on_first_review.best),
        review_comments=gen_dummy_df(s.first_comment_on_first_review.best),
        review_requests=gen_dummy_df(s.first_review_request.best))
        for i, s in enumerate(samples)]
    await store_precomputed_done_times(prs, samples, settings, mdb, pdb, cache)
    time_from = min(s.created.best for s in samples)
    time_to = max(s.max_timestamp() for s in samples)
    loaded_prs = await load_precomputed_done_times(
        time_from, time_to, ["one"], {}, False, settings, mdb, pdb, cache)
    assert set(loaded_prs) == {pr.pr[PullRequest.node_id.key] for pr in prs[::3]}
    loaded_prs = await load_precomputed_done_times(
        time_from, time_to, names, {ParticipationKind.AUTHOR: {"wow"},
                                    ParticipationKind.RELEASER: {"zzz"}},
        False, settings, mdb, pdb, cache)
    assert set(loaded_prs) == {pr.pr[PullRequest.node_id.key] for pr in prs[1::2]}
    loaded_prs = await load_precomputed_done_times(
        time_from, time_to, names, {ParticipationKind.COMMIT_AUTHOR: {"yyy"}},
        False, settings, mdb, pdb, cache)
    assert len(loaded_prs) == len(prs)


async def test_load_store_precomputed_done_match_by(pr_samples, mdb, pdb, cache):
    samples, prs, settings = _gen_one_pr(pr_samples)
    await store_precomputed_done_times(prs, samples, settings, mdb, pdb, cache)
    time_from = samples[0].created.best - timedelta(days=365)
    time_to = samples[0].released.best + timedelta(days=1)
    loaded_prs = await load_precomputed_done_times(
        time_from, time_to, ["src-d/go-git"], {}, False, settings, mdb, pdb, cache)
    assert len(loaded_prs) == 1
    settings = {
        "github.com/src-d/go-git": ReleaseMatchSetting("master", ".*", Match.branch),
    }
    loaded_prs = await load_precomputed_done_times(
        time_from, time_to, ["src-d/go-git"], {}, False, settings, mdb, pdb, cache)
    assert len(loaded_prs) == 1
    settings = {
        "github.com/src-d/go-git": ReleaseMatchSetting("nope", ".*", Match.tag_or_branch),
    }
    loaded_prs = await load_precomputed_done_times(
        time_from, time_to, ["src-d/go-git"], {}, False, settings, mdb, pdb, cache)
    assert len(loaded_prs) == 0
    settings = {
        "github.com/src-d/go-git": ReleaseMatchSetting("{{default}}", ".*", Match.tag),
    }
    loaded_prs = await load_precomputed_done_times(
        time_from, time_to, ["src-d/go-git"], {}, False, settings, mdb, pdb, cache)
    assert len(loaded_prs) == 0
    prs[0].release[matched_by_column] = 1
    await store_precomputed_done_times(prs, samples, settings, mdb, pdb, cache)
    loaded_prs = await load_precomputed_done_times(
        time_from, time_to, ["src-d/go-git"], {}, False, settings, mdb, pdb, cache)
    assert len(loaded_prs) == 1
    settings = {
        "github.com/src-d/go-git": ReleaseMatchSetting("{{default}}", "xxx", Match.tag),
    }
    loaded_prs = await load_precomputed_done_times(
        time_from, time_to, ["src-d/go-git"], {}, False, settings, mdb, pdb, cache)
    assert len(loaded_prs) == 0


async def test_load_store_precomputed_done_exclude_inactive(pr_samples, mdb, pdb, cache):
    while True:
        samples = pr_samples(2)  # type: Sequence[PullRequestTimes]
        samples = sorted(samples, key=lambda s: s.first_comment_on_first_review.best)
        deltas = [(samples[1].first_comment_on_first_review.best -
                   samples[0].first_comment_on_first_review.best),
                  samples[0].first_comment_on_first_review.best - samples[1].created.best,
                  samples[1].created.best - samples[0].created.best]
        if all(d > timedelta(days=2) for d in deltas):
            break
    settings = {"github.com/one": ReleaseMatchSetting("{{default}}", ".*", Match.tag)}
    prs = [MinedPullRequest(
        pr={PullRequest.repository_full_name.key: "one",
            PullRequest.user_login.key: "xxx",
            PullRequest.merged_by_login.key: "yyy",
            PullRequest.node_id.key: uuid.uuid4().hex},
        release={matched_by_column: settings["github.com/one"].match,
                 Release.author.key: "zzz"},
        comments=gen_dummy_df(s.first_comment_on_first_review.best),
        commits=pd.DataFrame.from_records(
            [["yyy", "yyy", s.first_comment_on_first_review.best]],
            columns=[
                PullRequestCommit.committer_login.key,
                PullRequestCommit.author_login.key,
                PullRequestCommit.committed_date.key,
            ],
        ),
        reviews=gen_dummy_df(s.first_comment_on_first_review.best),
        review_comments=gen_dummy_df(s.first_comment_on_first_review.best),
        review_requests=gen_dummy_df(s.first_comment_on_first_review.best))
        for s in samples]
    await store_precomputed_done_times(prs, samples, settings, mdb, pdb, cache)
    time_from = samples[1].created.best + timedelta(days=1)
    time_to = samples[0].first_comment_on_first_review.best
    loaded_prs = await load_precomputed_done_times(
        time_from, time_to, ["one"], {}, True, settings, mdb, pdb, cache)
    assert len(loaded_prs) == 1
    assert loaded_prs[prs[0].pr[PullRequest.node_id.key]] == samples[0]
    time_from = samples[1].created.best - timedelta(days=1)
    time_to = samples[1].created.best + timedelta(seconds=1)
    loaded_prs = await load_precomputed_done_times(
        time_from, time_to, ["one"], {}, True, settings, mdb, pdb, cache)
    assert len(loaded_prs) == 1
    assert loaded_prs[prs[1].pr[PullRequest.node_id.key]] == samples[1]


def _gen_one_pr(pr_samples):
    samples = pr_samples(1)  # type: Sequence[PullRequestTimes]
    s = samples[0]
    settings = {
        "github.com/src-d/go-git": ReleaseMatchSetting("{{default}}", ".*", Match.tag_or_branch),
    }
    prs = [MinedPullRequest(
        pr={PullRequest.repository_full_name.key: "src-d/go-git",
            PullRequest.user_login.key: "xxx",
            PullRequest.merged_by_login.key: "yyy",
            PullRequest.node_id.key: uuid.uuid4().hex},
        release={matched_by_column: 0,
                 Release.author.key: "zzz"},
        comments=gen_dummy_df(s.first_comment_on_first_review.best),
        commits=pd.DataFrame.from_records(
            [["zzz", "zzz", s.first_commit.best]],
            columns=[
                PullRequestCommit.committer_login.key,
                PullRequestCommit.author_login.key,
                PullRequestCommit.committed_date.key,
            ],
        ),
        reviews=gen_dummy_df(s.first_comment_on_first_review.best),
        review_comments=gen_dummy_df(s.first_comment_on_first_review.best),
        review_requests=gen_dummy_df(s.first_review_request.best))]
    return samples, prs, settings


async def test_load_precomputed_done_candidates_smoke(pr_samples, mdb, pdb, cache):
    samples, prs, settings = _gen_one_pr(pr_samples)
    await store_precomputed_done_times(prs, samples, settings, mdb, pdb, cache)
    time_from = samples[0].created.best
    time_to = samples[0].released.best
    loaded_prs = await load_precomputed_done_candidates(
        time_from, time_to, ["one"], settings, mdb, pdb, cache)
    assert len(loaded_prs) == 0
    loaded_prs = await load_precomputed_done_candidates(
        time_from, time_to, ["src-d/go-git"], settings, mdb, pdb, cache)
    assert loaded_prs == {prs[0].pr[PullRequest.node_id.key]}
    loaded_prs = await load_precomputed_done_candidates(
        time_from, time_from, ["src-d/go-git"], settings, mdb, pdb, cache)
    assert len(loaded_prs) == 0
