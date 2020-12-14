import asyncio
import dataclasses
from datetime import datetime, timedelta, timezone
from itertools import repeat
import math
import pickle
from typing import Sequence
import uuid

import pandas as pd
import pytest
from sqlalchemy import and_, select, update

from athenian.api.async_utils import read_sql_query
from athenian.api.controllers.features.entries import calc_pull_request_facts_github
from athenian.api.controllers.features.github.pull_request_filter import _fetch_pull_requests
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.precomputed_prs import \
    discover_inactive_merged_unreleased_prs, load_merged_unreleased_pull_request_facts, \
    load_open_pull_request_facts, load_open_pull_request_facts_unfresh, \
    load_precomputed_done_candidates, load_precomputed_done_facts_filters, \
    load_precomputed_done_facts_reponums, load_precomputed_pr_releases, \
    store_merged_unreleased_pull_request_facts, store_open_pull_request_facts, \
    store_precomputed_done_facts, update_unreleased_prs
from athenian.api.controllers.miners.github.release_load import load_releases
from athenian.api.controllers.miners.github.release_match import map_prs_to_releases
from athenian.api.controllers.miners.github.released_pr import matched_by_column, \
    new_released_prs_df
from athenian.api.controllers.miners.types import MinedPullRequest, PRParticipationKind, \
    PullRequestFacts
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.metadata.github import Branch, PullRequest, PullRequestCommit, Release
from athenian.api.models.precomputed.models import GitHubMergedPullRequestFacts, \
    GitHubOpenPullRequestFacts


def gen_dummy_df(dt: datetime) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [["xxx", dt, dt]], columns=["user_login", "created_at", "submitted_at"])


async def test_load_store_precomputed_done_smoke(pdb, pr_samples):
    samples = pr_samples(200)  # type: Sequence[PullRequestFacts]
    for i in range(1, 6):
        # merged but unreleased
        kwargs = dataclasses.asdict(samples[-i])
        kwargs["released"] = None
        samples[-i] = PullRequestFacts(**kwargs)
    for i in range(6, 11):
        # rejected
        kwargs = dataclasses.asdict(samples[-i])
        kwargs["released"] = kwargs["merged"] = None
        samples[-i] = PullRequestFacts(**kwargs)
    names = ["one", "two", "three"]
    settings = {"github.com/" + k: ReleaseMatchSetting("{{default}}", ".*", ReleaseMatch(i))
                for i, k in enumerate(names)}
    default_branches = {k: "master" for k in names}
    prs = [MinedPullRequest(
        pr={PullRequest.created_at.key: s.created,
            PullRequest.repository_full_name.key: names[i % len(names)],
            PullRequest.user_login.key: "xxx",
            PullRequest.merged_by_login.key: "yyy",
            PullRequest.number.key: i + 1,
            PullRequest.node_id.key: uuid.uuid4().hex},
        release={matched_by_column: settings["github.com/" + names[i % len(names)]].match % 2,
                 Release.author.key: "zzz",
                 Release.url.key: "https://release",
                 Release.id.key: "MD%d" % i},
        comments=gen_dummy_df(s.first_comment_on_first_review),
        commits=pd.DataFrame.from_records(
            [["zzz", "zzz", s.first_commit]],
            columns=[
                PullRequestCommit.committer_login.key,
                PullRequestCommit.author_login.key,
                PullRequestCommit.committed_date.key,
            ],
        ),
        reviews=gen_dummy_df(s.first_comment_on_first_review),
        review_comments=gen_dummy_df(s.first_comment_on_first_review),
        review_requests=gen_dummy_df(s.first_review_request),
        labels=pd.DataFrame.from_records(([["bug"]], [["feature"]])[i % 2], columns=["name"]),
        jiras=pd.DataFrame(),
    ) for i, s in enumerate(samples)]
    await store_precomputed_done_facts(
        prs, [(names[i % len(names)], s) for i, s in enumerate(samples)], default_branches,
        settings, pdb)
    # we should not crash on repeat
    await store_precomputed_done_facts(
        prs, [(names[i % len(names)], s) for i, s in enumerate(samples)], default_branches,
        settings, pdb)
    released_ats = sorted((t.released, i) for i, t in enumerate(samples[:-10]))
    time_from = released_ats[len(released_ats) // 2][0]
    time_to = released_ats[-1][0]
    n = len(released_ats) - len(released_ats) // 2 + \
        sum(1 for s in samples[-10:-5] if s.closed >= time_from)
    loaded_prs, _ = await load_precomputed_done_facts_filters(
        time_from, time_to, names, {}, LabelFilter.empty(), default_branches,
        False, settings, pdb)
    assert len(loaded_prs) == n
    true_prs = {prs[i].pr[PullRequest.node_id.key]: samples[i] for _, i in released_ats[-n:]}
    for i, s in enumerate(samples[-10:-5]):
        if s.closed >= time_from:
            true_prs[prs[-10 + i].pr[PullRequest.node_id.key]] = s
    diff_keys = set(loaded_prs) - set(true_prs)
    assert not diff_keys
    for k, (r, load_value) in loaded_prs.items():
        assert load_value == true_prs[k], k
        assert r in names


async def test_load_store_precomputed_done_filters(pr_samples, pdb):
    samples = pr_samples(102)  # type: Sequence[PullRequestFacts]
    names = ["one", "two", "three"]
    settings = {"github.com/" + k: ReleaseMatchSetting("{{default}}", ".*", ReleaseMatch(i))
                for i, k in enumerate(names)}
    default_branches = {k: "master" for k in names}
    prs = [MinedPullRequest(
        pr={PullRequest.created_at.key: s.created,
            PullRequest.repository_full_name.key: names[i % len(names)],
            PullRequest.user_login.key: ["xxx", "wow"][i % 2],
            PullRequest.merged_by_login.key: "yyy",
            PullRequest.number.key: i + 1,
            PullRequest.node_id.key: uuid.uuid4().hex},
        release={matched_by_column: settings["github.com/" + names[i % len(names)]].match % 2,
                 Release.author.key: ["foo", "zzz"][i % 2],
                 Release.url.key: "https://release",
                 Release.id.key: "MD%d" % i},
        comments=gen_dummy_df(s.first_comment_on_first_review),
        commits=pd.DataFrame.from_records(
            [["yyy", "yyy", s.first_commit]],
            columns=[
                PullRequestCommit.committer_login.key,
                PullRequestCommit.author_login.key,
                PullRequestCommit.committed_date.key,
            ],
        ),
        reviews=gen_dummy_df(s.first_comment_on_first_review),
        review_comments=gen_dummy_df(s.first_comment_on_first_review),
        review_requests=gen_dummy_df(s.first_review_request),
        labels=pd.DataFrame.from_records(([["bug"]],
                                          [["feature"]],
                                          [["bug"], ["bad"]],
                                          [["feature"], ["bad"]])[i % 4], columns=["name"]),
        jiras=pd.DataFrame(),
    ) for i, s in enumerate(samples)]
    await store_precomputed_done_facts(
        prs, [(names[i % len(names)], s) for i, s in enumerate(samples)], default_branches,
        settings, pdb)
    time_from = min(s.created for s in samples)
    time_to = max(s.max_timestamp() for s in samples)
    loaded_prs, _ = await load_precomputed_done_facts_filters(
        time_from, time_to, ["one"], {}, LabelFilter.empty(), default_branches,
        False, settings, pdb)
    assert set(loaded_prs) == {pr.pr[PullRequest.node_id.key] for pr in prs[::3]}
    loaded_prs, _ = await load_precomputed_done_facts_filters(
        time_from, time_to, names, {PRParticipationKind.AUTHOR: {"wow"},
                                    PRParticipationKind.RELEASER: {"zzz"}},
        LabelFilter.empty(), default_branches, False, settings, pdb)
    assert set(loaded_prs) == {pr.pr[PullRequest.node_id.key] for pr in prs[1::2]}
    loaded_prs, _ = await load_precomputed_done_facts_filters(
        time_from, time_to, names, {PRParticipationKind.COMMIT_AUTHOR: {"yyy"}},
        LabelFilter.empty(), default_branches, False, settings, pdb)
    assert len(loaded_prs) == len(prs)
    loaded_prs, _ = await load_precomputed_done_facts_filters(
        time_from, time_to, names, {}, LabelFilter({"bug", "xxx"}, set()),
        default_branches, False, settings, pdb)
    assert len(loaded_prs) == len(prs) / 2
    loaded_prs, _ = await load_precomputed_done_facts_filters(
        time_from, time_to, names, {}, LabelFilter({"bug"}, {"bad"}),
        default_branches, False, settings, pdb)
    assert len(loaded_prs) == int(math.ceil(len(prs) / 4.0))


async def test_load_store_precomputed_done_match_by(pr_samples, default_branches, pdb):
    samples, prs, settings = _gen_one_pr(pr_samples)
    await store_precomputed_done_facts(
        prs, zip(repeat("src-d/go-git"), samples), default_branches, settings, pdb)
    time_from = samples[0].created - timedelta(days=365)
    time_to = samples[0].released + timedelta(days=1)
    loaded_prs, _ = await load_precomputed_done_facts_filters(
        time_from, time_to, ["src-d/go-git"], {}, LabelFilter.empty(),
        default_branches, False, settings, pdb)
    assert len(loaded_prs) == 1
    settings = {
        "github.com/src-d/go-git": ReleaseMatchSetting("master", ".*", ReleaseMatch.branch),
    }
    loaded_prs, _ = await load_precomputed_done_facts_filters(
        time_from, time_to, ["src-d/go-git"], {}, LabelFilter.empty(),
        default_branches, False, settings, pdb)
    assert len(loaded_prs) == 1
    settings = {
        "github.com/src-d/go-git": ReleaseMatchSetting("nope", ".*", ReleaseMatch.tag_or_branch),
    }
    loaded_prs, ambiguous = await load_precomputed_done_facts_filters(
        time_from, time_to, ["src-d/go-git"], {}, LabelFilter.empty(),
        default_branches, False, settings, pdb)
    assert len(loaded_prs) == 0
    assert len(ambiguous) == 0
    settings = {
        "github.com/src-d/go-git": ReleaseMatchSetting("{{default}}", ".*", ReleaseMatch.tag),
    }
    loaded_prs, _ = await load_precomputed_done_facts_filters(
        time_from, time_to, ["src-d/go-git"], {}, LabelFilter.empty(),
        default_branches, False, settings, pdb)
    assert len(loaded_prs) == 0
    prs[0].release[matched_by_column] = 1
    await store_precomputed_done_facts(
        prs, zip(repeat("src-d/go-git"), samples), default_branches, settings, pdb)
    loaded_prs, _ = await load_precomputed_done_facts_filters(
        time_from, time_to, ["src-d/go-git"], {}, LabelFilter.empty(),
        default_branches, False, settings, pdb)
    assert len(loaded_prs) == 1
    settings = {
        "github.com/src-d/go-git": ReleaseMatchSetting("{{default}}", "xxx", ReleaseMatch.tag),
    }
    loaded_prs, _ = await load_precomputed_done_facts_filters(
        time_from, time_to, ["src-d/go-git"], {}, LabelFilter.empty(),
        default_branches, False, settings, pdb)
    assert len(loaded_prs) == 0


async def test_load_store_precomputed_done_exclude_inactive(pr_samples, default_branches, pdb):
    while True:
        samples = pr_samples(2)  # type: Sequence[PullRequestFacts]
        samples = sorted(samples, key=lambda s: s.first_comment_on_first_review)
        deltas = [(samples[1].first_comment_on_first_review -
                   samples[0].first_comment_on_first_review),
                  samples[0].first_comment_on_first_review - samples[1].created,
                  samples[1].created - samples[0].created]
        if all(d > timedelta(days=2) for d in deltas):
            break
    settings = {"github.com/one": ReleaseMatchSetting("{{default}}", ".*", ReleaseMatch.tag)}
    prs = [MinedPullRequest(
        pr={PullRequest.created_at.key: s.created,
            PullRequest.repository_full_name.key: "one",
            PullRequest.user_login.key: "xxx",
            PullRequest.merged_by_login.key: "yyy",
            PullRequest.number.key: 777,
            PullRequest.node_id.key: uuid.uuid4().hex},
        release={matched_by_column: settings["github.com/one"].match,
                 Release.author.key: "zzz",
                 Release.url.key: "https://release",
                 Release.id.key: "MDwhatever="},
        comments=gen_dummy_df(s.first_comment_on_first_review),
        commits=pd.DataFrame.from_records(
            [["yyy", "yyy", s.first_comment_on_first_review]],
            columns=[
                PullRequestCommit.committer_login.key,
                PullRequestCommit.author_login.key,
                PullRequestCommit.committed_date.key,
            ],
        ),
        reviews=gen_dummy_df(s.first_comment_on_first_review),
        review_comments=gen_dummy_df(s.first_comment_on_first_review),
        review_requests=gen_dummy_df(s.first_comment_on_first_review),
        labels=pd.DataFrame.from_records([["bug"]], columns=["name"]),
        jiras=pd.DataFrame(),
    ) for s in samples]
    await store_precomputed_done_facts(
        prs, zip(repeat("one"), samples), default_branches, settings, pdb)
    time_from = samples[1].created + timedelta(days=1)
    time_to = samples[0].first_comment_on_first_review
    loaded_prs, _ = await load_precomputed_done_facts_filters(
        time_from, time_to, ["one"], {}, LabelFilter.empty(), default_branches,
        True, settings, pdb)
    assert len(loaded_prs) == 1
    assert loaded_prs[prs[0].pr[PullRequest.node_id.key]] == ("one", samples[0])
    time_from = samples[1].created - timedelta(days=1)
    time_to = samples[1].created + timedelta(seconds=1)
    loaded_prs, _ = await load_precomputed_done_facts_filters(
        time_from, time_to, ["one"], {}, LabelFilter.empty(), default_branches,
        True, settings, pdb)
    assert len(loaded_prs) == 1
    assert loaded_prs[prs[1].pr[PullRequest.node_id.key]] == ("one", samples[1])


async def test_load_precomputed_done_times_reponums_smoke(pr_samples, pdb):
    samples = pr_samples(12)  # type: Sequence[PullRequestFacts]
    names = ["one", "two", "three"]
    settings = {"github.com/" + k: ReleaseMatchSetting("{{default}}", ".*", ReleaseMatch(i))
                for i, k in enumerate(names)}
    default_branches = {k: "master" for k in names}
    prs = [MinedPullRequest(
        pr={PullRequest.created_at.key: s.created,
            PullRequest.repository_full_name.key: names[i % len(names)],
            PullRequest.user_login.key: ["xxx", "wow"][i % 2],
            PullRequest.merged_by_login.key: "yyy",
            PullRequest.number.key: i + 1,
            PullRequest.node_id.key: uuid.uuid4().hex},
        release={matched_by_column: settings["github.com/" + names[i % len(names)]].match % 2,
                 Release.author.key: ["foo", "zzz"][i % 2],
                 Release.url.key: "https://release",
                 Release.id.key: "MD%d" % i},
        comments=gen_dummy_df(s.first_comment_on_first_review),
        commits=pd.DataFrame.from_records(
            [["yyy", "yyy", s.first_commit]],
            columns=[
                PullRequestCommit.committer_login.key,
                PullRequestCommit.author_login.key,
                PullRequestCommit.committed_date.key,
            ],
        ),
        reviews=gen_dummy_df(s.first_comment_on_first_review),
        review_comments=gen_dummy_df(s.first_comment_on_first_review),
        review_requests=gen_dummy_df(s.first_review_request),
        labels=pd.DataFrame.from_records(([["bug"]], [["feature"]])[i % 2], columns=["name"]),
        jiras=pd.DataFrame(),
    ) for i, s in enumerate(samples)]
    await store_precomputed_done_facts(
        prs, [(names[i % len(names)], s) for i, s in enumerate(samples)], default_branches,
        settings, pdb)
    query1 = {"one": {pr.pr[PullRequest.number.key] for pr in prs
                      if pr.pr[PullRequest.repository_full_name.key] == "one"}}
    assert len(query1["one"]) == 4
    new_prs, _ = await load_precomputed_done_facts_reponums(
        query1, default_branches, settings, pdb)
    assert new_prs == {pr.pr[PullRequest.node_id.key]: ("one", s)
                       for pr, s in zip(prs, samples)
                       if pr.pr[PullRequest.repository_full_name.key] == "one"}
    query2 = {"one": set()}
    new_prs, _ = await load_precomputed_done_facts_reponums(
        query2, default_branches, settings, pdb)
    assert len(new_prs) == 0
    query3 = {"one": {100500}}
    new_prs, _ = await load_precomputed_done_facts_reponums(
        query3, default_branches, settings, pdb)
    assert len(new_prs) == 0


def _gen_one_pr(pr_samples):
    samples = pr_samples(1)  # type: Sequence[PullRequestFacts]
    s = samples[0]
    settings = {
        "github.com/src-d/go-git": ReleaseMatchSetting(
            "{{default}}", ".*", ReleaseMatch.tag_or_branch),
    }
    prs = [MinedPullRequest(
        pr={PullRequest.created_at.key: s.created,
            PullRequest.repository_full_name.key: "src-d/go-git",
            PullRequest.user_login.key: "xxx",
            PullRequest.merged_by_login.key: "yyy",
            PullRequest.number.key: 777,
            PullRequest.node_id.key: uuid.uuid4().hex},
        release={matched_by_column: ReleaseMatch.branch,
                 Release.author.key: "zzz",
                 Release.url.key: "https://release",
                 Release.id.key: "MDwhatever="},
        comments=gen_dummy_df(s.first_comment_on_first_review),
        commits=pd.DataFrame.from_records(
            [["zzz", "zzz", s.first_commit]],
            columns=[
                PullRequestCommit.committer_login.key,
                PullRequestCommit.author_login.key,
                PullRequestCommit.committed_date.key,
            ],
        ),
        reviews=gen_dummy_df(s.first_comment_on_first_review),
        review_comments=gen_dummy_df(s.first_comment_on_first_review),
        review_requests=gen_dummy_df(s.first_review_request),
        labels=pd.DataFrame.from_records([["bug"]], columns=["name"]),
        jiras=pd.DataFrame(),
    )]
    return samples, prs, settings


async def test_store_precomputed_done_facts_empty(pdb):
    await store_precomputed_done_facts([], [], None, None, pdb)


async def test_load_precomputed_done_candidates_smoke(pr_samples, default_branches, pdb):
    samples, prs, settings = _gen_one_pr(pr_samples)
    await store_precomputed_done_facts(
        prs, zip(repeat("src-d/go-git"), samples), default_branches, settings, pdb)
    time_from = samples[0].created
    time_to = samples[0].released
    loaded_prs, _ = await load_precomputed_done_candidates(
        time_from, time_to, ["one"], {"one": "master"}, settings, pdb)
    assert len(loaded_prs) == 0
    loaded_prs, _ = await load_precomputed_done_candidates(
        time_from, time_to, ["src-d/go-git"], default_branches, settings, pdb)
    assert loaded_prs == {prs[0].pr[PullRequest.node_id.key]}
    loaded_prs, _ = await load_precomputed_done_candidates(
        time_from, time_from, ["src-d/go-git"], default_branches, settings, pdb)
    assert len(loaded_prs) == 0


@with_defer
async def test_load_precomputed_pr_releases_smoke(pr_samples, default_branches, pdb, cache):
    samples, prs, settings = _gen_one_pr(pr_samples)
    await store_precomputed_done_facts(
        prs, zip(repeat("src-d/go-git"), samples), default_branches, settings, pdb)
    for i in range(2):
        released_prs = await load_precomputed_pr_releases(
            [pr.pr[PullRequest.node_id.key] for pr in prs],
            max(s.released for s in samples) + timedelta(days=1),
            {pr.pr[PullRequest.repository_full_name.key]: ReleaseMatch.branch for pr in prs},
            default_branches, settings, pdb if i == 0 else None, cache)
        await wait_deferred()
        for s, pr in zip(samples, prs):
            rpr = released_prs.loc[pr.pr[PullRequest.node_id.key]]
            for col in (Release.author.key, Release.url.key, Release.id.key, matched_by_column):
                assert rpr[col] == pr.release[col], i
            assert rpr[Release.published_at.key] == s.released, i
            assert rpr[Release.repository_full_name.key] == \
                pr.pr[PullRequest.repository_full_name.key], i


async def test_load_precomputed_pr_releases_time_to(pr_samples, default_branches, pdb):
    samples, prs, settings = _gen_one_pr(pr_samples)
    await store_precomputed_done_facts(
        prs, zip(repeat("src-d/go-git"), samples), default_branches, settings, pdb)
    released_prs = await load_precomputed_pr_releases(
        [pr.pr[PullRequest.node_id.key] for pr in prs],
        min(s.released for s in samples),
        {pr.pr[PullRequest.repository_full_name.key]: ReleaseMatch.branch for pr in prs},
        default_branches, settings, pdb, None)
    assert released_prs.empty


async def test_load_precomputed_pr_releases_release_mismatch(pr_samples, default_branches, pdb):
    samples, prs, settings = _gen_one_pr(pr_samples)
    await store_precomputed_done_facts(
        prs, zip(repeat("src-d/go-git"), samples), default_branches, settings, pdb)
    released_prs = await load_precomputed_pr_releases(
        [pr.pr[PullRequest.node_id.key] for pr in prs],
        max(s.released for s in samples) + timedelta(days=1),
        {pr.pr[PullRequest.repository_full_name.key]: ReleaseMatch.tag for pr in prs},
        default_branches, settings, pdb, None)
    assert released_prs.empty
    released_prs = await load_precomputed_pr_releases(
        [pr.pr[PullRequest.node_id.key] for pr in prs],
        max(s.released for s in samples) + timedelta(days=1),
        {pr.pr[PullRequest.repository_full_name.key]: ReleaseMatch.branch for pr in prs},
        {"src-d/go-git": "xxx"}, settings, pdb, None)
    assert released_prs.empty


async def test_load_precomputed_pr_releases_tag(pr_samples, default_branches, pdb):
    samples, prs, settings = _gen_one_pr(pr_samples)
    prs[0].release[matched_by_column] = ReleaseMatch.tag
    await store_precomputed_done_facts(
        prs, zip(repeat("src-d/go-git"), samples), default_branches, settings, pdb)
    released_prs = await load_precomputed_pr_releases(
        [pr.pr[PullRequest.node_id.key] for pr in prs],
        max(s.released for s in samples) + timedelta(days=1),
        {pr.pr[PullRequest.repository_full_name.key]: ReleaseMatch.tag for pr in prs},
        {}, settings, pdb, None)
    assert len(released_prs) == len(prs)
    released_prs = await load_precomputed_pr_releases(
        [pr.pr[PullRequest.node_id.key] for pr in prs],
        max(s.released for s in samples) + timedelta(days=1),
        {pr.pr[PullRequest.repository_full_name.key]: ReleaseMatch.tag for pr in prs},
        {}, {"github.com/src-d/go-git": ReleaseMatchSetting(
            tags="v.*", branches="", match=ReleaseMatch.tag),
        }, pdb, None)
    assert released_prs.empty


@with_defer
async def test_discover_update_unreleased_prs_smoke(
        mdb, pdb, default_branches, release_match_setting_tag):
    prs = await read_sql_query(
        select([PullRequest]).where(and_(PullRequest.number.in_(range(1000, 1010)),
                                         PullRequest.merged_at.isnot(None))),
        mdb, PullRequest, index=PullRequest.node_id.key)
    prs[prs[PullRequest.merged_at.key].isnull()] = datetime.now(tz=timezone.utc)
    utc = timezone.utc
    releases, matched_bys = await load_releases(
        (6366825,), ["src-d/go-git"], None, default_branches,
        datetime(2018, 9, 1, tzinfo=utc),
        datetime(2018, 11, 1, tzinfo=utc),
        release_match_setting_tag,
        mdb, pdb, None)
    assert len(releases) == 2
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
    empty_rdf = new_released_prs_df()
    await update_unreleased_prs(
        prs, empty_rdf, datetime(2018, 11, 1, tzinfo=utc), {},
        matched_bys, default_branches, release_match_setting_tag, pdb, asyncio.Event())
    releases, matched_bys = await load_releases(
        (6366825,), ["src-d/go-git"], None, default_branches,
        datetime(2018, 11, 1, tzinfo=utc),
        datetime(2018, 11, 20, tzinfo=utc),
        release_match_setting_tag,
        mdb, pdb, None)
    assert len(releases) == 1
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
    await update_unreleased_prs(
        prs, empty_rdf, datetime(2018, 11, 20, tzinfo=utc), {},
        matched_bys, default_branches, release_match_setting_tag, pdb, asyncio.Event())
    unreleased_prs = await load_merged_unreleased_pull_request_facts(
        prs, datetime(2018, 11, 20, tzinfo=utc), LabelFilter.empty(), matched_bys,
        default_branches, release_match_setting_tag, pdb)
    assert len(unreleased_prs) == 0
    await pdb.execute(update(GitHubMergedPullRequestFacts).values({
        GitHubMergedPullRequestFacts.data: pickle.dumps("fake"),
        GitHubMergedPullRequestFacts.updated_at: datetime.now(timezone.utc),
    }))
    unreleased_prs = await load_merged_unreleased_pull_request_facts(
        prs, datetime(2018, 11, 20, tzinfo=utc), LabelFilter.empty(), matched_bys,
        default_branches, release_match_setting_tag, pdb)
    assert set(prs.index) == set(unreleased_prs)
    releases, matched_bys = await load_releases(
        (6366825,), ["src-d/go-git"], None, default_branches,
        datetime(2018, 9, 1, tzinfo=utc),
        datetime(2018, 11, 1, tzinfo=utc),
        release_match_setting_tag,
        mdb, pdb, None)
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
    unreleased_prs = await load_merged_unreleased_pull_request_facts(
        prs, datetime(2018, 11, 1, tzinfo=utc), LabelFilter.empty(), matched_bys, default_branches,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="", tags="v.*", match=ReleaseMatch.tag)},
        pdb)
    assert len(unreleased_prs) == 0
    releases, matched_bys = await load_releases(
        (6366825,), ["src-d/go-git"], None, default_branches,
        datetime(2019, 1, 29, tzinfo=utc),
        datetime(2019, 2, 1, tzinfo=utc),
        release_match_setting_tag,
        mdb, pdb, None)
    assert len(releases) == 2
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
    unreleased_prs = await load_merged_unreleased_pull_request_facts(
        prs, datetime(2019, 2, 1, tzinfo=utc), LabelFilter.empty(), matched_bys, default_branches,
        release_match_setting_tag, pdb)
    assert len(unreleased_prs) == 0


@with_defer
async def test_discover_update_unreleased_prs_released(
        mdb, pdb, dag, default_branches, release_match_setting_tag):
    prs = await read_sql_query(
        select([PullRequest]).where(and_(PullRequest.number.in_(range(1000, 1010)),
                                         PullRequest.merged_at.isnot(None))),
        mdb, PullRequest, index=PullRequest.node_id.key)
    prs[prs[PullRequest.merged_at.key].isnull()] = datetime.now(tz=timezone.utc)
    utc = timezone.utc
    time_from = datetime(2018, 10, 1, tzinfo=utc)
    time_to = datetime(2018, 12, 1, tzinfo=utc)
    releases, matched_bys = await load_releases(
        (6366825,), ["src-d/go-git"], None, default_branches,
        time_from,
        time_to,
        release_match_setting_tag,
        mdb, pdb, None)
    released_prs, _, _ = await map_prs_to_releases(
        prs, releases, matched_bys, pd.DataFrame(columns=[Branch.commit_id.key]), {}, time_to, dag,
        release_match_setting_tag, (6366825,), mdb, pdb, None)
    await wait_deferred()
    await update_unreleased_prs(
        prs, released_prs, time_to, {},
        matched_bys, default_branches, release_match_setting_tag, pdb, asyncio.Event())
    await pdb.execute(update(GitHubMergedPullRequestFacts).values({
        GitHubMergedPullRequestFacts.data: pickle.dumps("fake"),
        GitHubMergedPullRequestFacts.updated_at: datetime.now(timezone.utc),
    }))
    unreleased_prs = await load_merged_unreleased_pull_request_facts(
        prs, time_to, LabelFilter.empty(), matched_bys, default_branches,
        release_match_setting_tag, pdb)
    assert len(unreleased_prs) == 1
    assert next(iter(unreleased_prs.keys())) == "MDExOlB1bGxSZXF1ZXN0MjI2NTg3NjE1"
    unreleased_prs = await load_merged_unreleased_pull_request_facts(
        prs, datetime(2018, 11, 1, tzinfo=utc), LabelFilter.empty(),
        matched_bys, default_branches, release_match_setting_tag, pdb)
    assert len(unreleased_prs) == 7


@with_defer
async def test_discover_update_unreleased_prs_exclude_inactive(
        mdb, pdb, dag, default_branches, release_match_setting_tag):
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    prs = await read_sql_query(
        select([PullRequest]).where(and_(PullRequest.number.in_(range(1000, 1010)),
                                         PullRequest.merged_at.isnot(None))),
        mdb, PullRequest, index=PullRequest.node_id.key)
    prs[prs[PullRequest.merged_at.key].isnull()] = datetime.now(tz=timezone.utc)
    utc = timezone.utc
    time_from = datetime(2018, 10, 1, tzinfo=utc)
    time_to = datetime(2018, 12, 1, tzinfo=utc)
    releases, matched_bys = await load_releases(
        (6366825,), ["src-d/go-git"], None, default_branches,
        time_from,
        time_to,
        release_match_setting_tag,
        mdb, pdb, None)
    released_prs, _, _ = await map_prs_to_releases(
        prs, releases, matched_bys, pd.DataFrame(columns=[Branch.commit_id.key]), {}, time_to, dag,
        release_match_setting_tag, (6366825,), mdb, pdb, None)
    await wait_deferred()
    await update_unreleased_prs(
        prs, released_prs, time_to, {},
        matched_bys, default_branches, release_match_setting_tag, pdb, asyncio.Event())
    await pdb.execute(update(GitHubMergedPullRequestFacts).values({
        GitHubMergedPullRequestFacts.data: pickle.dumps("fake"),
        GitHubMergedPullRequestFacts.updated_at: datetime.now(timezone.utc),
        GitHubMergedPullRequestFacts.activity_days: [
            datetime(2018, 10, 15, tzinfo=timezone.utc) if postgres else "2018-10-15",
        ],
    }))
    unreleased_prs = await load_merged_unreleased_pull_request_facts(
        prs, datetime(2018, 11, 1, tzinfo=utc), LabelFilter.empty(),
        matched_bys, default_branches, release_match_setting_tag, pdb,
        time_from=datetime(2018, 10, 14, tzinfo=utc), exclude_inactive=True)
    assert len(unreleased_prs) == 7
    unreleased_prs = await load_merged_unreleased_pull_request_facts(
        prs, datetime(2018, 11, 1, tzinfo=utc), LabelFilter.empty(),
        matched_bys, default_branches, release_match_setting_tag, pdb,
        time_from=datetime(2018, 10, 16, tzinfo=utc), exclude_inactive=True)
    assert len(unreleased_prs) == 0


@with_defer
async def test_discover_old_merged_unreleased_prs_smoke(
        mdb, pdb, dag, release_match_setting_tag, cache):
    metrics_time_from = datetime(2018, 1, 1, tzinfo=timezone.utc)
    metrics_time_to = datetime(2020, 5, 1, tzinfo=timezone.utc)
    await calc_pull_request_facts_github(
        metrics_time_from, metrics_time_to, {"src-d/go-git"}, {}, LabelFilter.empty(),
        JIRAFilter.empty(), False, release_match_setting_tag, False, False,
        (6366825,), mdb, pdb, cache,
    )
    await wait_deferred()
    unreleased_time_from = datetime(2018, 11, 1, tzinfo=timezone.utc)
    unreleased_time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    unreleased_prs = (await discover_inactive_merged_unreleased_prs(
        unreleased_time_from, unreleased_time_to, {"src-d/go-git"},
        {PRParticipationKind.MERGER: {"mcuadros"}}, LabelFilter.empty(), {},
        release_match_setting_tag, pdb, cache))[0]
    await wait_deferred()
    assert len(unreleased_prs) == 11
    unreleased_prs = await read_sql_query(
        select([PullRequest]).where(PullRequest.node_id.in_(unreleased_prs)),
        mdb, PullRequest, index=PullRequest.node_id.key)
    assert (unreleased_prs[PullRequest.merged_at.key] >
            datetime(2018, 10, 17, tzinfo=timezone.utc)).all()
    unreleased_prs = (await discover_inactive_merged_unreleased_prs(
        unreleased_time_from, unreleased_time_to, {"src-d/go-git"},
        {PRParticipationKind.MERGER: {"mcuadros"}}, LabelFilter.empty(), {},
        release_match_setting_tag, None, cache))[0]
    assert len(unreleased_prs) == 11
    unreleased_prs = await read_sql_query(
        select([PullRequest]).where(PullRequest.node_id.in_(unreleased_prs)),
        mdb, PullRequest, index=PullRequest.node_id.key)
    releases, matched_bys = await load_releases(
        (6366825,), ["src-d/go-git"], None, None, metrics_time_from, unreleased_time_to,
        release_match_setting_tag, mdb, pdb, cache)
    await wait_deferred()
    released_prs, _, _ = await map_prs_to_releases(
        unreleased_prs, releases, matched_bys, pd.DataFrame(columns=[Branch.commit_id.key]), {},
        unreleased_time_to, dag, release_match_setting_tag, (6366825,), mdb, pdb, cache)
    await wait_deferred()
    assert released_prs.empty
    unreleased_time_from = datetime(2018, 11, 19, tzinfo=timezone.utc)
    unreleased_time_to = datetime(2018, 11, 20, tzinfo=timezone.utc)
    unreleased_prs = (await discover_inactive_merged_unreleased_prs(
        unreleased_time_from, unreleased_time_to, {"src-d/go-git"},
        {PRParticipationKind.MERGER: {"mcuadros"}}, LabelFilter.empty(), {},
        release_match_setting_tag, pdb, cache))[0]
    assert not unreleased_prs


@with_defer
async def test_discover_old_merged_unreleased_prs_labels(
        mdb, pdb, release_match_setting_tag, cache):
    metrics_time_from = datetime(2018, 5, 1, tzinfo=timezone.utc)
    metrics_time_to = datetime(2019, 1, 1, tzinfo=timezone.utc)
    await calc_pull_request_facts_github(
        metrics_time_from, metrics_time_to, {"src-d/go-git"}, {}, LabelFilter.empty(),
        JIRAFilter.empty(), False, release_match_setting_tag, False, False,
        (6366825,), mdb, pdb, cache,
    )
    await wait_deferred()
    unreleased_time_from = datetime(2018, 9, 19, tzinfo=timezone.utc)
    unreleased_time_to = datetime(2018, 9, 30, tzinfo=timezone.utc)
    unreleased_prs = (await discover_inactive_merged_unreleased_prs(
        unreleased_time_from, unreleased_time_to, {"src-d/go-git"},
        {}, LabelFilter({"bug", "plumbing"}, set()), {}, release_match_setting_tag, pdb, cache))[0]
    assert unreleased_prs == ["MDExOlB1bGxSZXF1ZXN0MjE2MTA0NzY1",
                              "MDExOlB1bGxSZXF1ZXN0MjEzODQ1NDUx"]
    unreleased_prs = (await discover_inactive_merged_unreleased_prs(
        unreleased_time_from, unreleased_time_to, {"src-d/go-git"},
        {}, LabelFilter({"enhancement"}, set()), {}, release_match_setting_tag, pdb, cache))[0]
    assert unreleased_prs == ["MDExOlB1bGxSZXF1ZXN0MjEzODQwMDc3"]
    unreleased_prs = (await discover_inactive_merged_unreleased_prs(
        unreleased_time_from, unreleased_time_to, {"src-d/go-git"},
        {}, LabelFilter({"bug"}, {"ssh"}), {}, release_match_setting_tag, pdb, cache))[0]
    assert unreleased_prs == ["MDExOlB1bGxSZXF1ZXN0MjE2MTA0NzY1"]


async def test_store_precomputed_done_none_assert(pdb, pr_samples):
    samples = pr_samples(1)  # type: Sequence[PullRequestFacts]
    settings = {"github.com/one": ReleaseMatchSetting("{{default}}", ".*", ReleaseMatch.tag)}
    default_branches = {"one": "master"}
    prs = [MinedPullRequest(
        pr={PullRequest.created_at.key: samples[0].merged,
            PullRequest.repository_full_name.key: "one",
            PullRequest.user_login.key: "xxx",
            PullRequest.merged_by_login.key: "yyy",
            PullRequest.number.key: 1,
            PullRequest.node_id.key: uuid.uuid4().hex},
        release={matched_by_column: settings["github.com/one"],
                 Release.author.key: "foo",
                 Release.url.key: "https://release",
                 Release.id.key: "MDwhatever="},
        comments=gen_dummy_df(samples[0].first_comment_on_first_review),
        commits=pd.DataFrame.from_records(
            [["yyy", "yyy", samples[0].first_commit]],
            columns=[
                PullRequestCommit.committer_login.key,
                PullRequestCommit.author_login.key,
                PullRequestCommit.committed_date.key,
            ],
        ),
        reviews=gen_dummy_df(samples[0].first_comment_on_first_review),
        review_comments=gen_dummy_df(samples[0].first_comment_on_first_review),
        review_requests=gen_dummy_df(samples[0].first_review_request),
        labels=pd.DataFrame.from_records([["bug"]], columns=["name"]),
        jiras=pd.DataFrame(),
    )]
    await store_precomputed_done_facts(
        prs, [("src-d/go-git", None)], default_branches, settings, pdb)
    with pytest.raises(AssertionError):
        await store_precomputed_done_facts(
            prs, zip(repeat("one"), samples), default_branches, settings, pdb)


@with_defer
async def test_store_merged_unreleased_pull_request_facts_smoke(
        mdb, pdb, default_branches, release_match_setting_tag):
    prs, dfs, facts, matched_bys = await _fetch_pull_requests(
        {"src-d/go-git": set(range(1000, 1010))},
        release_match_setting_tag, (6366825,), mdb, pdb, None)
    for pr in prs:
        if pr.pr[PullRequest.merged_at.key] is None:
            pr.pr[PullRequest.merged_at.key] = datetime.now(tz=timezone.utc)
    dfs.prs.loc[dfs.prs[PullRequest.merged_at.key].isnull(), PullRequest.merged_at.key] = \
        datetime.now(tz=timezone.utc)
    event = asyncio.Event()
    await update_unreleased_prs(
        dfs.prs, new_released_prs_df(), datetime(2018, 11, 1, tzinfo=timezone.utc), {},
        matched_bys, default_branches, release_match_setting_tag, pdb, event)
    samples_good, samples_bad = [], []
    for pr in prs:
        f = facts[pr.pr[PullRequest.node_id.key]]
        fields = dict(f)
        if f.merged is None:
            fields["merged"] = datetime.now(tz=timezone.utc)
        fields["released"] = None
        samples_good.append(PullRequestFacts(**fields))
        samples_bad.append(f)
    with pytest.raises(AssertionError):
        await store_merged_unreleased_pull_request_facts(
            zip(prs, samples_bad), matched_bys, default_branches, release_match_setting_tag, pdb,
            event)
    await store_merged_unreleased_pull_request_facts(
        zip(prs, samples_good), matched_bys, default_branches, release_match_setting_tag, pdb,
        event)
    true_dict = {pr.pr[PullRequest.node_id.key]: s for pr, s in zip(prs, samples_good)}
    ghmprf = GitHubMergedPullRequestFacts
    rows = await pdb.fetch_all(select([ghmprf]))
    assert len(rows) == 10
    for row in rows:
        assert isinstance(row[ghmprf.activity_days.key], list)
        assert len(row[ghmprf.activity_days.key]) > 0
    new_dict = {r[ghmprf.pr_node_id.key]: pickle.loads(r[ghmprf.data.key]) for r in rows}
    assert true_dict == new_dict


@with_defer
async def test_store_open_pull_request_facts_smoke(
        mdb, pdb, release_match_setting_tag):
    prs, dfs, facts, _ = await _fetch_pull_requests(
        {"src-d/go-git": set(range(1000, 1010))},
        release_match_setting_tag, (6366825,), mdb, pdb, None)
    with pytest.raises(AssertionError):
        await store_open_pull_request_facts(
            zip(prs, (facts[pr.pr[PullRequest.node_id.key]] for pr in prs)), pdb)
    samples = []
    true_dict = {}
    for pr in prs:
        f = facts[pr.pr[PullRequest.node_id.key]]
        fields = dict(f)
        fields["closed"] = None
        f = PullRequestFacts(**fields)
        samples.append(f)
        true_dict[pr.pr[PullRequest.node_id.key]] = f
    dfs.prs[PullRequest.closed_at.key] = None
    await store_open_pull_request_facts(zip(prs, samples), pdb)
    ghoprf = GitHubOpenPullRequestFacts
    rows = await pdb.fetch_all(select([ghoprf]))
    assert len(rows) == 10
    new_dict = {}
    for row in rows:
        assert isinstance(row[ghoprf.activity_days.key], list)
        assert len(row[ghoprf.activity_days.key]) > 0
        new_dict[row[ghoprf.pr_node_id.key]] = pickle.loads(row[ghoprf.data.key])
    assert true_dict == new_dict

    loaded_facts = await load_open_pull_request_facts(dfs.prs, pdb)
    for repo, _ in loaded_facts.values():
        assert repo == "src-d/go-git"
    loaded_facts = {k: v[1] for k, v in loaded_facts.items()}
    assert true_dict == loaded_facts

    loaded_facts = await load_open_pull_request_facts_unfresh(
        dfs.prs.index, datetime(2016, 1, 1), datetime(2020, 1, 1), True, pdb)
    for repo, _ in loaded_facts.values():
        assert repo == "src-d/go-git"
    loaded_facts = {k: v[1] for k, v in loaded_facts.items()}
    assert true_dict == loaded_facts
    loaded_facts = await load_open_pull_request_facts_unfresh(
        dfs.prs.index, datetime(2019, 11, 1), datetime(2020, 1, 1), True, pdb)
    assert len(loaded_facts) == 0
