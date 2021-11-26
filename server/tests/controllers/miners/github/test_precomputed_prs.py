import asyncio
from datetime import datetime, timedelta, timezone
import math
from typing import Sequence

import pandas as pd
import pytest
from sqlalchemy import and_, select, update

from athenian.api.async_utils import read_sql_query
from athenian.api.controllers.features.github.pull_request_filter import _fetch_pull_requests
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.deployment import mine_deployments
from athenian.api.controllers.miners.github.precomputed_prs import \
    delete_force_push_dropped_prs, discover_inactive_merged_unreleased_prs, \
    store_merged_unreleased_pull_request_facts, store_open_pull_request_facts, \
    store_precomputed_done_facts, update_unreleased_prs
from athenian.api.controllers.miners.github.release_match import PullRequestToReleaseMapper
from athenian.api.controllers.miners.github.released_pr import matched_by_column, \
    new_released_prs_df
from athenian.api.controllers.miners.types import MinedPullRequest, PRParticipationKind, \
    PullRequestFacts
from athenian.api.controllers.settings import LogicalRepositorySettings, ReleaseMatch, \
    ReleaseMatchSetting, ReleaseSettings
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.metadata.github import Branch, PullRequest, PullRequestCommit, Release
from athenian.api.models.persistentdata.models import DeploymentNotification
from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts, \
    GitHubMergedPullRequestFacts, GitHubOpenPullRequestFacts
from tests.conftest import with_preloading_env
from tests.controllers.conftest import FakeFacts, with_only_master_branch


def gen_dummy_df(dt: datetime) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [["vmarkovtsev", 40020, dt, dt]],
        columns=["user_login", "user_node_id", "created_at", "submitted_at"])


async def test_load_store_precomputed_done_smoke(
        pdb, pr_samples, done_prs_facts_loader, with_preloading_enabled, prefixer):
    samples = pr_samples(200)  # type: Sequence[PullRequestFacts]
    for i in range(1, 6):
        # merged but unreleased
        kwargs = dict(**samples[-i])
        kwargs["released"] = None
        samples[-i] = PullRequestFacts.from_fields(**kwargs)
    for i in range(6, 11):
        # rejected
        kwargs = dict(**samples[-i])
        kwargs["released"] = kwargs["merged"] = None
        samples[-i] = PullRequestFacts.from_fields(**kwargs)
    names = ["one", "two", "three"]
    settings = ReleaseSettings({
        "github.com/" + k: ReleaseMatchSetting("{{default}}", ".*", ".*", ReleaseMatch(i))
        for i, k in enumerate(names)})
    default_branches = {k: "master" for k in names}
    prs = [MinedPullRequest(
        pr={PullRequest.created_at.name: s.created,
            PullRequest.repository_full_name.name: names[i % len(names)],
            PullRequest.user_login.name: "vmarkovtsev",
            PullRequest.user_node_id.name: 40020,
            PullRequest.merged_by_login.name: "mcuadros",
            PullRequest.merged_by_id.name: 39789,
            PullRequest.number.name: i + 1,
            PullRequest.node_id.name: i + 100500},
        release={matched_by_column: settings.native[names[i % len(names)]].match % 2,
                 Release.author.name: "mcarmonaa",
                 Release.author_node_id.name: 39818,
                 Release.url.name: "https://release",
                 Release.node_id.name: i},
        comments=gen_dummy_df(s.first_comment_on_first_review),
        commits=pd.DataFrame.from_records(
            [["mcarmonaa", "mcarmonaa", 39818, 39818,
              s.first_commit]],
            columns=[
                PullRequestCommit.committer_login.name,
                PullRequestCommit.author_login.name,
                PullRequestCommit.committer_user_id.name,
                PullRequestCommit.author_user_id.name,
                PullRequestCommit.committed_date.name,
            ],
        ),
        reviews=gen_dummy_df(s.first_comment_on_first_review),
        review_comments=gen_dummy_df(s.first_comment_on_first_review),
        review_requests=gen_dummy_df(s.first_review_request),
        labels=pd.DataFrame.from_records(([["bug"]], [["feature"]])[i % 2], columns=["name"]),
        jiras=pd.DataFrame(),
        deployments=None,
    ) for i, s in enumerate(samples)]

    def with_mutables(s, repo):
        s.repository_full_name = repo
        s.author = "vmarkovtsev"
        s.merger = "mcuadros"
        s.releaser = "mcarmonaa"
        return s

    await store_precomputed_done_facts(
        prs, [with_mutables(s, names[i % len(names)])
              for i, s in enumerate(samples)],
        default_branches, settings, 1, pdb)
    # we should not crash on repeat
    await store_precomputed_done_facts(
        prs, [with_mutables(s, names[i % len(names)])
              for i, s in enumerate(samples)],
        default_branches, settings, 1, pdb)
    if with_preloading_enabled:
        await pdb.cache.refresh()
    released_ats = sorted((t.released, i) for i, t in enumerate(samples[:-10]))
    time_from = released_ats[len(released_ats) // 2][0].item().replace(tzinfo=timezone.utc)
    time_to = released_ats[-1][0].item().replace(tzinfo=timezone.utc)
    n = len(released_ats) - len(released_ats) // 2 + \
        sum(1 for s in samples[-10:-5]
            if s.closed.item().replace(tzinfo=timezone.utc) >= time_from)
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, names, {}, LabelFilter.empty(), default_branches,
        False, settings, prefixer, 1, pdb)
    assert len(loaded_prs) == n
    true_prs = {prs[i].pr[PullRequest.node_id.name]: samples[i] for _, i in released_ats[-n:]}
    for i, s in enumerate(samples[-10:-5]):
        if s.closed.item().replace(tzinfo=timezone.utc) >= time_from:
            true_prs[prs[-10 + i].pr[PullRequest.node_id.name]] = s
    diff_keys = {node_id for node_id, _ in loaded_prs} - set(true_prs)
    assert not diff_keys
    for (node_id, repo), load_value in loaded_prs.items():
        assert load_value == true_prs[node_id], node_id
        assert repo == load_value.repository_full_name
        assert load_value.repository_full_name in names
        assert load_value.author is not None
        assert load_value.merger is not None
        if load_value.released is not None:
            assert load_value.releaser is not None


async def test_load_store_precomputed_done_filters(
        pr_samples, pdb, done_prs_facts_loader, with_preloading_enabled, prefixer):
    samples = pr_samples(102)  # type: Sequence[PullRequestFacts]
    names = ["one", "two", "three"]
    settings = ReleaseSettings({
        "github.com/" + k: ReleaseMatchSetting("{{default}}", ".*", ".*", ReleaseMatch(i))
        for i, k in enumerate(names)})
    default_branches = {k: "master" for k in names}
    prs = [MinedPullRequest(
        pr={PullRequest.created_at.name: s.created,
            PullRequest.repository_full_name.name: names[i % len(names)],
            PullRequest.user_login.name: ["vmarkovtsev", "marnovo"][i % 2],
            PullRequest.user_node_id.name: [40020, 39792][i % 2],
            PullRequest.merged_by_login.name: "mcuadros",
            PullRequest.merged_by_id.name: 39789,
            PullRequest.number.name: i + 1,
            PullRequest.node_id.name: i + 100500},
        release={matched_by_column: settings.native[names[i % len(names)]].match % 2,
                 Release.author.name: ["marnovo", "mcarmonaa"][i % 2],
                 Release.author_node_id.name: [39792, 39818][i % 2],
                 Release.url.name: "https://release",
                 Release.node_id.name: i},
        comments=gen_dummy_df(s.first_comment_on_first_review),
        commits=pd.DataFrame.from_records(
            [["mcuadros", "mcuadros", 39789, 39789, s.first_commit]],
            columns=[
                PullRequestCommit.committer_login.name,
                PullRequestCommit.author_login.name,
                PullRequestCommit.committer_user_id.name,
                PullRequestCommit.author_user_id.name,
                PullRequestCommit.committed_date.name,
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
        deployments=None,
    ) for i, s in enumerate(samples)]

    def with_mutables(s, i):
        s.repository_full_name = names[i % len(names)]
        s.author = ["vmarkovtsev", "marnovo"][i % 2]
        s.merger = "mcuadros"
        s.releaser = ["marnovo", "mcarmonaa"][i % 2]
        return s

    await store_precomputed_done_facts(
        prs, [with_mutables(s, i) for i, s in enumerate(samples)],
        default_branches, settings, 1, pdb)
    if with_preloading_enabled:
        await pdb.cache.refresh()
    time_from = min(s.created for s in samples).item().replace(tzinfo=timezone.utc)
    time_to = max(s.max_timestamp() for s in samples).item().replace(tzinfo=timezone.utc)
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, ["one"], {}, LabelFilter.empty(), default_branches,
        False, settings, prefixer, 1, pdb)
    assert {node_id for node_id, _ in loaded_prs} == \
           {pr.pr[PullRequest.node_id.name] for pr in prs[::3]}
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, names, {PRParticipationKind.AUTHOR: {"marnovo"},
                                    PRParticipationKind.RELEASER: {"mcarmonaa"}},
        LabelFilter.empty(), default_branches, False, settings, prefixer, 1, pdb)
    assert {node_id for node_id, _ in loaded_prs} == \
           {pr.pr[PullRequest.node_id.name] for pr in prs[1::2]}
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, names, {PRParticipationKind.COMMIT_AUTHOR: {"mcuadros"}},
        LabelFilter.empty(), default_branches, False, settings, prefixer, 1, pdb)
    assert len(loaded_prs) == len(prs)
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, names, {}, LabelFilter({"bug", "vmarkovtsev"}, set()),
        default_branches, False, settings, prefixer, 1, pdb)
    assert len(loaded_prs) == len(prs) / 2
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, names, {}, LabelFilter({"bug"}, {"bad"}),
        default_branches, False, settings, prefixer, 1, pdb)
    assert len(loaded_prs) == int(math.ceil(len(prs) / 4.0))


async def test_load_store_precomputed_done_match_by(
        pr_samples, default_branches, pdb, done_prs_facts_loader,
        with_preloading_enabled, prefixer):
    samples, prs, settings = _gen_one_pr(pr_samples)

    def with_mutables(s):
        s.repository_full_name = "src-d/go-git"
        s.author = "vmarkovtsev"
        s.merger = "mcuadros"
        s.releaser = "mcarmonaa"
        return s

    await store_precomputed_done_facts(
        prs, [with_mutables(s) for s in samples],
        default_branches, settings, 1, pdb)
    if with_preloading_enabled:
        await pdb.cache.refresh()
    time_from = samples[0].created.item().replace(tzinfo=timezone.utc) - timedelta(days=365)
    time_to = samples[0].released.item().replace(tzinfo=timezone.utc) + timedelta(days=1)
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, ["src-d/go-git"], {}, LabelFilter.empty(),
        default_branches, False, settings, prefixer, 1, pdb)
    assert len(loaded_prs) == 1
    settings = ReleaseSettings({
        "github.com/src-d/go-git": ReleaseMatchSetting("master", ".*", ".*", ReleaseMatch.branch),
    })
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, ["src-d/go-git"], {}, LabelFilter.empty(),
        default_branches, False, settings, prefixer, 1, pdb)
    assert len(loaded_prs) == 1
    settings = ReleaseSettings({
        "github.com/src-d/go-git": ReleaseMatchSetting(
            "nope", ".*", ".*", ReleaseMatch.tag_or_branch),
    })
    loaded_prs, ambiguous = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, ["src-d/go-git"], {}, LabelFilter.empty(),
        default_branches, False, settings, prefixer, 1, pdb)
    assert len(loaded_prs) == 0
    assert len(ambiguous) == 0
    settings = ReleaseSettings({
        "github.com/src-d/go-git": ReleaseMatchSetting(
            "{{default}}", ".*", ".*", ReleaseMatch.tag),
    })
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, ["src-d/go-git"], {}, LabelFilter.empty(),
        default_branches, False, settings, prefixer, 1, pdb)
    assert len(loaded_prs) == 0
    prs[0].release[matched_by_column] = 1

    def with_mutables(s):
        s.repository_full_name = "src-d/go-git"
        s.author = "vmarkovtsev"
        s.merger = "mcuadros"
        s.releaser = "mcarmonaa"
        return s

    await store_precomputed_done_facts(
        prs, [with_mutables(s) for s in samples],
        default_branches, settings, 1, pdb)
    if with_preloading_enabled:
        await pdb.cache.refresh()
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, ["src-d/go-git"], {}, LabelFilter.empty(),
        default_branches, False, settings, prefixer, 1, pdb)
    assert len(loaded_prs) == 1
    settings = ReleaseSettings({
        "github.com/src-d/go-git":
            ReleaseMatchSetting("{{default}}", "vmarkovtsev", ".*", ReleaseMatch.tag),
    })
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, ["src-d/go-git"], {}, LabelFilter.empty(),
        default_branches, False, settings, prefixer, 1, pdb)
    assert len(loaded_prs) == 0


async def test_load_store_precomputed_done_exclude_inactive(
        pr_samples, pdb, done_prs_facts_loader,
        with_preloading_enabled, prefixer):
    default_branches = {
        "one": "master",
    }
    while True:
        samples = pr_samples(2)  # type: Sequence[PullRequestFacts]
        samples = sorted(samples, key=lambda s: s.first_comment_on_first_review)
        deltas = [(samples[1].first_comment_on_first_review -
                   samples[0].first_comment_on_first_review),
                  samples[0].first_comment_on_first_review - samples[1].created,
                  samples[1].created - samples[0].created]
        if all(d > timedelta(days=2) for d in deltas):
            break
    settings = ReleaseSettings({
        "github.com/one": ReleaseMatchSetting("{{default}}", ".*", ".*", ReleaseMatch.tag),
    })
    prs = [MinedPullRequest(
        pr={PullRequest.created_at.name: s.created,
            PullRequest.repository_full_name.name: "one",
            PullRequest.user_login.name: "vmarkovtsev",
            PullRequest.user_node_id.name: 40020,
            PullRequest.merged_by_login.name: "mcuadros",
            PullRequest.merged_by_id.name: 39789,
            PullRequest.number.name: 777,
            PullRequest.node_id.name: i + 100500},
        release={matched_by_column: settings.native["one"].match,
                 Release.author.name: "mcarmonaa",
                 Release.author_node_id.name: 39818,
                 Release.url.name: "https://release",
                 Release.node_id.name: 777},
        comments=gen_dummy_df(s.first_comment_on_first_review),
        commits=pd.DataFrame.from_records(
            [["mcuadros", "mcuadros", 39789, 39789,
              s.first_comment_on_first_review]],
            columns=[
                PullRequestCommit.committer_login.name,
                PullRequestCommit.author_login.name,
                PullRequestCommit.committer_user_id.name,
                PullRequestCommit.author_user_id.name,
                PullRequestCommit.committed_date.name,
            ],
        ),
        reviews=gen_dummy_df(s.first_comment_on_first_review),
        review_comments=gen_dummy_df(s.first_comment_on_first_review),
        review_requests=gen_dummy_df(s.first_comment_on_first_review),
        labels=pd.DataFrame.from_records([["bug"]], columns=["name"]),
        jiras=pd.DataFrame(),
        deployments=None,
    ) for i, s in enumerate(samples)]

    def with_mutables(s):
        s.repository_full_name = "one"
        s.author = "vmarkovtsev"
        s.merger = "mcuadros"
        s.releaser = "mcarmonaa"
        return s

    await store_precomputed_done_facts(
        prs, [with_mutables(s) for s in samples],
        default_branches, settings, 1, pdb)
    if with_preloading_enabled:
        await pdb.cache.refresh()
    time_from = samples[1].created.item().replace(tzinfo=timezone.utc) + timedelta(days=1)
    time_to = samples[0].first_comment_on_first_review.item().replace(tzinfo=timezone.utc)
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, ["one"], {}, LabelFilter.empty(), default_branches,
        True, settings, prefixer, 1, pdb)
    assert len(loaded_prs) == 1
    assert loaded_prs[(prs[0].pr[PullRequest.node_id.name], "one")] == \
           with_mutables(samples[0])
    time_from = samples[1].created.item().replace(tzinfo=timezone.utc) - timedelta(days=1)
    time_to = samples[1].created.item().replace(tzinfo=timezone.utc) + timedelta(seconds=1)
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, ["one"], {}, LabelFilter.empty(), default_branches,
        True, settings, prefixer, 1, pdb)
    assert len(loaded_prs) == 1
    assert loaded_prs[(prs[1].pr[PullRequest.node_id.name], "one")] == with_mutables(samples[1])


async def test_load_precomputed_done_times_reponums_smoke(
        pr_samples, pdb, done_prs_facts_loader, prefixer):
    samples = pr_samples(12)  # type: Sequence[PullRequestFacts]
    names = ["one", "two", "three"]
    settings = ReleaseSettings({
        "github.com/" + k: ReleaseMatchSetting("{{default}}", ".*", ".*", ReleaseMatch(i))
        for i, k in enumerate(names)
    })
    default_branches = {k: "master" for k in names}
    prs = [MinedPullRequest(
        pr={PullRequest.created_at.name: s.created,
            PullRequest.repository_full_name.name: names[i % len(names)],
            PullRequest.user_login.name: ["vmarkovtsev", "marnovo"][i % 2],
            PullRequest.user_node_id.name: [40020, 39792][i % 2],
            PullRequest.merged_by_login.name: "mcuadros",
            PullRequest.merged_by_id.name: 39789,
            PullRequest.number.name: i + 1,
            PullRequest.node_id.name: i + 100500},
        release={matched_by_column: settings.native[names[i % len(names)]].match % 2,
                 Release.author.name: ["marnovo", "mcarmonaa"][i % 2],
                 Release.author_node_id.name:
                     [39792, 39818][i % 2],
                 Release.url.name: "https://release",
                 Release.node_id.name: i},
        comments=gen_dummy_df(s.first_comment_on_first_review),
        commits=pd.DataFrame.from_records(
            [["mcuadros", "mcuadros", 39789, 39789,
              s.first_commit]],
            columns=[
                PullRequestCommit.committer_login.name,
                PullRequestCommit.author_login.name,
                PullRequestCommit.committer_user_id.name,
                PullRequestCommit.author_user_id.name,
                PullRequestCommit.committed_date.name,
            ],
        ),
        reviews=gen_dummy_df(s.first_comment_on_first_review),
        review_comments=gen_dummy_df(s.first_comment_on_first_review),
        review_requests=gen_dummy_df(s.first_review_request),
        labels=pd.DataFrame.from_records(([["bug"]], [["feature"]])[i % 2], columns=["name"]),
        jiras=pd.DataFrame(),
        deployments=None,
    ) for i, s in enumerate(samples)]

    def with_mutables(s, i):
        s.repository_full_name = names[i % len(names)]
        s.author = ["vmarkovtsev", "marnovo"][i % 2]
        s.merger = "mcuadros"
        s.releaser = ["marnovo", "mcarmonaa"][i % 2]
        return s

    await store_precomputed_done_facts(
        prs, [with_mutables(s, i) for i, s in enumerate(samples)],
        default_branches, settings, 1, pdb)
    query1 = {"one": {pr.pr[PullRequest.number.name] for pr in prs
                      if pr.pr[PullRequest.repository_full_name.name] == "one"}}
    assert len(query1["one"]) == 4
    new_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_reponums(
        query1, default_branches, settings, prefixer, 1, pdb)
    assert new_prs == {
        (pr.pr[PullRequest.node_id.name], pr.pr[PullRequest.repository_full_name.name]): s
        for pr, s in zip(prs, samples)
        if pr.pr[PullRequest.repository_full_name.name] == "one"
    }
    query2 = {"one": set()}
    new_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_reponums(
        query2, default_branches, settings, prefixer, 1, pdb)
    assert len(new_prs) == 0
    query3 = {"one": {100500}}
    new_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_reponums(
        query3, default_branches, settings, prefixer, 1, pdb)
    assert len(new_prs) == 0


@pytest.mark.xfail(with_preloading_env, reason="Not supported in the preloader")
@pytest.mark.parametrize("exclude_inactive", [False, True])
@with_defer
async def test_load_precomputed_done_times_deployments(
        metrics_calculator_factory, mdb, pdb, rdb, dag, release_match_setting_tag, cache,
        prefixer, release_loader, done_prs_facts_loader, default_branches,
        precomputed_deployments, exclude_inactive):
    metrics_calculator = metrics_calculator_factory(1, (6366825,), with_cache=True)
    time_from = datetime(2018, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2020, 5, 1, tzinfo=timezone.utc)
    await metrics_calculator.calc_pull_request_facts_github(
        time_from, time_to, {"src-d/go-git"}, {}, LabelFilter.empty(),
        JIRAFilter.empty(), False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, False, False,
    )
    await wait_deferred()
    time_from = datetime(2019, 10, 1, tzinfo=timezone.utc)
    time_to = datetime(2019, 12, 1, tzinfo=timezone.utc)
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_facts_filters(
        time_from, time_to, {"src-d/go-git"}, {}, LabelFilter.empty(), default_branches,
        False, release_match_setting_tag, prefixer, 1, pdb)
    assert len(loaded_prs) == 415  # 2 without deployments


def _gen_one_pr(pr_samples):
    samples = pr_samples(1)  # type: Sequence[PullRequestFacts]
    s = samples[0]
    settings = ReleaseSettings({
        "github.com/src-d/go-git": ReleaseMatchSetting(
            "{{default}}", ".*", ".*", ReleaseMatch.tag_or_branch),
    })
    prs = [MinedPullRequest(
        pr={PullRequest.created_at.name: s.created,
            PullRequest.repository_full_name.name: "src-d/go-git",
            PullRequest.user_login.name: "vmarkovtsev",
            PullRequest.user_node_id.name: 40020,
            PullRequest.merged_by_login.name: "mcuadros",
            PullRequest.merged_by_id.name: 39789,
            PullRequest.number.name: 777,
            PullRequest.node_id.name: 100500},
        release={matched_by_column: ReleaseMatch.branch,
                 Release.author.name: "mcarmonaa",
                 Release.author_node_id.name: 39818,
                 Release.url.name: "https://release",
                 Release.node_id.name: 777},
        comments=gen_dummy_df(s.first_comment_on_first_review),
        commits=pd.DataFrame.from_records(
            [["mcarmonaa", "mcarmonaa", 39818, 39818,
              s.first_commit]],
            columns=[
                PullRequestCommit.committer_login.name,
                PullRequestCommit.author_login.name,
                PullRequestCommit.committer_user_id.name,
                PullRequestCommit.author_user_id.name,
                PullRequestCommit.committed_date.name,
            ],
        ),
        reviews=gen_dummy_df(s.first_comment_on_first_review),
        review_comments=gen_dummy_df(s.first_comment_on_first_review),
        review_requests=gen_dummy_df(s.first_review_request),
        labels=pd.DataFrame.from_records([["bug"]], columns=["name"]),
        jiras=pd.DataFrame(),
        deployments=None,
    )]
    return samples, prs, settings


async def test_store_precomputed_done_facts_empty(pdb):
    await store_precomputed_done_facts([], [], None, None, 1, pdb)


async def test_load_precomputed_done_candidates_smoke(pr_samples, default_branches, pdb,
                                                      done_prs_facts_loader):
    samples, prs, settings = _gen_one_pr(pr_samples)

    def with_mutables(s):
        s.repository_full_name = "src-d/go-git"
        s.author = "vmarkovtsev"
        s.merger = "mcuadros"
        s.releaser = "mcarmonaa"
        return s

    await store_precomputed_done_facts(
        prs, [with_mutables(s) for s in samples],
        default_branches, settings, 1, pdb)
    time_from = samples[0].created.item().replace(tzinfo=timezone.utc)
    time_to = samples[0].released.item().replace(tzinfo=timezone.utc)
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_candidates(
        time_from, time_to, ["one"], {"one": "master"}, settings, 1, pdb)
    assert len(loaded_prs) == 0
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_candidates(
        time_from, time_to, ["src-d/go-git"], default_branches, settings, 1, pdb)
    assert loaded_prs == {prs[0].pr[PullRequest.node_id.name]}
    loaded_prs, _ = await done_prs_facts_loader.load_precomputed_done_candidates(
        time_from, time_from, ["src-d/go-git"],
        default_branches, settings, 1, pdb)
    assert len(loaded_prs) == 0


async def test_load_precomputed_done_candidates_ambiguous(pr_samples, default_branches, pdb,
                                                          done_prs_facts_loader):
    samples, prs, settings = _gen_one_pr(pr_samples)

    def with_mutables(s):
        s.repository_full_name = "src-d/go-git"
        s.author = "vmarkovtsev"
        s.merger = "mcuadros"
        s.releaser = "mcarmonaa"
        return s

    await store_precomputed_done_facts(
        prs, [with_mutables(s) for s in samples],
        default_branches, settings, 1, pdb)
    time_from = samples[0].created.item().replace(tzinfo=timezone.utc)
    time_to = samples[0].released.item().replace(tzinfo=timezone.utc)
    loaded_prs, ambiguous = await done_prs_facts_loader.load_precomputed_done_candidates(
        time_from, time_to, ["src-d/go-git"], default_branches, settings, 1, pdb)
    assert len(loaded_prs) == 1
    assert len(ambiguous["src-d/go-git"]) == 1

    prs[0].release[matched_by_column] = ReleaseMatch.tag
    await store_precomputed_done_facts(
        prs, [with_mutables(s) for s in samples],
        default_branches, settings, 1, pdb)
    loaded_prs, ambiguous = await done_prs_facts_loader.load_precomputed_done_candidates(
        time_from, time_to, ["src-d/go-git"], default_branches, settings, 1, pdb)
    assert len(loaded_prs) == 1
    assert len(ambiguous["src-d/go-git"]) == 0


@with_defer
async def test_load_precomputed_pr_releases_smoke(
        pr_samples, default_branches, pdb, cache, done_prs_facts_loader, prefixer):
    samples, prs, settings = _gen_one_pr(pr_samples)

    def with_mutables(s):
        s.repository_full_name = "src-d/go-git"
        s.author = "vmarkovtsev"
        s.merger = "mcuadros"
        s.releaser = "mcarmonaa"
        return s

    await store_precomputed_done_facts(
        prs, [with_mutables(s) for s in samples],
        default_branches, settings, 1, pdb)

    def df_from_prs():
        df = pd.DataFrame()
        df[PullRequest.node_id.name] = [pr.pr[PullRequest.node_id.name] for pr in prs]
        df[PullRequest.repository_full_name.name] = \
            [pr.pr[PullRequest.repository_full_name.name] for pr in prs]
        df.set_index([PullRequest.node_id.name, PullRequest.repository_full_name.name],
                     inplace=True)
        return df

    for i in range(2):
        released_prs = await done_prs_facts_loader.load_precomputed_pr_releases(
            df_from_prs(),
            max(s.released.item().replace(tzinfo=timezone.utc) for s in samples) +
            timedelta(days=1),
            {pr.pr[PullRequest.repository_full_name.name]: ReleaseMatch.branch for pr in prs},
            default_branches, settings, prefixer, 1, pdb if i == 0 else None, cache)
        await wait_deferred()
        for s, pr in zip(samples, prs):
            rpr = released_prs.loc[pr.pr[PullRequest.node_id.name], "src-d/go-git"]
            for col in (Release.author.name, Release.url.name, Release.node_id.name,
                        matched_by_column):
                assert rpr[col] == pr.release[col], i
            assert rpr[Release.published_at.name].replace(tzinfo=None) == s.released, i


async def test_load_precomputed_pr_releases_time_to(
        pr_samples, default_branches, pdb, done_prs_facts_loader, prefixer):
    samples, prs, settings = _gen_one_pr(pr_samples)

    def with_mutables(s):
        s.repository_full_name = "src-d/go-git"
        s.author = "vmarkovtsev"
        s.merger = "mcuadros"
        s.releaser = "mcarmonaa"
        return s

    await store_precomputed_done_facts(
        prs, [with_mutables(s) for s in samples],
        default_branches, settings, 1, pdb)

    def df_from_prs():
        df = pd.DataFrame()
        df[PullRequest.node_id.name] = [pr.pr[PullRequest.node_id.name] for pr in prs]
        df[PullRequest.repository_full_name.name] = \
            [pr.pr[PullRequest.repository_full_name.name] for pr in prs]
        df.set_index([PullRequest.node_id.name, PullRequest.repository_full_name.name],
                     inplace=True)
        return df

    released_prs = await done_prs_facts_loader.load_precomputed_pr_releases(
        df_from_prs(),
        min(s.released.item().replace(tzinfo=timezone.utc) for s in samples),
        {pr.pr[PullRequest.repository_full_name.name]: ReleaseMatch.branch for pr in prs},
        default_branches, settings, prefixer, 1, pdb, None)
    assert released_prs.empty


async def test_load_precomputed_pr_releases_release_mismatch(
        pr_samples, default_branches, pdb, done_prs_facts_loader, prefixer):
    samples, prs, settings = _gen_one_pr(pr_samples)

    def with_mutables(s):
        s.repository_full_name = "src-d/go-git"
        s.author = "vmarkovtsev"
        s.merger = "mcuadros"
        s.releaser = "mcarmonaa"
        return s

    await store_precomputed_done_facts(
        prs, [with_mutables(s) for s in samples],
        default_branches, settings, 1, pdb)

    def df_from_prs():
        df = pd.DataFrame()
        df[PullRequest.node_id.name] = [pr.pr[PullRequest.node_id.name] for pr in prs]
        df[PullRequest.repository_full_name.name] = \
            [pr.pr[PullRequest.repository_full_name.name] for pr in prs]
        df.set_index([PullRequest.node_id.name, PullRequest.repository_full_name.name],
                     inplace=True)
        return df

    released_prs = await done_prs_facts_loader.load_precomputed_pr_releases(
        df_from_prs(),
        max(s.released.item().replace(tzinfo=timezone.utc) for s in samples) + timedelta(days=1),
        {pr.pr[PullRequest.repository_full_name.name]: ReleaseMatch.tag for pr in prs},
        default_branches, settings, prefixer, 1, pdb, None)
    assert released_prs.empty
    released_prs = await done_prs_facts_loader.load_precomputed_pr_releases(
        df_from_prs(),
        max(s.released.item().replace(tzinfo=timezone.utc) for s in samples) + timedelta(days=1),
        {pr.pr[PullRequest.repository_full_name.name]: ReleaseMatch.branch for pr in prs},
        {"src-d/go-git": "vmarkovtsev"}, settings, prefixer, 1, pdb, None)
    assert released_prs.empty


async def test_load_precomputed_pr_releases_tag(
        pr_samples, default_branches, pdb, done_prs_facts_loader, prefixer):
    samples, prs, settings = _gen_one_pr(pr_samples)
    prs[0].release[matched_by_column] = ReleaseMatch.tag

    def with_repository_full_name(s):
        s.repository_full_name = "src-d/go-git"
        return s

    await store_precomputed_done_facts(
        prs, [with_repository_full_name(s) for s in samples],
        default_branches, settings, 1, pdb)

    def df_from_prs():
        df = pd.DataFrame()
        df[PullRequest.node_id.name] = [pr.pr[PullRequest.node_id.name] for pr in prs]
        df[PullRequest.repository_full_name.name] = \
            [pr.pr[PullRequest.repository_full_name.name] for pr in prs]
        df.set_index([PullRequest.node_id.name, PullRequest.repository_full_name.name],
                     inplace=True)
        return df

    released_prs = await done_prs_facts_loader.load_precomputed_pr_releases(
        df_from_prs(),
        max(s.released.item().replace(tzinfo=timezone.utc) for s in samples) + timedelta(days=1),
        {pr.pr[PullRequest.repository_full_name.name]: ReleaseMatch.tag for pr in prs},
        default_branches, settings, prefixer, 1, pdb, None)
    assert len(released_prs) == len(prs)
    released_prs = await done_prs_facts_loader.load_precomputed_pr_releases(
        df_from_prs(),
        max(s.released.item().replace(tzinfo=timezone.utc) for s in samples) + timedelta(days=1),
        {pr.pr[PullRequest.repository_full_name.name]: ReleaseMatch.tag for pr in prs},
        default_branches, ReleaseSettings({"github.com/src-d/go-git": ReleaseMatchSetting(
            tags="v.*", branches="", events="", match=ReleaseMatch.tag),
        }), prefixer, 1, pdb, None)
    assert released_prs.empty


@with_defer
async def test_discover_update_unreleased_prs_smoke(
        mdb, pdb, rdb, default_branches, release_match_setting_tag, release_loader,
        merged_prs_facts_loader, with_preloading_enabled, prefixer):
    prs = await read_sql_query(
        select([PullRequest]).where(and_(PullRequest.number.in_(range(1000, 1010)),
                                         PullRequest.merged_at.isnot(None))),
        mdb, PullRequest, index=[PullRequest.node_id.name, PullRequest.repository_full_name.name])
    prs[prs[PullRequest.merged_at.name].isnull()] = datetime.now(tz=timezone.utc)
    utc = timezone.utc
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], None, default_branches,
        datetime(2018, 9, 1, tzinfo=utc),
        datetime(2018, 11, 1, tzinfo=utc),
        release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    if pdb.url.dialect == "sqlite":
        await wait_deferred()
        if with_preloading_enabled:
            await pdb.cache.refresh()

    assert len(releases) == 2
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
    empty_rdf = new_released_prs_df()
    await update_unreleased_prs(
        prs, empty_rdf, datetime(2018, 11, 1, tzinfo=utc), {},
        matched_bys, default_branches, release_match_setting_tag, 1, pdb, asyncio.Event())
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], None, default_branches,
        datetime(2018, 11, 1, tzinfo=utc),
        datetime(2018, 11, 20, tzinfo=utc),
        release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    if pdb.url.dialect == "sqlite":
        await wait_deferred()
        if with_preloading_enabled:
            await pdb.cache.refresh()

    assert len(releases) == 1
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
    await update_unreleased_prs(
        prs, empty_rdf, datetime(2018, 11, 20, tzinfo=utc), {},
        matched_bys, default_branches, release_match_setting_tag, 1, pdb, asyncio.Event())
    unreleased_prs = await merged_prs_facts_loader.load_merged_unreleased_pull_request_facts(
        prs, datetime(2018, 11, 20, tzinfo=utc), LabelFilter.empty(), matched_bys,
        default_branches, release_match_setting_tag, prefixer, 1, pdb)
    assert len(unreleased_prs) == 0
    await pdb.execute(update(GitHubMergedPullRequestFacts).values({
        GitHubMergedPullRequestFacts.data: FakeFacts().data,
        GitHubMergedPullRequestFacts.updated_at: datetime.now(timezone.utc),
    }))

    if with_preloading_enabled:
        await pdb.cache.refresh()

    unreleased_prs = await merged_prs_facts_loader.load_merged_unreleased_pull_request_facts(
        prs, datetime(2018, 11, 20, tzinfo=utc), LabelFilter.empty(), matched_bys,
        default_branches, release_match_setting_tag, prefixer, 1, pdb)
    assert set(prs.index) == set(unreleased_prs)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], None, default_branches,
        datetime(2018, 9, 1, tzinfo=utc),
        datetime(2018, 11, 1, tzinfo=utc),
        release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    if pdb.url.dialect == "sqlite":
        await wait_deferred()
        if with_preloading_enabled:
            await pdb.cache.refresh()

    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
    unreleased_prs = await merged_prs_facts_loader.load_merged_unreleased_pull_request_facts(
        prs, datetime(2018, 11, 1, tzinfo=utc), LabelFilter.empty(), matched_bys, default_branches,
        ReleaseSettings({"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="", tags="v.*", events=".*", match=ReleaseMatch.tag)}),
        prefixer, 1, pdb)
    assert len(unreleased_prs) == 0
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], None, default_branches,
        datetime(2019, 1, 29, tzinfo=utc),
        datetime(2019, 2, 1, tzinfo=utc),
        release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    assert len(releases) == 2
    assert matched_bys == {"src-d/go-git": ReleaseMatch.tag}
    unreleased_prs = await merged_prs_facts_loader.load_merged_unreleased_pull_request_facts(
        prs, datetime(2019, 2, 1, tzinfo=utc), LabelFilter.empty(), matched_bys, default_branches,
        release_match_setting_tag, prefixer, 1, pdb)
    assert len(unreleased_prs) == 0


@with_defer
async def test_discover_update_unreleased_prs_released(
        mdb, pdb, rdb, dag, default_branches, release_match_setting_tag, release_loader,
        merged_prs_facts_loader, with_preloading_enabled, prefixer):
    prs = await read_sql_query(
        select([PullRequest]).where(and_(PullRequest.number.in_(range(1000, 1010)),
                                         PullRequest.merged_at.isnot(None))),
        mdb, PullRequest, index=[PullRequest.node_id.name, PullRequest.repository_full_name.name])
    prs["dead"] = False
    prs[prs[PullRequest.merged_at.name].isnull()] = datetime.now(tz=timezone.utc)
    utc = timezone.utc
    time_from = datetime(2018, 10, 1, tzinfo=utc)
    time_to = datetime(2018, 12, 1, tzinfo=utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], None, default_branches,
        time_from,
        time_to,
        release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    released_prs, _, _ = await PullRequestToReleaseMapper.map_prs_to_releases(
        prs, releases, matched_bys, pd.DataFrame(columns=[Branch.commit_id.name]),
        default_branches, time_to,
        dag, release_match_setting_tag, prefixer, 1, (6366825,), mdb, pdb, None)
    await wait_deferred()
    if with_preloading_enabled:
        await pdb.cache.refresh()

    await update_unreleased_prs(
        prs, released_prs, time_to, {},
        matched_bys, default_branches, release_match_setting_tag, 1, pdb, asyncio.Event())
    await pdb.execute(update(GitHubMergedPullRequestFacts).values({
        GitHubMergedPullRequestFacts.data: FakeFacts().data,
        GitHubMergedPullRequestFacts.updated_at: datetime.now(timezone.utc),
    }))
    if with_preloading_enabled:
        await pdb.cache.refresh()

    unreleased_prs = await merged_prs_facts_loader.load_merged_unreleased_pull_request_facts(
        prs, time_to, LabelFilter.empty(), matched_bys, default_branches,
        release_match_setting_tag, prefixer, 1, pdb)
    assert len(unreleased_prs) == 1
    assert next(iter(unreleased_prs.keys())) == (163287, "src-d/go-git")
    unreleased_prs = await merged_prs_facts_loader.load_merged_unreleased_pull_request_facts(
        prs, datetime(2018, 11, 1, tzinfo=utc), LabelFilter.empty(),
        matched_bys, default_branches, release_match_setting_tag, prefixer, 1, pdb)
    assert len(unreleased_prs) == 7


@pytest.fixture(scope="function")
@with_defer
async def precomputed_merged_unreleased(
        mdb, pdb, rdb, dag, default_branches, release_match_setting_tag, release_loader,
        with_preloading_enabled, prefixer):
    postgres = pdb.url.dialect == "postgresql"
    prs = await read_sql_query(
        select([PullRequest]).where(and_(PullRequest.number.in_(range(1000, 1010)),
                                         PullRequest.merged_at.isnot(None))),
        mdb, PullRequest, index=[PullRequest.node_id.name, PullRequest.repository_full_name.name])
    prs["dead"] = False
    prs[prs[PullRequest.merged_at.name].isnull()] = datetime.now(tz=timezone.utc)
    utc = timezone.utc
    time_from = datetime(2018, 10, 1, tzinfo=utc)
    time_to = datetime(2018, 12, 1, tzinfo=utc)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], None, default_branches,
        time_from,
        time_to,
        release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    released_prs, _, _ = await PullRequestToReleaseMapper.map_prs_to_releases(
        prs, releases, matched_bys, pd.DataFrame(columns=[Branch.commit_id.name]),
        default_branches, time_to,
        dag, release_match_setting_tag, prefixer, 1, (6366825,), mdb, pdb, None)
    await wait_deferred()
    if with_preloading_enabled:
        await pdb.cache.refresh()

    await update_unreleased_prs(
        prs, released_prs, time_to, {},
        matched_bys, default_branches, release_match_setting_tag, 1, pdb, asyncio.Event())

    await pdb.execute(update(GitHubMergedPullRequestFacts).values({
        GitHubMergedPullRequestFacts.data: FakeFacts().data,
        GitHubMergedPullRequestFacts.updated_at: datetime.now(timezone.utc),
        GitHubMergedPullRequestFacts.activity_days: [
            datetime(2018, 10, 15, tzinfo=timezone.utc) if postgres else "2018-10-15",
        ],
    }))
    if with_preloading_enabled:
        await pdb.cache.refresh()
    return prs, matched_bys


@with_defer
async def test_discover_update_unreleased_prs_exclude_inactive(
        mdb, pdb, rdb, dag, default_branches, release_match_setting_tag,
        merged_prs_facts_loader, prefixer, precomputed_merged_unreleased):
    utc = timezone.utc
    prs, matched_bys = precomputed_merged_unreleased
    unreleased_prs = await merged_prs_facts_loader.load_merged_unreleased_pull_request_facts(
        prs, datetime(2018, 11, 1, tzinfo=utc), LabelFilter.empty(),
        matched_bys, default_branches, release_match_setting_tag, prefixer, 1, pdb,
        time_from=datetime(2018, 10, 14, tzinfo=utc), exclude_inactive=True)
    assert len(unreleased_prs) == 7
    unreleased_prs = await merged_prs_facts_loader.load_merged_unreleased_pull_request_facts(
        prs, datetime(2018, 11, 1, tzinfo=utc), LabelFilter.empty(),
        matched_bys, default_branches, release_match_setting_tag, prefixer, 1, pdb,
        time_from=datetime(2018, 10, 16, tzinfo=utc), exclude_inactive=True)
    assert len(unreleased_prs) == 0


@pytest.mark.xfail(with_preloading_env, reason="Not supported in the preloader")
@with_defer
async def test_discover_update_unreleased_prs_deployments(
        mdb, pdb, rdb, dag, branches, default_branches, release_match_setting_tag,
        merged_prs_facts_loader, prefixer, precomputed_merged_unreleased):
    await rdb.execute(update(DeploymentNotification).values({
        DeploymentNotification.started_at: datetime(2018, 10, 30, 10, 15, tzinfo=timezone.utc),
        DeploymentNotification.finished_at: datetime(2018, 10, 30, 12, 15, tzinfo=timezone.utc),
        DeploymentNotification.updated_at: datetime.now(timezone.utc),
    }))
    await mine_deployments(
        [40550], {},
        datetime(2018, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        branches, default_branches, prefixer,
        1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    utc = timezone.utc
    prs, matched_bys = precomputed_merged_unreleased
    unreleased_prs = await merged_prs_facts_loader.load_merged_unreleased_pull_request_facts(
        prs, datetime(2018, 11, 1, tzinfo=utc), LabelFilter.empty(),
        matched_bys, default_branches, release_match_setting_tag, prefixer, 1, pdb,
        time_from=datetime(2018, 10, 16, tzinfo=utc), exclude_inactive=True)
    assert len(unreleased_prs) == 7  # 0 without deployments


@with_defer
async def test_discover_old_merged_unreleased_prs_smoke(
        metrics_calculator_factory, mdb, pdb, rdb, dag, release_match_setting_tag, cache,
        prefixer, release_loader, default_branches):
    metrics_calculator = metrics_calculator_factory(1, (6366825,), with_cache=True)
    metrics_time_from = datetime(2018, 1, 1, tzinfo=timezone.utc)
    metrics_time_to = datetime(2020, 5, 1, tzinfo=timezone.utc)
    await metrics_calculator.calc_pull_request_facts_github(
        metrics_time_from, metrics_time_to, {"src-d/go-git"}, {}, LabelFilter.empty(),
        JIRAFilter.empty(), False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, False, False,
    )
    await wait_deferred()
    unreleased_time_from = datetime(2018, 11, 1, tzinfo=timezone.utc)
    unreleased_time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    unreleased_prs = await discover_inactive_merged_unreleased_prs(
        unreleased_time_from, unreleased_time_to, {"src-d/go-git"},
        {PRParticipationKind.MERGER: {"mcuadros"}}, LabelFilter.empty(), {},
        release_match_setting_tag, prefixer, 1, pdb, cache)
    await wait_deferred()
    assert len(unreleased_prs) == 11
    unreleased_prs = await read_sql_query(
        select([PullRequest]).where(PullRequest.node_id.in_(unreleased_prs)),
        mdb, PullRequest, index=[PullRequest.node_id.name, PullRequest.repository_full_name.name])
    assert (unreleased_prs[PullRequest.merged_at.name] >
            datetime(2018, 10, 17, tzinfo=timezone.utc)).all()
    unreleased_prs = await discover_inactive_merged_unreleased_prs(
        unreleased_time_from, unreleased_time_to, {"src-d/go-git"},
        {PRParticipationKind.MERGER: {"mcuadros"}}, LabelFilter.empty(), {},
        release_match_setting_tag, prefixer, 1, None, cache)
    assert len(unreleased_prs) == 11
    unreleased_prs = await read_sql_query(
        select([PullRequest]).where(PullRequest.node_id.in_(unreleased_prs)),
        mdb, PullRequest, index=[PullRequest.node_id.name, PullRequest.repository_full_name.name])
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], None, None, metrics_time_from, unreleased_time_to,
        release_match_setting_tag, LogicalRepositorySettings.empty(), prefixer,
        1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    unreleased_prs["dead"] = False
    released_prs, _, _ = await PullRequestToReleaseMapper.map_prs_to_releases(
        unreleased_prs, releases, matched_bys, pd.DataFrame(columns=[Branch.commit_id.name]),
        default_branches, unreleased_time_to, dag, release_match_setting_tag,
        prefixer, 1, (6366825,), mdb, pdb, cache)
    await wait_deferred()
    assert released_prs.empty
    unreleased_time_from = datetime(2018, 11, 19, tzinfo=timezone.utc)
    unreleased_time_to = datetime(2018, 11, 20, tzinfo=timezone.utc)
    unreleased_prs = await discover_inactive_merged_unreleased_prs(
        unreleased_time_from, unreleased_time_to, {"src-d/go-git"},
        {PRParticipationKind.MERGER: {"mcuadros"}}, LabelFilter.empty(), {},
        release_match_setting_tag, prefixer, 1, pdb, cache)
    assert not unreleased_prs


@with_defer
async def test_discover_old_merged_unreleased_prs_labels(
        metrics_calculator_factory, mdb, pdb, rdb, release_match_setting_tag,
        prefixer, cache):
    metrics_calculator = metrics_calculator_factory(1, (6366825,), with_cache=True)
    metrics_time_from = datetime(2018, 5, 1, tzinfo=timezone.utc)
    metrics_time_to = datetime(2019, 1, 1, tzinfo=timezone.utc)
    await metrics_calculator.calc_pull_request_facts_github(
        metrics_time_from, metrics_time_to, {"src-d/go-git"}, {}, LabelFilter.empty(),
        JIRAFilter.empty(), False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, False, False,
    )
    await wait_deferred()
    unreleased_time_from = datetime(2018, 9, 19, tzinfo=timezone.utc)
    unreleased_time_to = datetime(2018, 9, 30, tzinfo=timezone.utc)
    unreleased_prs = await discover_inactive_merged_unreleased_prs(
        unreleased_time_from, unreleased_time_to, {"src-d/go-git"},
        {}, LabelFilter({"bug", "plumbing"}, set()), {}, release_match_setting_tag,
        prefixer, 1, pdb, cache)
    assert unreleased_prs.keys() == {163253, 163272}
    unreleased_prs = await discover_inactive_merged_unreleased_prs(
        unreleased_time_from, unreleased_time_to, {"src-d/go-git"},
        {}, LabelFilter({"enhancement"}, set()), {}, release_match_setting_tag,
        prefixer, 1, pdb, cache)
    assert unreleased_prs.keys() == {163273}
    unreleased_prs = await discover_inactive_merged_unreleased_prs(
        unreleased_time_from, unreleased_time_to, {"src-d/go-git"},
        {}, LabelFilter({"bug"}, {"ssh"}), {}, release_match_setting_tag,
        prefixer, 1, pdb, cache)
    assert unreleased_prs.keys() == {163253}


async def test_store_precomputed_done_none_assert(pdb, pr_samples):
    samples = pr_samples(1)  # type: Sequence[PullRequestFacts]
    settings = ReleaseSettings({
        "github.com/one": ReleaseMatchSetting("{{default}}", ".*", ".*", ReleaseMatch.tag),
    })
    default_branches = {"one": "master"}
    prs = [MinedPullRequest(
        pr={PullRequest.created_at.name: samples[0].merged,
            PullRequest.repository_full_name.name: "one",
            PullRequest.user_login.name: "vmarkovtsev",
            PullRequest.merged_by_login.name: "mcuadros",
            PullRequest.number.name: 1,
            PullRequest.node_id.name: 100500},
        release={matched_by_column: settings.native["one"],
                 Release.author.name: "foo",
                 Release.url.name: "https://release",
                 Release.node_id.name: 777},
        comments=gen_dummy_df(samples[0].first_comment_on_first_review),
        commits=pd.DataFrame.from_records(
            [["mcuadros", "mcuadros", samples[0].first_commit]],
            columns=[
                PullRequestCommit.committer_login.name,
                PullRequestCommit.author_login.name,
                PullRequestCommit.committed_date.name,
            ],
        ),
        reviews=gen_dummy_df(samples[0].first_comment_on_first_review),
        review_comments=gen_dummy_df(samples[0].first_comment_on_first_review),
        review_requests=gen_dummy_df(samples[0].first_review_request),
        labels=pd.DataFrame.from_records([["bug"]], columns=["name"]),
        jiras=pd.DataFrame(),
        deployments=None,
    )]
    await store_precomputed_done_facts(prs, [None], default_branches, settings, 1, pdb)

    def with_repository_full_name(s):
        s.repository_full_name = "one"
        return s

    with pytest.raises(AssertionError):
        await store_precomputed_done_facts(
            prs, [with_repository_full_name(s) for s in samples],
            default_branches, settings, 1, pdb)


@with_defer
async def test_store_merged_unreleased_pull_request_facts_smoke(
        mdb, pdb, rdb, default_branches, release_match_setting_tag, prefixer):
    prs, dfs, facts, matched_bys, deps_task, cr_task = await _fetch_pull_requests(
        {"src-d/go-git": set(range(1000, 1010))},
        release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    deps_task.cancel()
    cr_task.cancel()
    for pr in prs:
        if pr.pr[PullRequest.merged_at.name] is None:
            pr.pr[PullRequest.merged_at.name] = datetime.now(tz=timezone.utc)
    dfs.prs.loc[dfs.prs[PullRequest.merged_at.name].isnull(), PullRequest.merged_at.name] = \
        datetime.now(tz=timezone.utc)
    event = asyncio.Event()
    await update_unreleased_prs(
        dfs.prs, new_released_prs_df(), datetime(2018, 11, 1, tzinfo=timezone.utc), {},
        matched_bys, default_branches, release_match_setting_tag, 1, pdb, event)
    samples_good, samples_bad = [], []
    for pr in prs:
        f = facts[(pr.pr[PullRequest.node_id.name], "src-d/go-git")]
        fields = dict(f)
        if f.merged is None:
            fields["merged"] = datetime.now(tz=timezone.utc)
        fields.update({
            "released": None,
            "releaser": None,
        })
        samples_good.append(PullRequestFacts.from_fields(**fields))
        samples_bad.append(f)
    with pytest.raises(AssertionError):
        await store_merged_unreleased_pull_request_facts(
            zip(prs, samples_bad), matched_bys, default_branches, release_match_setting_tag,
            1, pdb, event)
    await store_merged_unreleased_pull_request_facts(
        zip(prs, samples_good), matched_bys, default_branches, release_match_setting_tag,
        1, pdb, event)
    true_dict = {pr.pr[PullRequest.node_id.name]: s for pr, s in zip(prs, samples_good)}
    ghmprf = GitHubMergedPullRequestFacts
    rows = await pdb.fetch_all(select([ghmprf]))
    assert len(rows) == 10
    for row in rows:
        assert isinstance(row[ghmprf.activity_days.name], list)
        assert len(row[ghmprf.activity_days.name]) > 0

    new_dict = {
        r[ghmprf.pr_node_id.name]: PullRequestFacts(
            data=r[ghmprf.data.name],
            repository_full_name="src-d/go-git",
            author=r[ghmprf.author.name],
            merger=r[ghmprf.merger.name])
        for r in rows
    }
    assert true_dict == new_dict


@with_defer
async def test_store_open_pull_request_facts_smoke(
        mdb, pdb, rdb, release_match_setting_tag, open_prs_facts_loader,
        with_preloading_enabled, prefixer):
    prs, dfs, facts, _, deps_task, cr_task = await _fetch_pull_requests(
        {"src-d/go-git": set(range(1000, 1010))},
        release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    deps_task.cancel()
    cr_task.cancel()
    with pytest.raises(AssertionError):
        await store_open_pull_request_facts(
            zip(prs, (facts[(pr.pr[PullRequest.node_id.name], "src-d/go-git")] for pr in prs)),
            1, pdb)
    samples = []
    true_dict = {}
    authors = {}
    for pr in prs:
        authors[pr.pr[PullRequest.node_id.name]] = pr.pr[PullRequest.user_login.name]
        f = facts[(pr.pr[PullRequest.node_id.name], "src-d/go-git")]
        fields = dict(f)
        fields.update({
            "closed": None,
            "repository_full_name": "src-d/go-git",
            "merger": None,
            "releaser": None,
        })
        fields["closed"] = None
        f = PullRequestFacts.from_fields(**fields)
        samples.append(f)
        true_dict[(pr.pr[PullRequest.node_id.name], "src-d/go-git")] = f
    dfs.prs[PullRequest.closed_at.name] = None
    await store_open_pull_request_facts(zip(prs, samples), 1, pdb)
    if with_preloading_enabled:
        await pdb.cache.refresh()
    ghoprf = GitHubOpenPullRequestFacts
    rows = await pdb.fetch_all(select([ghoprf]))
    assert len(rows) == 10
    new_dict = {}
    for row in rows:
        assert isinstance(row[ghoprf.activity_days.name], list)
        assert len(row[ghoprf.activity_days.name]) > 0
        new_dict[(row[ghoprf.pr_node_id.name], "src-d/go-git")] = PullRequestFacts(
            data=row[ghoprf.data.name],
            repository_full_name="src-d/go-git",
            author=authors[row[ghoprf.pr_node_id.name]])
    assert true_dict == new_dict

    loaded_facts = await open_prs_facts_loader.load_open_pull_request_facts(
        dfs.prs, {"src-d/go-git"}, 1, pdb)
    for facts in loaded_facts.values():
        assert facts.repository_full_name == "src-d/go-git"
    assert true_dict == loaded_facts

    loaded_facts = await open_prs_facts_loader.load_open_pull_request_facts_unfresh(
        dfs.prs.index.get_level_values(0),
        datetime(2016, 1, 1), datetime(2020, 1, 1),
        {"src-d/go-git"}, True, authors, 1, pdb)
    for (node_id, _), facts in loaded_facts.items():
        assert facts.repository_full_name == "src-d/go-git"
        assert facts.author == authors[node_id]
    assert true_dict == loaded_facts
    loaded_facts = await open_prs_facts_loader.load_open_pull_request_facts_unfresh(
        dfs.prs.index.get_level_values(0),
        datetime(2019, 11, 1), datetime(2020, 1, 1),
        {"src-d/go-git"}, True, authors, 1, pdb)
    assert len(loaded_facts) == 0


@with_only_master_branch
@with_defer
async def test_rescan_prs_mark_force_push_dropped(
        mdb_rw, pdb, branches, default_branches, pr_samples):
    mdb = mdb_rw
    samples, prs, settings = _gen_one_pr(pr_samples)
    prs[0].pr[PullRequest.node_id.name] = 163437

    def with_mutables(s):
        s.repository_full_name = "src-d/go-git"
        s.author = "vmarkovtsev"
        s.merger = "mcuadros"
        s.releaser = "mcarmonaa"
        return s

    await store_precomputed_done_facts(
        prs, [with_mutables(s) for s in samples],
        default_branches, settings, 1, pdb)
    release_match = await pdb.fetch_val(select([GitHubDonePullRequestFacts.release_match]))
    assert release_match == "branch|master"
    node_ids = await delete_force_push_dropped_prs(
        ["src-d/go-git"], branches, 1, (6366825,), mdb, pdb, None)
    assert list(node_ids) == [163437]
    release_match = await pdb.fetch_val(select([GitHubDonePullRequestFacts.release_match]))
    assert release_match is None


async def test_load_precomputed_done_facts_ids(
        pdb, default_branches, pr_samples, done_prs_facts_loader, prefixer):
    sfacts, prs, settings = _gen_one_pr(pr_samples)

    def with_mutables(s):
        s.repository_full_name = "src-d/go-git"
        s.author = "vmarkovtsev"
        s.merger = "mcuadros"
        s.releaser = "mcarmonaa"
        return s

    sfacts = [with_mutables(s) for s in sfacts]
    await store_precomputed_done_facts(prs, sfacts, default_branches, settings, 1, pdb)
    pfacts, ambiguous = await done_prs_facts_loader.load_precomputed_done_facts_ids(
        [prs[0].pr[PullRequest.node_id.name]],
        default_branches, settings, prefixer, 1, pdb)
    assert sfacts == list(pfacts.values())
    assert len(ambiguous["src-d/go-git"]) == 1
