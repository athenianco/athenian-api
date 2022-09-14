import dataclasses
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas._testing import assert_frame_equal
from sqlalchemy import insert

from athenian.api.db import Database
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.precomputed_prs import store_precomputed_done_facts
from athenian.api.internal.miners.github.released_pr import matched_by_column
from athenian.api.internal.miners.jira.issue import (
    ISSUE_PRS_BEGAN,
    ISSUE_PRS_RELEASED,
    PullRequestJiraMapper,
    _fetch_released_prs,
    fetch_jira_issues,
    generate_jira_prs_query,
)
from athenian.api.internal.miners.types import (
    JIRAEntityToFetch,
    MinedPullRequest,
    PullRequestCheckRun,
    PullRequestFacts,
    PullRequestJIRADetails,
)
from athenian.api.internal.settings import (
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseMatchSetting,
    ReleaseSettings,
)
from athenian.api.models.metadata.github import PullRequest, PullRequestCommit, Release
from athenian.precomputer.db.models import GitHubDonePullRequestFacts
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_MD_ACCOUNT_ID


class TestFetchJIRAIssues:
    @with_defer
    async def test_releases(
        self,
        metrics_calculator_factory,
        mdb,
        pdb,
        rdb,
        default_branches,
        release_match_setting_tag,
        prefixer,
        bots,
        cache,
    ):
        metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
        time_from = datetime(2016, 1, 1, tzinfo=timezone.utc)
        time_to = datetime(2021, 1, 1, tzinfo=timezone.utc)
        await metrics_calculator_no_cache.calc_pull_request_facts_github(
            time_from,
            time_to,
            {"src-d/go-git"},
            {},
            LabelFilter.empty(),
            JIRAFilter.empty(),
            False,
            bots,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            prefixer,
            False,
            0,
        )
        await wait_deferred()
        args = [
            time_from,
            time_to,
            JIRAFilter.empty().replace(account=1, projects=["10003", "10009"]),
            False,
            [],
            [],
            [],
            False,
            default_branches,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            1,
            (6366825,),
            mdb,
            pdb,
            cache,
        ]
        issues = await fetch_jira_issues(*args)

        assert issues[ISSUE_PRS_BEGAN].notnull().sum() == 55  # 56 without cleaning
        assert issues[ISSUE_PRS_RELEASED].notnull().sum() == 54  # 55 without cleaning
        assert (
            issues[ISSUE_PRS_RELEASED][issues[ISSUE_PRS_RELEASED].notnull()]
            > issues[ISSUE_PRS_BEGAN][issues[ISSUE_PRS_RELEASED].notnull()]
        ).all()

        await wait_deferred()
        args[-3] = args[-2] = None
        cached_issues = await fetch_jira_issues(*args)
        assert_frame_equal(issues, cached_issues)
        args[-7] = ReleaseSettings({})
        args[-3] = mdb
        args[-2] = pdb
        ghdprf = GitHubDonePullRequestFacts
        await pdb.execute(
            insert(ghdprf).values(
                {
                    ghdprf.acc_id: 1,
                    ghdprf.pr_node_id: 163250,
                    ghdprf.repository_full_name: "src-d/go-git",
                    ghdprf.release_match: "branch|master",
                    ghdprf.pr_done_at: datetime(2018, 7, 17, tzinfo=timezone.utc),
                    ghdprf.pr_created_at: datetime(2018, 5, 17, tzinfo=timezone.utc),
                    ghdprf.number: 1,
                    ghdprf.updated_at: datetime.now(timezone.utc),
                    ghdprf.format_version: ghdprf.__table__.columns[
                        ghdprf.format_version.key
                    ].default.arg,
                    ghdprf.data: b"test",
                },
            ),
        )
        issues = await fetch_jira_issues(*args)
        assert issues[ISSUE_PRS_BEGAN].notnull().sum() == 55
        assert issues[ISSUE_PRS_RELEASED].notnull().sum() == 55

    @with_defer
    async def test_no_times(
        self,
        mdb,
        pdb,
        default_branches,
        release_match_setting_tag,
        cache,
    ):
        args = [
            None,
            None,
            JIRAFilter.empty().replace(account=1, projects=["10003", "10009"]),
            False,
            [],
            [],
            [],
            False,
            default_branches,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            1,
            (6366825,),
            mdb,
            pdb,
            cache,
        ]
        issues = await fetch_jira_issues(*args)
        await wait_deferred()
        cached_issues = await fetch_jira_issues(*args)
        assert_frame_equal(issues, cached_issues)

    @with_defer
    async def test_none_assignee(
        self,
        mdb,
        pdb,
        default_branches,
        release_match_setting_tag,
        cache,
    ):
        args = [
            None,
            None,
            JIRAFilter.empty().replace(account=1, projects=["10003", "10009"]),
            False,
            [],
            ["vadim markovtsev", None],
            [],
            False,
            default_branches,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            1,
            (6366825,),
            mdb,
            pdb,
            cache,
        ]
        issues = await fetch_jira_issues(*args)
        assert len(issues) == 716  # 730 without cleaning
        await wait_deferred()
        cached_issues = await fetch_jira_issues(*args)
        assert_frame_equal(issues, cached_issues)


class TestFetchReleasedPRs:
    async def test_fetch_released_prs_release_settings_events(
        self,
        pr_samples,
        pdb,
        done_prs_facts_loader,
        prefixer,
    ):
        samples = pr_samples(12)  # type: Sequence[PullRequestFacts]
        names = ["one", "two", "three"]
        settings = ReleaseSettings(
            {
                "github.com/"
                + k: ReleaseMatchSetting("{{default}}", ".*", ".*", ReleaseMatch.tag_or_branch)
                for k in names
            },
        )
        default_branches = {k: "master" for k in names}
        prs = [
            MinedPullRequest(
                pr={
                    PullRequest.created_at.name: s.created,
                    PullRequest.repository_full_name.name: names[i % len(names)],
                    PullRequest.user_login.name: ["vmarkovtsev", "marnovo"][i % 2],
                    PullRequest.user_node_id.name: [40020, 39792][i % 2],
                    PullRequest.merged_by_login.name: "mcuadros",
                    PullRequest.merged_by_id.name: 39789,
                    PullRequest.number.name: i + 1,
                    PullRequest.node_id.name: i + 100500,
                },
                release={
                    matched_by_column: match,
                    Release.author.name: ["marnovo", "mcarmonaa"][i % 2],
                    Release.author_node_id.name: [39792, 39818][i % 2],
                    Release.url.name: "https://release",
                    Release.node_id.name: i,
                },
                comments=self._gen_dummy_df(s.first_comment_on_first_review),
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
                reviews=self._gen_dummy_df(s.first_comment_on_first_review),
                review_comments=self._gen_dummy_df(s.first_comment_on_first_review),
                review_requests=self._gen_dummy_df(s.first_review_request),
                labels=pd.DataFrame.from_records(
                    ([["bug"]], [["feature"]])[i % 2], columns=["name"],
                ),
                jiras=pd.DataFrame(),
                deployments=None,
                check_run={PullRequestCheckRun.f.name: None},
            )
            for match in (ReleaseMatch.tag, ReleaseMatch.event)
            for i, s in enumerate(samples)
        ]

        def with_mutables(s, i):
            s.repository_full_name = names[i % len(names)]
            s.author = ["vmarkovtsev", "marnovo"][i % 2]
            s.merger = "mcuadros"
            s.releaser = ["marnovo", "mcarmonaa"][i % 2]
            return s

        await store_precomputed_done_facts(
            prs,
            [with_mutables(s, i) for i, s in enumerate(samples)] * 2,
            datetime(2050, 1, 1, tzinfo=timezone.utc),
            default_branches,
            settings,
            1,
            pdb,
        )

        new_prs = await _fetch_released_prs(
            [i + 100500 for i in range(len(samples))],
            default_branches,
            settings,
            1,
            pdb,
        )

        assert len(new_prs) == len(samples)

    @classmethod
    def _gen_dummy_df(cls, dt: datetime) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [["vmarkovtsev", 40020, dt, dt]],
            columns=["user_login", "user_node_id", "created_at", "submitted_at"],
        )


class TestGenerateJIRAPRsQuery:
    async def test_priority_filter(self, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.PullRequestFactory(node_id=1),
                md_factory.PullRequestFactory(node_id=2),
                md_factory.PullRequestFactory(node_id=3),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=1, jira_id="1"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=1, jira_id="2"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=2, jira_id="3"),
                md_factory.JIRAIssueFactory(id="1", project_id="1", priority_name="PR1"),
                md_factory.JIRAIssueFactory(id="2", project_id="1", priority_name="PR1"),
                md_factory.JIRAIssueFactory(id="3", project_id="1", priority_name="PR2"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            jira_filter = JIRAFilter(
                1,
                frozenset(("1",)),
                LabelFilter.empty(),
                frozenset(),
                frozenset(),
                {"PR1", "PR2"},
                False,
                False,
            )
            res = await self._fetch_with_jira_filter(mdb_rw, jira_filter)
            assert [r[PullRequest.node_id.name] for r in res] == [1, 1, 2]

            jira_filter = dataclasses.replace(jira_filter, priorities={"PR1"})
            res = await self._fetch_with_jira_filter(mdb_rw, jira_filter)
            assert [r[PullRequest.node_id.name] for r in res] == [1, 1]

            jira_filter = dataclasses.replace(jira_filter, priorities={"PR2"})
            res = await self._fetch_with_jira_filter(mdb_rw, jira_filter)
            assert [r[PullRequest.node_id.name] for r in res] == [2]

    async def test_multiple_filters(self, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.PullRequestFactory(node_id=1),
                md_factory.PullRequestFactory(node_id=2),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=1, jira_id="1"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=2, jira_id="2"),
                md_factory.JIRAIssueFactory(id="1", project_id="P1", priority_name="PR1"),
                md_factory.JIRAIssueFactory(id="2", project_id="P2", priority_name="PR2"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            jira_filter = JIRAFilter(
                1, ("P1",), LabelFilter.empty(), frozenset(), frozenset(), {"PR1"}, False, False,
            )
            res = await self._fetch_with_jira_filter(mdb_rw, jira_filter)
            assert [r[PullRequest.node_id.name] for r in res] == [1]

            jira_filter = dataclasses.replace(jira_filter, priorities={"PR2"})
            res = await self._fetch_with_jira_filter(mdb_rw, jira_filter)
            assert [r[PullRequest.node_id.name] for r in res] == []

    @classmethod
    async def _fetch_with_jira_filter(cls, mdb: Database, jira_filter):
        query = await generate_jira_prs_query(
            [], jira_filter, (DEFAULT_MD_ACCOUNT_ID,), mdb, None,
        )
        query = query.order_by(PullRequest.node_id)
        return await mdb.fetch_all(query)


class TestPullRequestJiraMapper:
    async def test_load_and_apply_to_pr_facts(self, mdb_rw: Database, sdb: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestJiraIssuesFactory(node_id=10, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=11, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=11, jira_id="21"),
                md_factory.JIRAIssueFactory(
                    id="20", project_id="P0", key="I20", priority_id="PR", type_id="T",
                ),
                md_factory.JIRAIssueFactory(
                    id="21", project_id="P0", key="I21", priority_id="PR", type_id="T",
                ),
            ]

            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            prs = {
                (10, "repo0"): PullRequestFacts(b""),
                (10, "repo1"): PullRequestFacts(b""),
                (11, "repo1"): PullRequestFacts(b""),
            }
            await PullRequestJiraMapper.load_and_apply_to_pr_facts(
                prs, JIRAEntityToFetch.EVERYTHING(), (DEFAULT_MD_ACCOUNT_ID,), mdb_rw,
            )

        assert sorted(prs) == [(10, "repo0"), (10, "repo1"), (11, "repo1")]

        assert prs[(10, "repo0")].jira == PullRequestJIRADetails(
            ids=["I20"], projects=[b"P0"], priorities=[b"PR"], types=[b"T"],
        )
        assert prs[(10, "repo0")].jira == prs[(10, "repo1")].jira
        assert_array_equal(prs[(11, "repo1")].jira.ids, np.array(["I20", "I21"]))
        assert_array_equal(prs[(11, "repo1")].jira.projects, np.array([b"P0", b"P0"]))
        assert_array_equal(prs[(11, "repo1")].jira.priorities, np.array([b"PR", b"PR"]))
        assert_array_equal(prs[(11, "repo1")].jira.types, np.array([b"T", b"T"]))

    async def test_load_only_issues(self, mdb_rw: Database, sdb: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestJiraIssuesFactory(node_id=10, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=10, jira_id="21"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=11, jira_id="22"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=12, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=13, jira_id="20"),
                md_factory.JIRAIssueFactory(id="20", key="I20"),
                md_factory.JIRAIssueFactory(id="21", key="I21"),
                md_factory.JIRAIssueFactory(id="22", key="I22"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            mapping = await PullRequestJiraMapper.load(
                [10, 11, 12, 14], JIRAEntityToFetch.ISSUES, (DEFAULT_MD_ACCOUNT_ID,), mdb_rw,
            )
            assert sorted(mapping) == [10, 11, 12]
            assert sorted(mapping[10].ids) == ["I20", "I21"]
            assert mapping[11].ids == ["I22"]
            assert mapping[12].ids == ["I20"]

            assert list(mapping[10].projects) == []
            assert list(mapping[10].priorities) == []
            assert list(mapping[10].types) == []

    async def test_load_everything(self, mdb_rw: Database, sdb: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestJiraIssuesFactory(node_id=10, jira_id="20"),
                md_factory.JIRAIssueFactory(
                    id="20", key="I20", project_id="P0", type_id="T0", priority_id="PR0",
                ),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            mapping = await PullRequestJiraMapper.load(
                [10], JIRAEntityToFetch.EVERYTHING(), (DEFAULT_MD_ACCOUNT_ID,), mdb_rw,
            )
            assert list(mapping) == [10]
            assert mapping[10].ids == ["I20"]
            assert mapping[10].projects == [b"P0"]
            assert mapping[10].priorities == [b"PR0"]
            assert mapping[10].types == [b"T0"]

    def test_apply_to_pr_facts(self) -> None:
        facts = {
            (10, "repo0"): PullRequestFacts(b""),
            (10, "repo1"): PullRequestFacts(b""),
            (11, "repo1"): PullRequestFacts(b""),
        }
        mapping = {
            10: PullRequestJIRADetails(["I1"], [b"P0"], [], []),
            11: PullRequestJIRADetails(["I0", "I1"], [b"P0", b"P1"], [], [b"bug", b"task"]),
        }

        PullRequestJiraMapper.apply_to_pr_facts(facts, mapping)
        assert sorted(facts) == [(10, "repo0"), (10, "repo1"), (11, "repo1")]
        assert facts[(10, "repo0")].jira.ids == ["I1"]
        assert facts[(10, "repo0")].jira.projects == [b"P0"]
        assert facts[(10, "repo0")].jira.priorities == []
        assert facts[(10, "repo0")].jira.types == []
        assert facts[(10, "repo1")] == facts[(10, "repo1")]

        assert sorted(facts[(11, "repo1")].jira.ids) == ["I0", "I1"]
        assert sorted(facts[(11, "repo1")].jira.projects) == [b"P0", b"P1"]
        assert facts[(11, "repo1")].jira.priorities == []
        assert sorted(facts[(11, "repo1")].jira.types) == [b"bug", b"task"]

    def test_apply_empty_to_pr_facts(self) -> None:
        facts = {(10, "r0"): PullRequestFacts(b"")}
        PullRequestJiraMapper.apply_empty_to_pr_facts(facts)
        assert list(facts) == [(10, "r0")]
        assert not facts[(10, "r0")].jira.ids.size
        assert not facts[(10, "r0")].jira.projects.size
        assert not facts[(10, "r0")].jira.types.size
        assert not facts[(10, "r0")].jira.priorities.size
