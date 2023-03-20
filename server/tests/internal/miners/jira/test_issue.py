import dataclasses
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas._testing import assert_frame_equal

from athenian.api.db import Database
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.jira import JIRAConfig
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.precomputed_prs import store_precomputed_done_facts
from athenian.api.internal.miners.github.released_pr import matched_by_column
from athenian.api.internal.miners.jira.issue import (
    ISSUE_PRS_BEGAN,
    ISSUE_PRS_RELEASED,
    PullRequestJiraMapper,
    _fetch_released_prs,
    fetch_jira_issues,
    fetch_jira_issues_by_keys,
    generate_jira_prs_query,
    resolve_resolved,
    resolve_work_began,
)
from athenian.api.internal.miners.types import (
    JIRAEntityToFetch,
    LoadedJIRADetails,
    MinedPullRequest,
    PullRequestCheckRun,
    PullRequestFacts,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import (
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseMatchSetting,
    ReleaseSettings,
    Settings,
)
from athenian.api.models.metadata.github import PullRequest, PullRequestCommit, Release
from athenian.api.models.metadata.jira import Issue, Status
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import (
    DEFAULT_ACCOUNT_ID,
    DEFAULT_JIRA_ACCOUNT_ID,
    DEFAULT_MD_ACCOUNT_ID,
)
from tests.testutils.factory.precomputed import GitHubDonePullRequestFactsFactory
from tests.testutils.factory.wizards import (
    insert_repo,
    jira_issue_models,
    pr_jira_issue_mappings,
    pr_models,
)
from tests.testutils.time import dt


class TestFetchJIRAIssues:
    @with_defer
    async def test_releases(
        self,
        pr_facts_calculator_factory,
        mdb,
        pdb,
        rdb,
        default_branches,
        release_match_setting_tag,
        prefixer,
        bots,
        cache,
    ):
        pr_facts_calculator_no_cache = pr_facts_calculator_factory(1, (6366825,))
        time_from = datetime(2016, 1, 1, tzinfo=timezone.utc)
        time_to = datetime(2021, 1, 1, tzinfo=timezone.utc)
        await pr_facts_calculator_no_cache(
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
        kwargs = self._kwargs(
            time_from=time_from,
            time_to=time_to,
            default_branches=default_branches,
            release_settings=release_match_setting_tag,
            mdb=mdb,
            pdb=pdb,
            cache=cache,
        )
        issues = await fetch_jira_issues(**kwargs)

        assert issues[ISSUE_PRS_BEGAN].notnull().sum() == 55  # 56 without cleaning
        assert issues[ISSUE_PRS_RELEASED].notnull().sum() == 54  # 55 without cleaning
        assert (
            issues[ISSUE_PRS_RELEASED][issues[ISSUE_PRS_RELEASED].notnull()]
            > issues[ISSUE_PRS_BEGAN][issues[ISSUE_PRS_RELEASED].notnull()]
        ).all()

        await wait_deferred()
        kwargs["mdb"] = kwargs["pdb"] = None
        cached_issues = await fetch_jira_issues(**kwargs)
        assert_frame_equal(issues, cached_issues)
        kwargs["release_settings"] = ReleaseSettings({})
        kwargs["mdb"] = mdb
        kwargs["pdb"] = pdb
        await models_insert(
            pdb,
            GitHubDonePullRequestFactsFactory(
                pr_node_id=163250,
                repository_full_name="src-d/go-git",
                pr_done_at=dt(2018, 7, 17),
                pr_created_at=dt(2018, 5, 17),
                number=1,
            ),
        )
        issues = await fetch_jira_issues(**kwargs)
        assert issues[ISSUE_PRS_BEGAN].notnull().sum() == 55
        assert issues[ISSUE_PRS_RELEASED].notnull().sum() == 55

    @with_defer
    async def test_no_times(self, mdb, pdb, default_branches, release_match_setting_tag, cache):
        kwargs = self._kwargs(
            default_branches=default_branches,
            release_settings=release_match_setting_tag,
            mdb=mdb,
            pdb=pdb,
            cache=cache,
        )
        issues = await fetch_jira_issues(**kwargs)
        await wait_deferred()
        cached_issues = await fetch_jira_issues(**kwargs)
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
        kwargs = self._kwargs(
            assignees=["vadim markovtsev", None],
            default_branches=default_branches,
            release_settings=release_match_setting_tag,
            mdb=mdb,
            pdb=pdb,
            cache=cache,
        )
        issues = await fetch_jira_issues(**kwargs)
        assert len(issues) == 716  # 730 without cleaning
        await wait_deferred()
        cached_issues = await fetch_jira_issues(**kwargs)
        assert_frame_equal(issues, cached_issues)

    async def test_status_categories(self, mdb, pdb, default_branches, release_match_setting_tag):
        jira_filter = JIRAFilter.empty().replace(
            account=DEFAULT_JIRA_ACCOUNT_ID, projects=["10003", "10009"],
        )
        kwargs = self._kwargs(
            time_from=dt(2021, 1, 1),
            time_to=dt(2021, 7, 1),
            default_branches=default_branches,
            release_settings=release_match_setting_tag,
            mdb=mdb,
            pdb=pdb,
        )
        all_issues = await fetch_jira_issues(**kwargs)
        assert len(all_issues) == 125

        kwargs["jira_filter"] = jira_filter.replace(
            status_categories=frozenset([Status.CATEGORY_TODO]),
        )
        todo_issues = await fetch_jira_issues(**kwargs)
        assert len(todo_issues) == 116

        kwargs["jira_filter"] = jira_filter.replace(
            status_categories=frozenset([Status.CATEGORY_IN_PROGRESS]),
        )
        in_progress_issues = await fetch_jira_issues(**kwargs)
        assert len(in_progress_issues) == 9

    @with_defer
    async def test_time_from_handle_pr_release_time(
        self,
        sdb,
        mdb_rw,
        pdb,
        default_branches,
        pr_facts_calculator_factory,
    ):
        kwargs = self._kwargs(
            time_from=dt(2023, 2, 1),
            time_to=dt(2023, 3, 1),
            jira_filter=JIRAFilter.empty().replace(
                account=DEFAULT_JIRA_ACCOUNT_ID, projects=["1"],
            ),
            default_branches=default_branches,
            mdb=mdb_rw,
            pdb=pdb,
        )

        issue_kwargs = {"project_id": "1", "created": dt(2023, 1, 1)}
        pr_kwargs = {
            "repository_full_name": "org/repo",
            "created_at": dt(2023, 1, 1),
        }
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=99, full_name="org/repo")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
            models = [
                md_factory.JIRAProjectFactory(id="1", key="P1"),
                *jira_issue_models("1", resolved=dt(2023, 1, 30), **issue_kwargs),
                *jira_issue_models("2", resolved=dt(2023, 2, 2), **issue_kwargs),
                *jira_issue_models("3", resolved=dt(2023, 2, 5), **issue_kwargs),
                *jira_issue_models("4", resolved=dt(2023, 1, 30), **issue_kwargs),
                *pr_models(99, 1, 1, closed_at=dt(2023, 2, 2), **pr_kwargs),
                *pr_models(99, 4, 4, closed_at=dt(2023, 1, 30), **pr_kwargs),
                *pr_jira_issue_mappings((1, "1"), (4, "4")),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            prefixer = await Prefixer.load((DEFAULT_MD_ACCOUNT_ID,), mdb_rw, None)
            settings = Settings.from_account(1, prefixer, sdb, mdb_rw, None, None)
            release_settings = await settings.list_release_matches()

            pr_facts_calculator_no_cache = pr_facts_calculator_factory(1, (DEFAULT_MD_ACCOUNT_ID,))
            # filtering by PR release time needs pdb, let facts calculator fill it
            await pr_facts_calculator_no_cache(
                dt(2022, 10, 1),
                dt(2024, 3, 1),
                {"org/repo"},
                {},
                LabelFilter.empty(),
                JIRAFilter.empty(),
                False,
                {},
                release_settings,
                LogicalRepositorySettings.empty(),
                prefixer,
                False,
                0,
            )
            await wait_deferred()
            kwargs["release_settings"] = release_settings
            issues = await fetch_jira_issues(**kwargs)
            # issue 1 is included because its PR is released on Feb.
            # issue 4 is excluded because its PR is released on Jan.
            assert sorted(issues.index.values) == [b"1", b"2", b"3"]

    @with_defer
    async def test_mapped_prs_from_multiple_accounts(
        self,
        sdb,
        mdb_rw,
        pdb,
        default_branches,
        pr_facts_calculator_factory,
    ):
        meta_ids = (DEFAULT_MD_ACCOUNT_ID, 10)
        kwargs = self._kwargs(
            time_from=dt(2023, 2, 1),
            time_to=dt(2023, 3, 1),
            jira_filter=JIRAFilter.empty().replace(
                account=DEFAULT_JIRA_ACCOUNT_ID, projects=["1"],
            ),
            default_branches=default_branches,
            meta_ids=meta_ids,
            mdb=mdb_rw,
            pdb=pdb,
        )

        issue_kwargs = {"project_id": "1", "created": dt(2023, 1, 1)}
        pr_kwargs = {"created_at": dt(2023, 1, 1)}
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=98, full_name="org/r")
            repo1 = md_factory.RepositoryFactory(node_id=99, acc_id=10, full_name="org1/r")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
            await insert_repo(repo1, mdb_cleaner, mdb_rw, sdb, md_acc_id=10)
            models = [
                md_factory.JIRAProjectFactory(id="1", key="P1"),
                *jira_issue_models("1", resolved=dt(2023, 2, 15), **issue_kwargs),
                *pr_models(98, 1, 1, repository_full_name="org/r", **pr_kwargs),
                *pr_models(99, 2, 2, acc_id=10, repository_full_name="org1/r", **pr_kwargs),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=1, jira_id="1"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=2, jira_id="1", node_acc=10),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            prefixer = await Prefixer.load(meta_ids, mdb_rw, None)
            settings = Settings.from_account(1, prefixer, sdb, mdb_rw, None, None)
            release_settings = await settings.list_release_matches()

            pr_facts_calculator_no_cache = pr_facts_calculator_factory(1, meta_ids)
            await pr_facts_calculator_no_cache(
                dt(2022, 10, 1),
                dt(2024, 3, 1),
                {"org/r", "org1/r"},
                {},
                LabelFilter.empty(),
                JIRAFilter.empty(),
                False,
                {},
                release_settings,
                LogicalRepositorySettings.empty(),
                prefixer,
                False,
                0,
            )

            await wait_deferred()
            kwargs["release_settings"] = release_settings
            issues = await fetch_jira_issues(**kwargs)

        assert len(issues.pr_ids.values) == 1
        assert sorted(issues.pr_ids.values[0]) == [1, 2]

    @classmethod
    def _kwargs(cls, **extra) -> dict[str, Any]:
        return {
            "time_from": None,
            "time_to": None,
            "jira_filter": JIRAFilter.empty().replace(
                account=DEFAULT_JIRA_ACCOUNT_ID, projects=["10003", "10009"],
            ),
            "exclude_inactive": False,
            "reporters": [],
            "assignees": [],
            "commenters": [],
            "nested_assignees": False,
            "logical_settings": LogicalRepositorySettings.empty(),
            "account": DEFAULT_ACCOUNT_ID,
            "meta_ids": (DEFAULT_MD_ACCOUNT_ID,),
            "cache": None,
            **extra,
        }


class TestResolveWorkBegan:
    def test_smoke(self) -> None:
        work_began = np.array([1, 2, None, 5, None], dtype="datetime64[s]")
        prs_began = np.array([2, 1, 4, None, None], dtype="datetime64[s]")
        res = resolve_work_began(work_began, prs_began)
        expected = np.array([1, 1, None, 5, None], dtype="datetime64[s]")
        assert_array_equal(res, expected)


class TestResolveResolved:
    def test_smoke(self) -> None:
        issue_resolved = np.array([2, None, 1, 1, 1, 2], dtype="datetime64[s]")
        prs_began = np.array([1, None, None, 1, None, None], dtype="datetime64[s]")
        prs_released = np.array([1, None, None, 2, None, None], dtype="datetime64[s]")

        res = resolve_resolved(issue_resolved, prs_began, prs_released)
        expected = np.array([2, None, 1, 2, 1, 2], dtype="datetime64[s]")

        assert_array_equal(res, expected)

    def test_issue_unresolved_pr_released(self) -> None:
        issue_resolved = np.array([None], dtype="datetime64[s]")
        prs_began = np.array([1], dtype="datetime64[s]")
        prs_released = np.array([2], dtype="datetime64[s]")

        res = resolve_resolved(issue_resolved, prs_began, prs_released)
        expected = np.array([None], dtype="datetime64[s]")

        assert_array_equal(res, expected)


class TestFetchJIRAIssuesByKeys:
    async def test_base(self, mdb, pdb, default_branches, release_match_setting_tag) -> None:
        kwargs = self._kwargs(
            keys=["DEV-90", "DEV-69", "DEV-729", "DEV-1012"],
            default_branches=default_branches,
            release_settings=release_match_setting_tag,
            extra_columns=[Issue.key],
            mdb=mdb,
            pdb=pdb,
        )
        issues = await fetch_jira_issues_by_keys(**kwargs)
        issues.sort_values("key", inplace=True)  # no order is guaranteed
        assert list(issues.key) == ["DEV-1012", "DEV-69", "DEV-729", "DEV-90"]
        assert list(issues.index) == [b"12465", b"10101", b"12110", b"10308"]

    @classmethod
    def _kwargs(cls, **extra) -> dict[str, Any]:
        return {
            "jira_config": JIRAConfig(
                DEFAULT_JIRA_ACCOUNT_ID, projects={"10003": "", "10009": ""}, epics={},
            ),
            "logical_settings": LogicalRepositorySettings.empty(),
            "account": DEFAULT_ACCOUNT_ID,
            "meta_ids": (DEFAULT_MD_ACCOUNT_ID,),
            "cache": None,
            **extra,
        }


class TestFetchReleasedPRs:
    async def test_fetch_released_prs_release_settings_events(
        self,
        pr_samples,
        pdb,
        done_prs_facts_loader,
        prefixer,
    ):
        samples: Sequence[PullRequestFacts] = pr_samples(12)
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
                *pr_jira_issue_mappings((1, "1"), (1, "2"), (2, "3")),
                md_factory.JIRAIssueFactory(id="1", project_id="1", priority_name="Pr1"),
                md_factory.JIRAIssueFactory(id="2", project_id="1", priority_name="pr1"),
                md_factory.JIRAIssueFactory(id="3", project_id="1", priority_name="PR2"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            jira_filter = JIRAFilter(
                1, frozenset(("1",)), priorities=frozenset(("pr1", "pr2")), custom_projects=False,
            )
            res = await self._fetch_with_jira_filter(mdb_rw, jira_filter)
            assert [r[PullRequest.node_id.name] for r in res] == [1, 1, 2]

            jira_filter = dataclasses.replace(jira_filter, priorities={"pr1"})
            res = await self._fetch_with_jira_filter(mdb_rw, jira_filter)
            assert [r[PullRequest.node_id.name] for r in res] == [1, 1]

            jira_filter = dataclasses.replace(jira_filter, priorities={"pr2"})
            res = await self._fetch_with_jira_filter(mdb_rw, jira_filter)
            assert [r[PullRequest.node_id.name] for r in res] == [2]

    async def test_multiple_filters(self, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.PullRequestFactory(node_id=1),
                md_factory.PullRequestFactory(node_id=2),
                *pr_jira_issue_mappings((1, "1"), (2, "2")),
                md_factory.JIRAIssueFactory(id="1", project_id="P1", priority_name="PR1"),
                md_factory.JIRAIssueFactory(id="2", project_id="P2", priority_name="PR2"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            jira_filter = JIRAFilter(
                1, frozenset(("P1",)), priorities=(("pr1",)), custom_projects=False,
            )
            res = await self._fetch_with_jira_filter(mdb_rw, jira_filter)
            assert [r[PullRequest.node_id.name] for r in res] == [1]

            jira_filter = dataclasses.replace(jira_filter, priorities={"pr2"})
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
        models = [
            *pr_jira_issue_mappings((10, "20"), (11, "20"), (11, "21")),
            md_factory.JIRAIssueFactory(
                id="20", project_id="P0", key="I20", priority_id="PR", type_id="T", labels=["l0"],
            ),
            md_factory.JIRAIssueFactory(
                id="21",
                project_id="P0",
                key="I21",
                priority_id="PR",
                type_id="T",
                labels=["l0", "l1"],
            ),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
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

        assert prs[(10, "repo0")].jira == LoadedJIRADetails(
            ids=["I20"],
            projects=[b"P0"],
            priorities=[b"PR"],
            types=[b"T"],
            labels=["l0"],
        )
        assert prs[(10, "repo0")].jira == prs[(10, "repo1")].jira
        assert_array_equal(prs[(11, "repo1")].jira.ids, np.array(["I20", "I21"]))
        assert_array_equal(prs[(11, "repo1")].jira.projects, np.array([b"P0", b"P0"]))
        assert_array_equal(prs[(11, "repo1")].jira.priorities, np.array([b"PR", b"PR"]))
        assert_array_equal(prs[(11, "repo1")].jira.types, np.array([b"T", b"T"]))
        assert_array_equal(prs[(11, "repo1")].jira.labels, ["l0", "l0", "l1"])

    async def test_labels(self, mdb_rw: Database, sdb: Database) -> None:
        models = [
            *pr_jira_issue_mappings(
                (10, "20"), (11, "21"), (12, "22"), (13, "20"), (13, "22"), (14, "20"), (14, "21"),
            ),
            md_factory.JIRAIssueFactory(id="20", labels=["l0"]),
            md_factory.JIRAIssueFactory(id="21", labels=["l0", "l1"]),
            md_factory.JIRAIssueFactory(id="22", labels=[]),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            prs = {
                k: PullRequestFacts(b"")
                for k in ((10, "r"), (11, "r"), (12, "r"), (13, "r"), (14, "r"))
            }
            await PullRequestJiraMapper.load_and_apply_to_pr_facts(
                prs, JIRAEntityToFetch.EVERYTHING(), (DEFAULT_MD_ACCOUNT_ID,), mdb_rw,
            )
            assert_array_equal(prs[(10, "r")].jira.labels, np.array(["l0"]))
            assert_array_equal(prs[(11, "r")].jira.labels, np.array(["l0", "l1"]))
            assert_array_equal(prs[(12, "r")].jira.labels, np.array([], dtype="U"))
            assert_array_equal(prs[(13, "r")].jira.labels, np.array(["l0"]))
            assert_array_equal(prs[(14, "r")].jira.labels, np.array(["l0", "l0", "l1"]))

    async def test_components_as_labels(self, mdb_rw: Database, sdb: Database) -> None:
        models = [
            *pr_jira_issue_mappings((10, "20"), (10, "21"), (11, "21"), (12, "22"), (13, "23")),
            md_factory.JIRAComponentFactory(id="0", name="c0"),
            md_factory.JIRAComponentFactory(id="1", name="c1"),
            md_factory.JIRAIssueFactory(id="20", labels=["l0"], components=["1", "0"]),
            md_factory.JIRAIssueFactory(id="21", labels=["l0", "l1"]),
            md_factory.JIRAIssueFactory(id="22", labels=[], components=["1"]),
            md_factory.JIRAIssueFactory(id="23", labels=[], components=["0"]),
        ]
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            prs = {k: PullRequestFacts(b"") for k in ((10, "r"), (11, "r"), (12, "r"), (13, "r"))}
            await PullRequestJiraMapper.load_and_apply_to_pr_facts(
                prs, JIRAEntityToFetch.EVERYTHING(), (DEFAULT_MD_ACCOUNT_ID,), mdb_rw,
            )
            assert_array_equal(
                np.sort(prs[(10, "r")].jira.labels), np.array(["c0", "c1", "l0", "l0", "l1"]),
            )
            assert_array_equal(np.sort(prs[(11, "r")].jira.labels), np.array(["l0", "l1"]))
            assert_array_equal(prs[(12, "r")].jira.labels, np.array(["c1"]))
            assert_array_equal(prs[(13, "r")].jira.labels, np.array(["c0"]))

    async def test_load_only_issues(self, mdb_rw: Database, sdb: Database) -> None:
        models = [
            *pr_jira_issue_mappings((10, "20"), (10, "21"), (11, "22"), (12, "20"), (13, "20")),
            md_factory.JIRAIssueFactory(id="20", key="I20"),
            md_factory.JIRAIssueFactory(id="21", key="I21"),
            md_factory.JIRAIssueFactory(id="22", key="I22"),
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
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
                *pr_jira_issue_mappings((10, "20")),
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
            10: LoadedJIRADetails(["I1"], [b"P0"], [], [], []),
            11: LoadedJIRADetails(["I0", "I1"], [b"P0", b"P1"], [], [b"bug", b"task"], []),
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
