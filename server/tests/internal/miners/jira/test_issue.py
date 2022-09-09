import numpy as np
from numpy.testing import assert_array_equal

from athenian.api.db import Database
from athenian.api.internal.miners.jira.issue import PullRequestJiraMapper
from athenian.api.internal.miners.types import (
    JIRAEntityToFetch,
    PullRequestFacts,
    PullRequestJIRADetails,
)
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_MD_ACCOUNT_ID


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
