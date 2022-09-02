from athenian.api.db import Database
from athenian.api.internal.miners.jira.issue import PullRequestJiraMapper
from athenian.api.internal.miners.types import PullRequestFacts
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_MD_ACCOUNT_ID


class TestPullRequestJiraMapper:
    async def test_append_pr_jira_mapping(self, mdb_rw: Database, sdb: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestJiraIssuesFactory(node_id=10, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=10, jira_id="21"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=11, jira_id="20"),
                md_factory.JIRAIssueFactory(id="20", key="I20"),
                md_factory.JIRAIssueFactory(id="21", key="I21"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            prs = {
                (10, "repo0"): PullRequestFacts(b""),
                (11, "repo0"): PullRequestFacts(b""),
            }
            await PullRequestJiraMapper.append_pr_jira_mapping(
                prs, (DEFAULT_MD_ACCOUNT_ID,), mdb_rw,
            )

            assert sorted(prs[(10, "repo0")].jira_ids) == ["I20", "I21"]
            assert prs[(11, "repo0")].jira_ids == ["I20"]

    async def test_load_pr_jira_mapping(self, mdb_rw: Database, sdb: Database) -> None:
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

            mapping = await PullRequestJiraMapper.load_pr_jira_mapping(
                [10, 11, 12, 14], (DEFAULT_MD_ACCOUNT_ID,), mdb_rw,
            )

            assert sorted(mapping) == [10, 11, 12]
            assert sorted(mapping[10]) == ["I20", "I21"]
            assert mapping[11] == ["I22"]
            assert mapping[12] == ["I20"]
