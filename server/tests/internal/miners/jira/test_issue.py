import dataclasses

from athenian.api.db import Database
from athenian.api.internal.miners.jira.issue import PRJIRAMapping, PullRequestJiraMapper
from athenian.api.internal.miners.types import PullRequestFacts
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_MD_ACCOUNT_ID


class TestPullRequestJiraMapper:
    async def test_append(self, mdb_rw: Database, sdb: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestJiraIssuesFactory(node_id=10, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=11, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=11, jira_id="21"),
                md_factory.JIRAIssueFactory(
                    id="20", project_id="P0", key="I20", priority_name="PR", type="T",
                ),
                md_factory.JIRAIssueFactory(
                    id="21", project_id="P0", key="I21", priority_name="PR", type="T",
                ),
            ]

            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            prs = {
                (10, "repo0"): PullRequestFacts(b""),
                (10, "repo1"): PullRequestFacts(b""),
                (11, "repo1"): PullRequestFacts(b""),
            }
            await PullRequestJiraMapper.append(prs, (DEFAULT_MD_ACCOUNT_ID,), mdb_rw)

        assert sorted(prs) == [(10, "repo0"), (10, "repo1"), (11, "repo1")]

        assert prs[(10, "repo0")].jira_ids == ["I20"]
        assert prs[(10, "repo0")].jira_projects == ["P0"]
        assert prs[(10, "repo0")].jira_priorities == ["PR"]
        assert prs[(10, "repo0")].jira_types == ["T"]

        assert prs[(10, "repo1")].jira_ids == ["I20"]
        assert prs[(10, "repo1")].jira_projects == ["P0"]
        assert prs[(10, "repo1")].jira_priorities == ["PR"]
        assert prs[(10, "repo1")].jira_types == ["T"]

        assert sorted(prs[(11, "repo1")].jira_ids) == ["I20", "I21"]
        assert prs[(11, "repo1")].jira_projects == ["P0"]
        assert prs[(11, "repo1")].jira_priorities == ["PR"]
        assert prs[(11, "repo1")].jira_types == ["T"]

    async def test_load(self, mdb_rw: Database, sdb: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestJiraIssuesFactory(node_id=10, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=11, jira_id="21"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=11, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=12, jira_id="22"),
                md_factory.JIRAIssueFactory(
                    id="20", key="I20", project_id="P0", type="Task", priority_name="High",
                ),
                md_factory.JIRAIssueFactory(
                    id="21", key="I21", project_id="P1", type="Bug", priority_name="Low",
                ),
                md_factory.JIRAIssueFactory(
                    id="22", key="I22", project_id="P2", type="Bug", priority_name=None,
                ),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            mapping = await PullRequestJiraMapper.load(
                [10, 11, 12], (DEFAULT_MD_ACCOUNT_ID,), mdb_rw,
            )

        assert mapping[10] == PRJIRAMapping(["I20"], ["P0"], ["High"], ["Task"])

        mapping_11 = PRJIRAMapping(
            **{f: sorted(v) for f, v in dataclasses.asdict(mapping[11]).items()},
        )
        assert mapping_11 == PRJIRAMapping(
            ["I20", "I21"], ["P0", "P1"], ["High", "Low"], ["Bug", "Task"],
        )
        assert mapping[12] == PRJIRAMapping(["I22"], ["P2"], [], ["Bug"])

    async def test_load_duplicates(self, mdb_rw: Database, sdb: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestJiraIssuesFactory(node_id=10, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=10, jira_id="21"),
                md_factory.JIRAIssueFactory(id="20", project_id="30", type="Task"),
                md_factory.JIRAIssueFactory(id="21", project_id="31", type="Task"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            mapping = await PullRequestJiraMapper.load([10], (DEFAULT_MD_ACCOUNT_ID,), mdb_rw)

        assert mapping[10].types == ["Task"]

    async def test_load_ids(self, mdb_rw: Database, sdb: Database) -> None:
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

            mapping = await PullRequestJiraMapper.load_ids(
                [10, 11, 12, 14], (DEFAULT_MD_ACCOUNT_ID,), mdb_rw,
            )

            assert sorted(mapping) == [10, 11, 12]
            assert sorted(mapping[10]) == ["I20", "I21"]
            assert mapping[11] == ["I22"]
            assert mapping[12] == ["I20"]
