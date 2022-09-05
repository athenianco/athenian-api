import dataclasses

import pandas as pd

from athenian.api.db import Database
from athenian.api.internal.miners.jira.issue import PRJIRAMapping, PullRequestJiraMapper
from athenian.api.internal.miners.types import PullRequestFacts
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
            await PullRequestJiraMapper.load_and_apply_to_pr_facts(
                prs, (DEFAULT_MD_ACCOUNT_ID,), mdb_rw,
            )

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

    async def test_load_duplicates(self, mdb_rw: Database) -> None:
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

    def test_load_from_df(self) -> None:
        df = pd.DataFrame(
            {
                "pr_id": [1, 1, 2, 3],
                "issue_key": ["I1", "I2", "I2", "I3"],
                "project_id": ["P0", "P1", "P0", "P1"],
                "priority_name": ["Low", "High", None, "Low"],
                "type": ["bug", "bug", "task", "bug"],
            },
        )
        df.set_index(["pr_id", "issue_key"], inplace=True)

        mapping = PullRequestJiraMapper.load_from_df(df)
        assert sorted(mapping) == [1, 2, 3]
        assert sorted(mapping[1].ids) == ["I1", "I2"]
        assert sorted(mapping[1].projects) == ["P0", "P1"]
        assert sorted(mapping[1].priorities) == ["High", "Low"]
        assert mapping[1].types == ["bug"]

        assert mapping[2].ids == ["I2"]
        assert mapping[2].projects == ["P0"]
        assert mapping[2].priorities == []
        assert mapping[2].types == ["task"]

        assert mapping[3].ids == ["I3"]
        assert mapping[3].projects == ["P1"]
        assert mapping[3].priorities == ["Low"]
        assert mapping[3].types == ["bug"]

    def test_apply_to_pr_facts(self) -> None:
        facts = {
            (10, "repo0"): PullRequestFacts(b""),
            (10, "repo1"): PullRequestFacts(b""),
            (11, "repo1"): PullRequestFacts(b""),
        }
        mapping = {
            10: PRJIRAMapping(["I1"], ["P0"], [], []),
            11: PRJIRAMapping(["I0", "I1"], ["P0", "P1"], [], ["bug", "task"]),
        }

        PullRequestJiraMapper.apply_to_pr_facts(facts, mapping)
        assert sorted(facts) == [(10, "repo0"), (10, "repo1"), (11, "repo1")]
        assert facts[(10, "repo0")].jira_ids == ["I1"]
        assert facts[(10, "repo0")].jira_projects == ["P0"]
        assert facts[(10, "repo0")].jira_priorities == []
        assert facts[(10, "repo0")].jira_types == []
        assert facts[(10, "repo1")] == facts[(10, "repo1")]

        assert sorted(facts[(11, "repo1")].jira_ids) == ["I0", "I1"]
        assert sorted(facts[(11, "repo1")].jira_projects) == ["P0", "P1"]
        assert facts[(11, "repo1")].jira_priorities == []
        assert sorted(facts[(11, "repo1")].jira_types) == ["bug", "task"]

    def test_apply_empty_to_pr_facts(self) -> None:
        facts = {(10, "r0"): PullRequestFacts(b"")}
        PullRequestJiraMapper.apply_empty_to_pr_facts(facts)
        assert list(facts) == [(10, "r0")]
        assert facts[(10, "r0")].jira_ids == []
        assert facts[(10, "r0")].jira_projects == []
        assert facts[(10, "r0")].jira_priorities == []
        assert facts[(10, "r0")].jira_types == []
