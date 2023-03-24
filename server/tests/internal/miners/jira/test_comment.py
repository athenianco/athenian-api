from athenian.api.db import Database
from athenian.api.internal.miners.jira.comment import fetch_issues_comments
from athenian.api.models.metadata.jira import Comment
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory.common import DEFAULT_JIRA_ACCOUNT_ID
from tests.testutils.factory.metadata import JIRACommentFactory
from tests.testutils.factory.wizards import jira_issue_models


class TestFetchIssuesComments:
    async def test_base(self, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                *jira_issue_models("1", key="P1-1"),
                *jira_issue_models("2"),
                *jira_issue_models("3"),
                *jira_issue_models("4"),
                JIRACommentFactory(issue_id="1", author_id="1"),
                JIRACommentFactory(issue_id="1", author_id="2", body="Hello, World!"),
                JIRACommentFactory(issue_id="2", author_id="3", body="RB"),
                JIRACommentFactory(issue_id="4"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            comments = await fetch_issues_comments(
                ["1", "2", "3"],
                DEFAULT_JIRA_ACCOUNT_ID,
                mdb_rw,
                [Comment.body, Comment.author_id],
            )
            comments = comments.sort_values(("issue_id", "author_id"))

            assert list(comments[Comment.issue_id.name]) == [b"1", b"1", b"2"]
            assert list(comments[Comment.author_id.name]) == ["1", "2", "3"]
            assert comments[Comment.body.name][1] == "Hello, World!"
            assert comments[Comment.body.name][2] == "RB"

    async def test_deleted_comments_are_ignored(self, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                *jira_issue_models("1"),
                JIRACommentFactory(issue_id="1", body="C0", is_deleted=False),
                JIRACommentFactory(issue_id="1", body="C1", is_deleted=True),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            comments = await fetch_issues_comments(
                ["1"], DEFAULT_JIRA_ACCOUNT_ID, mdb_rw, [Comment.body],
            )

            assert list(comments[Comment.issue_id.name]) == [b"1"]
            assert list(comments[Comment.body.name]) == ["C0"]
