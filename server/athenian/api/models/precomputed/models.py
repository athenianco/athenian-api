from datetime import datetime, timezone

from sqlalchemy import CHAR, Column, func, Integer, JSON, LargeBinary, String, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import ARRAY, HSTORE

from athenian.api.models import create_base


Base = create_base()


TSARRAY = ARRAY(TIMESTAMP(timezone=True)).with_variant(JSON(), "sqlite")
JHSTORE = HSTORE().with_variant(JSON(), "sqlite")
RepositoryFullName = String(64 + 1 + 100)  # user / project taken from the official GitHub docs


class UpdatedMixin:
    """Declare "updated_at" column."""

    updated_at = Column(TIMESTAMP(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc),
                        server_default=func.now(),
                        onupdate=lambda ctx: datetime.now(timezone.utc))


class GitHubPullRequestTimes(Base, UpdatedMixin):
    """
    Mined PullRequestTimes.

    Tricky columns:
        * `release_match`: the description of the release match strategy applied to this PR. \
                           Note that `pr_done_at` depends on that.
        * `pr_done_at`: PR closure timestamp if it is not merged or PR release timestamp if it is.
        * `HSTORE` a set of developers with which we can efficiently check an intersection.
        * `data`: pickle-d PullRequestTimes (may change in the future).
        * `format_version`: version of the table, used for smooth upgrades and downgrades.
    """

    __tablename__ = "github_pull_request_times"

    pr_node_id = Column(CHAR(32), primary_key=True)
    release_match = Column(Text(), primary_key=True)
    format_version = Column(Integer(), primary_key=True, default=3, server_default="3")
    repository_full_name = Column(RepositoryFullName, nullable=False)
    pr_created_at = Column(TIMESTAMP(timezone=True), nullable=False)
    pr_done_at = Column(TIMESTAMP(timezone=True))
    author = Column(CHAR(100))  # can be null, see @ghost
    merger = Column(CHAR(100))
    releaser = Column(CHAR(100))
    release_url = Column(Text())
    reviewers = Column(JHSTORE, nullable=False, server_default="")
    commenters = Column(JHSTORE, nullable=False, server_default="")
    commit_authors = Column(JHSTORE, nullable=False, server_default="")
    commit_committers = Column(JHSTORE, nullable=False, server_default="")
    activity_days = Column(TSARRAY, nullable=False, server_default="{}")
    data = Column(LargeBinary(), nullable=False)


class GitHubCommitHistory(Base, UpdatedMixin):
    """
    Mined Git commit graph.

    We save one graph per repository. There are (rare) cases when the same commit has different
    parents in different branches, but we ignore them.
    Format: 40 char -> [40 char] * n mapping, marshal-ed.
    Direction: HEAD -> ROOT. In other words, this is the *reverse* Git commit relationship.
    """

    __tablename__ = "github_commit_history"

    repository_full_name = Column(RepositoryFullName, primary_key=True)
    format_version = Column(Integer(), primary_key=True, default=1, server_default="1")
    dag = Column(LargeBinary(), nullable=False)


class GitHubCommitFirstParents(Base, UpdatedMixin):
    """
    Mined Git commit first parents - commits that follow the main branch.

    We save first parent commit node identifiers and their commit timestamps per repo.
    They are independently pickled and their byte streams are concatenated to `commits`.
    """

    __tablename__ = "github_commit_first_parents"

    repository_full_name = Column(RepositoryFullName, primary_key=True)
    format_version = Column(Integer(), primary_key=True, default=1, server_default="1")
    commits = Column(LargeBinary(), nullable=False)


class GitHubMergedPullRequest(Base, UpdatedMixin):
    """
    Mined releases that do *not* contain the given pull request.

    `pr_node_id` is a merged but not released yet PR node identifier.
    According to `release_match`, any release with a node id in `checked_releases` does not
    contain that PR.
    """

    __tablename__ = "github_merged_pull_requests"

    pr_node_id = Column(CHAR(32), primary_key=True)
    release_match = Column(Text(), primary_key=True)
    merged_at = Column(TIMESTAMP(timezone=True), nullable=False)
    repository_full_name = Column(RepositoryFullName, nullable=False)
    checked_releases = Column(JHSTORE, nullable=False, server_default="")
