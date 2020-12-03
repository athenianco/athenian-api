from datetime import datetime, timezone

from sqlalchemy import Column, func, Integer, JSON, LargeBinary, String, Text, TIMESTAMP
from sqlalchemy.dialects import postgresql, sqlite

from athenian.api.models import create_base

Base = create_base()


TSARRAY = postgresql.ARRAY(TIMESTAMP(timezone=True)).with_variant(JSON(), sqlite.dialect.name)
HSTORE = postgresql.HSTORE().with_variant(JSON(), sqlite.dialect.name)


class UpdatedMixin:
    """Declare "updated_at" column."""

    updated_at = Column(TIMESTAMP(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc),
                        server_default=func.now(),
                        onupdate=lambda ctx: datetime.now(timezone.utc))


class GitHubDonePullRequestFacts(Base, UpdatedMixin):
    """
    Mined PullRequestFacts about released/rejected PRs.

    Tricky columns:
        * `release_match`: the description of the release match strategy applied to this PR. \
                           Note that `pr_done_at` depends on that.
        * `pr_done_at`: PR closure timestamp if it is not merged or PR release timestamp if it is.
        * `HSTORE` a set of developers with which we can efficiently check an intersection.
        * `data`: pickle-d PullRequestFacts (may change in the future).
        * `format_version`: version of the table, used for smooth upgrades and downgrades.
    """

    __tablename__ = "github_done_pull_request_facts"

    pr_node_id = Column(String(), primary_key=True)
    release_match = Column(Text(), primary_key=True)
    format_version = Column(Integer(), primary_key=True, default=9, server_default="9")
    repository_full_name = Column(String(), nullable=False)
    pr_created_at = Column(TIMESTAMP(timezone=True), nullable=False)
    pr_done_at = Column(TIMESTAMP(timezone=True))
    number = Column(Integer(), nullable=False)
    author = Column(String())  # can be null, see @ghost
    merger = Column(String())
    releaser = Column(String())
    release_url = Column(Text())
    release_node_id = Column(Text())
    reviewers = Column(HSTORE, nullable=False, server_default="")
    commenters = Column(HSTORE, nullable=False, server_default="")
    commit_authors = Column(HSTORE, nullable=False, server_default="")
    commit_committers = Column(HSTORE, nullable=False, server_default="")
    labels = Column(HSTORE, nullable=False, server_default="")
    activity_days = Column(TSARRAY, nullable=False, server_default="{}")
    data = Column(LargeBinary(), nullable=False)


class GitHubOpenPullRequestFacts(Base, UpdatedMixin):
    """Mined PullRequestFacts about open PRs."""

    __tablename__ = "github_open_pull_request_facts"

    pr_node_id = Column(String(), primary_key=True)
    format_version = Column(Integer(), primary_key=True, default=6, server_default="6")
    repository_full_name = Column(String(), nullable=False)
    pr_created_at = Column(TIMESTAMP(timezone=True), nullable=False)
    number = Column(Integer(), nullable=False)
    pr_updated_at = Column(TIMESTAMP(timezone=True), nullable=False)
    activity_days = Column(TSARRAY, nullable=False, server_default="{}")
    data = Column(LargeBinary(), nullable=False)


class GitHubMergedPullRequestFacts(Base, UpdatedMixin):
    """
    Merged PRs (released and unreleased).

    We attach the mined releases that do *not* contain the given pull request.

    `pr_node_id` is a merged but not released yet PR node identifier.
    According to `release_match`, any release between `merged_at` and `checked_until` does not
    contain that PR.
    """

    __tablename__ = "github_merged_pull_request_facts"

    pr_node_id = Column(String(), primary_key=True)
    release_match = Column(Text(), primary_key=True)
    format_version = Column(Integer(), primary_key=True, default=6, server_default="6")
    merged_at = Column(TIMESTAMP(timezone=True), nullable=False)
    repository_full_name = Column(String(), nullable=False)
    checked_until = Column(TIMESTAMP(timezone=True), nullable=False)
    author = Column(String())  # can be null, see @ghost
    merger = Column(String())  # @ghost can merge, too
    labels = Column(HSTORE, nullable=False, server_default="")
    activity_days = Column(TSARRAY, nullable=False, server_default="{}")
    data = Column(LargeBinary())  # can be null, we run a 2-step update procedure


class GitHubCommitHistory(Base, UpdatedMixin):
    """
    Mined Git commit graph with all the releases.

    We save one graph per repository. There are (rare) cases when the same commit has different
    parents in different branches, but we ignore them.
    Format: 3 numpy arrays: hashes, vertexes (indexes for their edge offsets), \
    edges (children vertex indexes).
    Direction: HEAD -> ROOT. In other words, this is the *reverse* Git commit relationship.
    """

    __tablename__ = "github_commit_history"

    repository_full_name = Column(String(), primary_key=True)
    format_version = Column(Integer(), primary_key=True, default=4, server_default="4")
    dag = Column(LargeBinary(), nullable=False)


class GitHubRepository(Base, UpdatedMixin):
    """Mined facts about repositories."""

    __tablename__ = "github_repositories"

    node_id = Column(String(), primary_key=True)
    repository_full_name = Column(String(), nullable=False)
    first_commit = Column(TIMESTAMP(timezone=True), nullable=False)


class GitHubRelease(Base):
    """Mined repository releases."""

    __tablename__ = "github_releases"

    id = Column(Text, primary_key=True)
    release_match = Column(Text(), primary_key=True)
    repository_full_name = Column(String(), nullable=False)
    repository_node_id = Column(Text, nullable=False)
    author = Column(Text)
    name = Column(Text, nullable=False)
    published_at = Column(TIMESTAMP(timezone=True), nullable=False)
    tag = Column(Text)
    url = Column(Text, nullable=False)
    sha = Column(Text, nullable=False)
    commit_id = Column(Text, nullable=False)


class GitHubReleaseFacts(Base):
    """Mined facts about repository releases."""

    __tablename__ = "github_release_facts"

    id = Column(Text, primary_key=True)
    release_match = Column(Text(), primary_key=True)
    format_version = Column(Integer(), primary_key=True, default=5, server_default="5")
    repository_full_name = Column(String(), nullable=False)
    published_at = Column(TIMESTAMP(timezone=True), nullable=False)
    data = Column(LargeBinary(), nullable=False)


class GitHubReleaseMatchTimespan(Base):
    """For which dates we matched releases."""

    __tablename__ = "github_release_match_spans"

    repository_full_name = Column(String(), primary_key=True)
    release_match = Column(Text(), primary_key=True)
    time_from = Column(TIMESTAMP(timezone=True), nullable=False)
    time_to = Column(TIMESTAMP(timezone=True), nullable=False)
