from datetime import datetime

import dateutil.parser
from sqlalchemy import ARRAY, BigInteger, Boolean, Column, ForeignKey, Integer, Text, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import synonym

Base = declarative_base()


# -- MIXINS --


class IDMixin:
    id = Column(BigInteger, primary_key=True)
    node_id = Column(Text, nullable=False)


class IDMixinNG:
    id = Column(Text, primary_key=True)
    discovered_at = Column(TIMESTAMP, default=datetime.utcnow)
    fetched_at = Column(TIMESTAMP, default=datetime.utcnow)

    @declared_attr
    def node_id(self):
        """Return 'id' as 'node_id'."""
        return synonym("id")


class BodyMixin:
    body = Column(Text)


class UpdatedMixin:
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)


class UserMixin:
    user_id = Column(BigInteger)
    user_login = Column(Text)


class RepositoryMixin:
    repository_full_name = Column(Text, nullable=False)


class ParentChildMixin:
    parent_id = Column(Text, primary_key=True)
    child_id = Column(Text, primary_key=True)

# -- TABLES --


class Installation(Base, UpdatedMixin):
    __tablename__ = "github_installations"

    id = Column(BigInteger, primary_key=True)
    delivery_id = Column(Text, nullable=False)
    app_id = Column(BigInteger, nullable=False)
    target_id = Column(BigInteger, nullable=False)
    target_type = Column(Text, nullable=False)
    html_url = Column(Text)


class InstallationOwner(Base, UpdatedMixin):
    __tablename__ = "github_installation_owners"

    install_id = Column(BigInteger,
                        ForeignKey("github_installations.id", name="fk_github_installation_owner"),
                        primary_key=True)
    user_id = Column(BigInteger, primary_key=True)
    user_login = Column(Text, nullable=False)


class InstallationRepo(Base):
    __tablename__ = "github_installation_repos_compat"

    install_id = Column(BigInteger,
                        ForeignKey("github_installations.id", name="fk_github_installation_repo"),
                        primary_key=True)
    repo_id = Column(BigInteger, primary_key=True)
    repo_full_name = Column(Text, nullable=False)
    updated_at = Column(TIMESTAMP, nullable=False)


class FetchProgress(Base, UpdatedMixin):
    __tablename__ = "github_fetch_progress"

    event_id = Column(Text, primary_key=True)
    node_type = Column(Text, primary_key=True)
    nodes_processed = Column(BigInteger, default=0)
    nodes_total = Column(BigInteger, nullable=False)


class PullRequestComment(Base,
                         BodyMixin,
                         IDMixin,
                         UpdatedMixin,
                         UserMixin,
                         RepositoryMixin,
                         ):
    __tablename__ = "github_pull_request_comments_compat"

    author_association = Column(Text)
    html_url = Column(Text)
    pull_request_node_id = Column(Text, nullable=False)


class PullRequestReviewComment(Base,
                               BodyMixin,
                               IDMixin,
                               UpdatedMixin,
                               UserMixin,
                               RepositoryMixin,
                               ):
    __tablename__ = "github_pull_request_review_comments_compat"

    author_association = Column(Text)
    commit_id = Column(Text)
    diff_hunk = Column(Text)
    htmlurl = Column(Text)
    in_reply_to = Column(BigInteger)
    original_commit_id = Column(Text)
    original_position = Column(BigInteger)
    path = Column(Text)
    position = Column(BigInteger)
    pull_request_node_id = Column(Text, nullable=False)
    pull_request_review_id = Column(BigInteger)


class PullRequestCommit(Base, RepositoryMixin):
    __tablename__ = "github_pull_request_commits_compat"

    node_id = Column(Text, nullable=False)
    commit_node_id = Column(Text, nullable=False)
    author_login = Column(Text)
    author_email = Column(Text)
    author_name = Column(Text)
    author_date = Column(Text, nullable=False)
    authored_date = Column(TIMESTAMP, nullable=False)
    committer_login = Column(Text)
    committer_email = Column(Text)
    committer_name = Column(Text)
    commit_date = Column(Text, nullable=False)
    committed_date = Column(TIMESTAMP, nullable=False)
    pull_request_node_id = Column(Text, primary_key=True)
    sha = Column(Text, primary_key=True)
    additions = Column(Integer, nullable=False)
    deletions = Column(Integer, nullable=False)
    message = Column(Text, nullable=False)
    created_at = synonym("committed_date")

    def parse_author_date(self):
        """Deserialize the date when the commit was authored.

        We have to store the timestamp in the original, string representation because PostgreSQL
        always discards the time zone. The time zone is important here because we are interested
        in the *local* time.
        """
        return dateutil.parser.parse(self.author_date)

    def parse_commit_date(self):
        """Deserialize the date when the commit was pushed.

        We have to store the timestamp in the original, string representation because PostgreSQL
        always discards the time zone. The time zone is important here because we are interested
        in the *local* time.
        """
        return dateutil.parser.parse(self.commit_date)


PullRequestCommit.created_at.key = "committed_date"


class PullRequestReview(Base,
                        BodyMixin,
                        IDMixin,
                        UserMixin,
                        ):
    __tablename__ = "github_pull_request_reviews_compat"

    commit_id = Column(Text, nullable=False)
    htmlurl = Column(Text)
    pull_request_node_id = Column(Text, nullable=False)
    state = Column(Text, nullable=False)
    submitted_at = Column(TIMESTAMP, nullable=False)
    repository_full_name = Column(Text)
    created_at = synonym("submitted_at")


PullRequestReview.created_at.key = "submitted_at"


class PullRequestReviewRequest(Base,
                               IDMixinNG,
                               ):
    __tablename__ = "github_node_review_requested_event"

    actor = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP)
    pull_request = Column(Text, nullable=False)
    pull_request_node_id = synonym("pull_request")
    # FIXME(vmarkovtsev): set nullable=False when ENG-303 is resolved
    requested_reviewer = Column(Text, nullable=True)


PullRequestReviewRequest.pull_request_node_id.key = "pull_request"
PullRequestReviewRequest.node_id.key = "id"


class PullRequest(Base,
                  BodyMixin,
                  IDMixin,
                  RepositoryMixin,
                  UpdatedMixin,
                  UserMixin,
                  ):
    __tablename__ = "github_pull_requests_compat"

    additions = Column(BigInteger)
    assignees = Column(ARRAY(BigInteger))
    author_association = Column(Text)
    base_ref = Column(Text, nullable=False)
    base_repository_full_name = Column(Text, nullable=False)
    base_sha = Column(Text, nullable=False)
    base_user = Column(Text, nullable=False)
    changed_files = Column(BigInteger)
    closed_at = Column(TIMESTAMP)
    deletions = Column(BigInteger)
    head_ref = Column(Text, nullable=False)
    # These are nullable because the head repository can be deleted by the owner.
    head_repository_full_name = Column(Text)
    head_user = Column(Text)
    # head_sha is always not null.
    head_sha = Column(Text, nullable=False)
    htmlurl = Column(Text)
    labels = Column(ARRAY(Text()))
    maintainer_can_modify = Column(Boolean)
    merge_commit_sha = Column(Text)
    mergeable = Column(Text)
    merged = Column(Boolean)
    merged_at = Column(TIMESTAMP)
    merged_by_id = Column(BigInteger)
    merged_by_login = Column(Text)
    milestone_id = Column(Text, nullable=False)
    milestone_title = Column(Text, nullable=False)
    number = Column(BigInteger, nullable=False)
    review_comments = Column(BigInteger)
    state = Column(Text)
    title = Column(Text)


class PushCommit(Base,
                 RepositoryMixin):
    __tablename__ = "github_push_commits_compat"

    node_id = Column(Text, primary_key=True)
    message = Column(Text, nullable=False)
    pushed_date = Column(TIMESTAMP)
    author_login = Column(Text)
    author_avatar_url = Column(Text)
    author_email = Column(Text)
    author_name = Column(Text)
    author_date = Column(Text, nullable=False)
    authored_date = Column(TIMESTAMP, nullable=False)
    url = Column(Text)
    sha = Column(Text, nullable=False)
    committer_login = Column(Text)
    committer_avatar_url = Column(Text)
    committer_email = Column(Text)
    committer_name = Column(Text)
    commit_date = Column(Text, nullable=False)
    committed_date = Column(TIMESTAMP, nullable=False)
    additions = Column(BigInteger, nullable=False)
    deletions = Column(BigInteger, nullable=False)
    changed_files = Column(BigInteger, nullable=False)


class Repository(Base,
                 IDMixin,
                 UpdatedMixin,
                 ):
    __tablename__ = "github_repositories_v2_compat"

    archived = Column(Boolean)
    default_branch = Column(Text)
    description = Column(Text)
    disabled = Column(Boolean)
    fork = Column(Boolean)
    full_name = Column(Text, nullable=False)
    html_url = Column(Text)
    language = Column(Text)
    name = Column(Text)
    owner = Column(BigInteger, nullable=False)
    private = Column(Boolean)
    pushed_at = Column(TIMESTAMP)
    ssh_url = Column(Text)


class User(Base,
           IDMixin,
           UpdatedMixin,
           ):
    __tablename__ = "github_users_v2_compat"

    avatar_url = Column(Text, nullable=False)
    company = Column(Text)
    email = Column(Text)
    url = Column(Text)
    blog = Column(Text)
    location = Column(Text)
    login = Column(Text, nullable=False)
    name = Column(Text)


class Release(Base, RepositoryMixin):
    __tablename__ = "github_releases_compat"

    id = Column(Text, primary_key=True)
    author = Column(Text, nullable=False)
    author_avatar_url = Column(Text)
    description_html = Column(Text)
    name = Column(Text)
    published_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
    tag = Column(Text)
    url = Column(Text)
    sha = Column(Text, nullable=False)
    commit_id = Column(Text, nullable=False)


class NodeCommit(Base):
    __tablename__ = "github_node_commit"

    id = Column(Text, primary_key=True)
    oid = Column(Text, nullable=False)
    sha = synonym(oid)
    committed_date = Column(TIMESTAMP)


NodeCommit.sha.key = "oid"


class NodeCommitParent(Base, ParentChildMixin):
    __tablename__ = "github_node_commit_parents"

    index = Column(Integer, nullable=False)


class NodePullRequestCommit(Base):
    __tablename__ = "github_node_pull_request_commit"

    id = Column(Text, primary_key=True)
    commit = Column(Text, nullable=False)
    pull_request = Column(Text, nullable=False)


class Branch(Base, RepositoryMixin):
    __tablename__ = "github_branches_compat"

    repo_id = Column(Text, primary_key=True)
    branch_id = Column(Text, primary_key=True)
    branch_name = Column(Text, nullable=False)
    is_default = Column(Boolean, nullable=False)
    commit_id = Column(Text, nullable=False)
    commit_sha = Column(Text, nullable=False)
