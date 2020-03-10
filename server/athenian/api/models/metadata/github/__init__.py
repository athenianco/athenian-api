from datetime import datetime

import dateutil.parser
from sqlalchemy import ARRAY, BigInteger, Boolean, Column, ForeignKey, Integer, Text, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
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


class BodyMixin:
    body = Column(Text)


class UpdatedMixin:
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)


class UserMixin:
    user_id = Column(BigInteger, nullable=False)
    user_login = Column(Text, nullable=False)


class RepositoryMixin:
    repository_name = Column(Text, nullable=False)
    repository_owner = Column(Text, nullable=False)
    repository_fullname = Column(Text, nullable=False)


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


class IssueComment(Base,
                   BodyMixin,
                   IDMixin,
                   UpdatedMixin,
                   UserMixin,
                   ):
    __tablename__ = "github_issue_comments_compat"

    author_association = Column(Text)
    html_url = Column(Text)
    issue_node_id = Column(Text, nullable=False)
    pull_request_node_id = synonym("issue_node_id")


IssueComment.pull_request_node_id.key = "issue_node_id"


class PullRequestComment(Base,
                         BodyMixin,
                         IDMixin,
                         UpdatedMixin,
                         UserMixin,
                         ):
    __tablename__ = "github_pull_request_comments_compat"

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


class PullRequestCommit(Base):
    __tablename__ = "github_pull_request_commits_compat"

    node_id = Column(Text, nullable=False)
    author_login = Column(Text)
    author_email = Column(Text)
    author_name = Column(Text)
    author_date = Column(Text)
    committer_login = Column(Text)
    committer_email = Column(Text)
    committer_name = Column(Text)
    commit_date = Column(Text)
    pull_request_node_id = Column(Text, primary_key=True)
    sha = Column(Text, primary_key=True)
    additions = Column(Integer)
    deletions = Column(Integer)
    message = Column(Text)
    created_at = synonym("commit_date")

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


PullRequestCommit.created_at.key = "commit_date"


class PullRequestReview(Base,
                        BodyMixin,
                        IDMixin,
                        UserMixin,
                        ):
    __tablename__ = "github_pull_request_reviews_compat"

    commit_id = Column(Text)
    htmlurl = Column(Text)
    pull_request_node_id = Column(Text, nullable=False)
    state = Column(Text)
    submitted_at = Column(TIMESTAMP)
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
    base_repository_name = Column(Text, nullable=False)
    base_repository_owner = Column(Text, nullable=False)
    base_repository_fullname = Column(Text, nullable=False)
    base_sha = Column(Text, nullable=False)
    base_user = Column(Text, nullable=False)
    changed_files = Column(BigInteger)
    closed_at = Column(TIMESTAMP)
    comments = Column(BigInteger)
    commits = Column(BigInteger)
    deletions = Column(BigInteger)
    head_ref = Column(Text, nullable=False)
    # These are nullable because the head repository can be deleted by the owner.
    head_repository_name = Column(Text)
    head_repository_owner = Column(Text)
    head_repository_fullname = Column(Text)
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
    merged_by_id = Column(BigInteger, nullable=False)
    merged_by_login = Column(Text, nullable=False)
    milestone_id = Column(Text, nullable=False)
    milestone_title = Column(Text, nullable=False)
    number = Column(BigInteger)
    review_comments = Column(BigInteger)
    state = Column(Text)
    title = Column(Text)


class PushCommit(Base,
                 RepositoryMixin):
    __tablename__ = "github_push_commits_compat"

    timestamp = Column(TIMESTAMP)
    id = Column(BigInteger)
    message = Column(Text)
    author_login = Column(Text)
    url = Column(Text)
    sha = Column(Text, primary_key=True)
    committer_login = Column(Text)
    added = Column(BigInteger)
    removed = Column(BigInteger)
    modified = Column(BigInteger)
    pusher_login = Column(Text)


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
    full_name = Column(Text)
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

    avatar_url = Column(Text)
    company = Column(Text)
    email = Column(Text)
    url = Column(Text)
    blog = Column(Text)
    location = Column(Text)
    login = Column(Text)
    name = Column(Text)
    login = Column(Text)


class Release(Base,
              IDMixinNG,
              UpdatedMixin,
              ):
    __tablename__ = "github_node_release"

    author = Column(Text)
    description = Column(Text)
    description_html = Column(Text)
    is_draft = Column(Boolean)
    is_prerelease = Column(Boolean)
    name = Column(Text)
    published_at = Column(TIMESTAMP)
    resource_path = Column(Text)
    tag = Column(Text)
    tag_name = Column(Text)
    url = Column(Text)
