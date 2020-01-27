from sqlalchemy import ARRAY, BigInteger, Boolean, Column, ForeignKey, Integer, Text, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import synonym

Base = declarative_base()


# -- MIXINS --


class IDMixin:
    id = Column(BigInteger, primary_key=True)
    node_id = Column(Text)


class DeliveryMixin:
    delivery_id = Column(Text, nullable=False)
    action = Column(Text)
    timestamp = Column(TIMESTAMP)


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


class InstallationRepo(Base):
    __tablename__ = "github_installation_repos"

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
                   DeliveryMixin,
                   IDMixin,
                   RepositoryMixin,
                   UpdatedMixin,
                   UserMixin,
                   ):
    __tablename__ = "github_issue_comments"

    author_association = Column(Text)
    htmlurl = Column(Text)
    issue_number = Column(BigInteger, nullable=False)
    pull_request_number = synonym("issue_number")


IssueComment.pull_request_number.key = "issue_number"


class Issue(Base,
            BodyMixin,
            DeliveryMixin,
            IDMixin,
            RepositoryMixin,
            UpdatedMixin,
            UserMixin,
            ):
    __tablename__ = "github_issues"

    assignees = Column(ARRAY(Text()), nullable=False)
    closed_at = Column(TIMESTAMP)
    closed_by_id = Column(BigInteger, nullable=False)
    closed_by_login = Column(Text, nullable=False)
    comments = Column(BigInteger)
    htmlurl = Column(Text)
    labels = Column(ARRAY(Text()), nullable=False)
    locked = Column(Boolean)
    milestone_id = Column(Text, nullable=False)
    milestone_title = Column(Text, nullable=False)
    number = Column(BigInteger)
    state = Column(Text)
    title = Column(Text)


class Organization(Base,
                   DeliveryMixin,
                   IDMixin,
                   UpdatedMixin):
    __tablename__ = "github_organizations"

    avatar_url = Column(Text)
    collaborators = Column(BigInteger)
    description = Column(Text)
    email = Column(Text)
    htmlurl = Column(Text)
    login = Column(Text)
    name = Column(Text)
    owned_private_repos = Column(BigInteger)
    public_repos = Column(BigInteger)
    total_private_repos = Column(BigInteger)


class PullRequestComment(Base,
                         BodyMixin,
                         DeliveryMixin,
                         IDMixin,
                         RepositoryMixin,
                         UpdatedMixin,
                         UserMixin,
                         ):
    __tablename__ = "github_pull_request_comments"

    author_association = Column(Text)
    commit_id = Column(Text)
    diff_hunk = Column(Text)
    htmlurl = Column(Text)
    in_reply_to = Column(BigInteger)
    original_commit_id = Column(Text)
    original_position = Column(BigInteger)
    path = Column(Text)
    position = Column(BigInteger)
    pull_request_number = Column(BigInteger, nullable=False)
    pull_request_review_id = Column(BigInteger)


class PullRequestCommit(Base,
                        RepositoryMixin,
                        ):
    __tablename__ = "github_pull_request_commits"

    pull_request_id = Column(BigInteger, primary_key=True)
    pull_request_number = Column(BigInteger)
    author_login = Column(Text)
    author_email = Column(Text)
    author_name = Column(Text)
    author_date = Column(TIMESTAMP(True))
    commiter_login = Column(Text)
    commiter_email = Column(Text)
    commiter_name = Column(Text)
    commit_date = Column(TIMESTAMP(True))
    sha = Column(Text, primary_key=True)
    additions = Column(Integer)
    deletions = Column(Integer)
    message = Column(Text)
    created_at = synonym("commit_date")


PullRequestCommit.created_at.key = "commit_date"


class PullRequestReview(Base,
                        BodyMixin,
                        DeliveryMixin,
                        IDMixin,
                        RepositoryMixin,
                        UserMixin,
                        ):
    __tablename__ = "github_pull_request_reviews"

    commit_id = Column(Text)
    htmlurl = Column(Text)
    pull_request_number = Column(BigInteger, nullable=False)
    state = Column(Text)
    submitted_at = Column(TIMESTAMP)
    created_at = synonym("submitted_at")


PullRequestReview.created_at.key = "submitted_at"


class PullRequest(Base,
                  BodyMixin,
                  DeliveryMixin,
                  IDMixin,
                  RepositoryMixin,
                  UpdatedMixin,
                  UserMixin,
                  ):
    __tablename__ = "github_pull_requests"

    additions = Column(BigInteger)
    assignees = Column(ARRAY(Text()), nullable=False)
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
    head_repository_name = Column(Text, nullable=False)
    head_repository_owner = Column(Text, nullable=False)
    head_repository_fullname = Column(Text, nullable=False)
    head_sha = Column(Text, nullable=False)
    head_user = Column(Text, nullable=False)
    htmlurl = Column(Text)
    labels = Column(ARRAY(Text()), nullable=False)
    maintainer_can_modify = Column(Boolean)
    merge_commit_sha = Column(Text)
    mergeable = Column(Boolean)
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
    __tablename__ = "github_push_commits"

    delivery_id = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP)
    id = Column(BigInteger)
    push_id = Column(Text)
    message = Column(Text)
    author_login = Column(Text)
    url = Column(Text)
    sha = Column(Text, primary_key=True)
    committer_login = Column(Text)
    added = Column(ARRAY(Text()))
    removed = Column(ARRAY(Text()))
    modified = Column(ARRAY(Text()))
    pusher_login = Column(Text, nullable=False)


class Repository(Base,
                 DeliveryMixin,
                 IDMixin,
                 UpdatedMixin,
                 ):
    __tablename__ = "github_repositories"

    allow_merge_commit = Column(Boolean)
    allow_rebase_merge = Column(Boolean)
    allow_squash_merge = Column(Boolean)
    archived = Column(Boolean)
    clone_url = Column(Text)
    default_branch = Column(Text)
    description = Column(Text)
    disabled = Column(Boolean)
    fork = Column(Boolean)
    forks_count = Column(BigInteger)
    fullname = Column(Text)
    has_issues = Column(Boolean)
    has_wiki = Column(Boolean)
    homepage = Column(Text)
    htmlurl = Column(Text)
    language = Column(Text)
    name = Column(Text)
    open_issues_count = Column(BigInteger)
    owner_id = Column(BigInteger, nullable=False)
    owner_login = Column(Text, nullable=False)
    owner_type = Column(Text, nullable=False)
    private = Column(Boolean)
    pushed_at = Column(TIMESTAMP)
    sshurl = Column(Text)
    stargazers_count = Column(BigInteger)
    topics = Column(ARRAY(Text()), nullable=False)
    watchers_count = Column(BigInteger)


class User(Base,
           IDMixin,
           UpdatedMixin,
           ):
    __tablename__ = "github_users"

    avatar_url = Column(Text)
    bio = Column(Text)
    company = Column(Text)
    email = Column(Text)
    followers = Column(BigInteger)
    following = Column(BigInteger)
    hireable = Column(Boolean)
    htmlurl = Column(Text)
    location = Column(Text)
    login = Column(Text)
    name = Column(Text)
    organization_id = Column(BigInteger)
    organization_login = Column(Text)
    owned_private_repos = Column(BigInteger)
    private_gists = Column(BigInteger)
    public_gists = Column(BigInteger)
    public_repos = Column(BigInteger)
    total_private_repos = Column(BigInteger)
