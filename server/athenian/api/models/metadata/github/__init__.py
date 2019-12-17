from sqlalchemy import ARRAY, BigInteger, Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class VersionedMixin:
    versions = Column("versions", ARRAY(Integer()), index=True)
    sum256 = Column("sum256", String(64), primary_key=True)
    id = Column("id", BigInteger)
    node_id = Column("node_id", Text)


class BodyMixin:
    body = Column("body", Text)


class UpdatedMixin:
    created_at = Column("created_at", DateTime(True))
    updated_at = Column("updated_at", DateTime(True))


class UserMixin:
    user_id = Column("user_id", BigInteger, nullable=False)
    user_login = Column("user_login", Text, nullable=False)


class RepositoryMixin:
    repository_name = Column("repository_name", Text, nullable=False)
    repository_owner = Column("repository_owner", Text, nullable=False)


"""
These models were generated from the real DB schema with

sqlacodegen "postgresql://postgres:postgres@0.0.0.0:5432/postgres" --noclasses --nocomments --outfile __init__.py  # noqa

Sadly, the output had to be polished manually, that is:

- Declarative classes instead of imperative table definition.
- Renames.
- Mixins to stay DRY.
"""


class IssueComment(Base, VersionedMixin, BodyMixin, UpdatedMixin, UserMixin, RepositoryMixin):
    __tablename__ = "github_issue_comments_versioned"
    author_association = Column("author_association", Text)
    htmlurl = Column("htmlurl", Text)
    issue_number = Column("issue_number", BigInteger, nullable=False)


class Issue(Base, VersionedMixin, BodyMixin, UpdatedMixin, UserMixin, RepositoryMixin):
    __tablename__ = "github_issues_versioned"
    assignees = Column("assignees", ARRAY(Text()), nullable=False)
    closed_at = Column("closed_at", DateTime(True))
    closed_by_id = Column("closed_by_id", BigInteger, nullable=False)
    closed_by_login = Column("closed_by_login", Text, nullable=False)
    comments = Column("comments", BigInteger)
    htmlurl = Column("htmlurl", Text)
    labels = Column("labels", ARRAY(Text()), nullable=False)
    locked = Column("locked", Boolean)
    milestone_id = Column("milestone_id", Text, nullable=False)
    milestone_title = Column("milestone_title", Text, nullable=False)
    number = Column("number", BigInteger)
    state = Column("state", Text)
    title = Column("title", Text)


class Organization(Base, VersionedMixin, UpdatedMixin):
    __tablename__ = "github_organizations_versioned"
    avatar_url = Column("avatar_url", Text)
    collaborators = Column("collaborators", BigInteger)
    description = Column("description", Text)
    email = Column("email", Text)
    htmlurl = Column("htmlurl", Text)
    login = Column("login", Text)
    name = Column("name", Text)
    owned_private_repos = Column("owned_private_repos", BigInteger)
    public_repos = Column("public_repos", BigInteger)
    total_private_repos = Column("total_private_repos", BigInteger)


class PullRequestComment(Base, VersionedMixin, BodyMixin, UpdatedMixin, UserMixin,
                         RepositoryMixin):
    __tablename__ = "github_pull_request_comments_versioned"
    author_association = Column("author_association", Text)
    commit_id = Column("commit_id", Text)
    diff_hunk = Column("diff_hunk", Text)
    htmlurl = Column("htmlurl", Text)
    in_reply_to = Column("in_reply_to", BigInteger)
    original_commit_id = Column("original_commit_id", Text)
    original_position = Column("original_position", BigInteger)
    path = Column("path", Text)
    position = Column("position", BigInteger)
    pull_request_number = Column("pull_request_number", BigInteger, nullable=False)
    pull_request_review_id = Column("pull_request_review_id", BigInteger)


class PullRequestReview(Base, VersionedMixin, BodyMixin, UserMixin, RepositoryMixin):
    __tablename__ = "github_pull_request_reviews_versioned"
    commit_id = Column("commit_id", Text)
    htmlurl = Column("htmlurl", Text)
    pull_request_number = Column("pull_request_number", BigInteger, nullable=False)
    state = Column("state", Text)
    submitted_at = Column("submitted_at", DateTime(True))


class PullRequest(Base, VersionedMixin, BodyMixin, UpdatedMixin, UserMixin, RepositoryMixin):
    __tablename__ = "github_pull_requests_versioned"
    additions = Column("additions", BigInteger)
    assignees = Column("assignees", ARRAY(Text()), nullable=False)
    author_association = Column("author_association", Text)
    base_ref = Column("base_ref", Text, nullable=False)
    base_repository_name = Column("base_repository_name", Text, nullable=False)
    base_repository_owner = Column("base_repository_owner", Text, nullable=False)
    base_sha = Column("base_sha", Text, nullable=False)
    base_user = Column("base_user", Text, nullable=False)
    changed_files = Column("changed_files", BigInteger)
    closed_at = Column("closed_at", DateTime(True))
    comments = Column("comments", BigInteger)
    commits = Column("commits", BigInteger)
    deletions = Column("deletions", BigInteger)
    head_ref = Column("head_ref", Text, nullable=False)
    head_repository_name = Column("head_repository_name", Text, nullable=False)
    head_repository_owner = Column("head_repository_owner", Text, nullable=False)
    head_sha = Column("head_sha", Text, nullable=False)
    head_user = Column("head_user", Text, nullable=False)
    htmlurl = Column("htmlurl", Text)
    labels = Column("labels", ARRAY(Text()), nullable=False)
    maintainer_can_modify = Column("maintainer_can_modify", Boolean)
    merge_commit_sha = Column("merge_commit_sha", Text)
    mergeable = Column("mergeable", Boolean)
    merged = Column("merged", Boolean)
    merged_at = Column("merged_at", DateTime(True))
    merged_by_id = Column("merged_by_id", BigInteger, nullable=False)
    merged_by_login = Column("merged_by_login", Text, nullable=False)
    milestone_id = Column("milestone_id", Text, nullable=False)
    milestone_title = Column("milestone_title", Text, nullable=False)
    number = Column("number", BigInteger)
    review_comments = Column("review_comments", BigInteger)
    state = Column("state", Text)
    title = Column("title", Text)


class Repository(Base, VersionedMixin, UpdatedMixin):
    __tablename__ = "github_repositories_versioned"
    allow_merge_commit = Column("allow_merge_commit", Boolean)
    allow_rebase_merge = Column("allow_rebase_merge", Boolean)
    allow_squash_merge = Column("allow_squash_merge", Boolean)
    archived = Column("archived", Boolean)
    clone_url = Column("clone_url", Text)
    default_branch = Column("default_branch", Text)
    description = Column("description", Text)
    disabled = Column("disabled", Boolean)
    fork = Column("fork", Boolean)
    forks_count = Column("forks_count", BigInteger)
    full_name = Column("full_name", Text)
    has_issues = Column("has_issues", Boolean)
    has_wiki = Column("has_wiki", Boolean)
    homepage = Column("homepage", Text)
    htmlurl = Column("htmlurl", Text)
    language = Column("language", Text)
    name = Column("name", Text)
    open_issues_count = Column("open_issues_count", BigInteger)
    owner_id = Column("owner_id", BigInteger, nullable=False)
    owner_login = Column("owner_login", Text, nullable=False)
    owner_type = Column("owner_type", Text, nullable=False)
    private = Column("private", Boolean)
    pushed_at = Column("pushed_at", DateTime(True))
    sshurl = Column("sshurl", Text)
    stargazers_count = Column("stargazers_count", BigInteger)
    topics = Column("topics", ARRAY(Text()), nullable=False)
    watchers_count = Column("watchers_count", BigInteger)


class User(Base, VersionedMixin, UpdatedMixin):
    __tablename__ = "github_users_versioned"
    avatar_url = Column("avatar_url", Text)
    bio = Column("bio", Text)
    company = Column("company", Text)
    email = Column("email", Text)
    followers = Column("followers", BigInteger)
    following = Column("following", BigInteger)
    hireable = Column("hireable", Boolean)
    htmlurl = Column("htmlurl", Text)
    location = Column("location", Text)
    login = Column("login", Text)
    name = Column("name", Text)
    organization_id = Column("organization_id", BigInteger, nullable=False)
    organization_login = Column("organization_login", Text, nullable=False)
    owned_private_repos = Column("owned_private_repos", BigInteger)
    private_gists = Column("private_gists", BigInteger)
    public_gists = Column("public_gists", BigInteger)
    public_repos = Column("public_repos", BigInteger)
    total_private_repos = Column("total_private_repos", BigInteger)
