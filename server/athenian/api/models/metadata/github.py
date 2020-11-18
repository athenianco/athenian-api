import dateutil.parser
from sqlalchemy import BigInteger, Boolean, Column, Integer, Text, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import synonym


Base = declarative_base()


# -- MIXINS --


class IDMixin:
    node_id = Column(Text, primary_key=True)


class IDMixinNG:
    id = Column(Text, primary_key=True)

    @declared_attr
    def node_id(self):
        """Return 'id' as 'node_id'."""
        return synonym("id")


class AccountMixin:
    acc_id = Column(BigInteger, primary_key=True)


class BodyMixin:
    body = Column(Text)


class UpdatedMixin:
    created_at = Column(TIMESTAMP(timezone=True))
    updated_at = Column(TIMESTAMP(timezone=True))


class UserMixin:
    user_node_id = Column(Text)
    user_login = Column(Text)


class RepositoryMixin:
    repository_full_name = Column(Text, nullable=False)
    repository_node_id = Column(Text, nullable=False)


class ParentChildMixin:
    parent_id = Column(Text, primary_key=True)
    child_id = Column(Text, primary_key=True)


class PullRequestMixin:
    pull_request_node_id = Column(Text, nullable=False)


class PullRequestPKMixin:
    pull_request_node_id = Column(Text, primary_key=True)


# -- TABLES --


class Account(Base, UpdatedMixin):
    __tablename__ = "accounts"
    __table_args__ = {"schema": "github"}

    id = Column(BigInteger, primary_key=True)
    owner_id = Column(BigInteger, nullable=False)
    owner_login = Column(Text, nullable=False)


class AccountRepository(Base):
    __tablename__ = "account_repos_log"
    __table_args__ = {"schema": "github"}

    acc_id = Column(BigInteger, primary_key=True)
    repo_node_id = Column(Text, primary_key=True)
    repo_full_name = Column(Text, nullable=False)
    event_id = Column(Text, nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False)
    enabled = Column(Boolean, nullable=False)


class OrganizationMember(Base):
    __tablename__ = "github_node_organization_members_with_role"

    parent_id = Column(Text, primary_key=True, comment="Organization ID")
    child_id = Column(Text, primary_key=True, comment="User ID")


class FetchProgress(Base, UpdatedMixin):
    __tablename__ = "github_fetch_progress"

    event_id = Column(Text, primary_key=True)
    node_type = Column(Text, primary_key=True)
    nodes_processed = Column(BigInteger, default=0)
    nodes_total = Column(BigInteger, nullable=False)


class PullRequestComment(Base,
                         IDMixin,
                         UpdatedMixin,
                         UserMixin,
                         RepositoryMixin,
                         PullRequestMixin,
                         ):
    __tablename__ = "github_pull_request_comments_compat"


class PullRequestReviewComment(Base,
                               IDMixin,
                               UpdatedMixin,
                               UserMixin,
                               RepositoryMixin,
                               PullRequestMixin,
                               ):
    __tablename__ = "github_pull_request_review_comments_compat"


class PullRequestCommit(Base,
                        RepositoryMixin,
                        PullRequestPKMixin,
                        ):
    __tablename__ = "github_pull_request_commits_compat"

    sha = Column(Text, primary_key=True)
    node_id = Column(Text, nullable=False)
    commit_node_id = Column(Text, nullable=False)
    author_login = Column(Text)
    author_date = Column(Text, nullable=False)
    authored_date = Column(TIMESTAMP(timezone=True), nullable=False)
    committer_login = Column(Text)
    commit_date = Column(Text, nullable=False)
    committed_date = Column(TIMESTAMP(timezone=True), nullable=False)
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
                        IDMixin,
                        UserMixin,
                        PullRequestMixin,
                        RepositoryMixin,
                        ):
    __tablename__ = "github_pull_request_reviews_compat"

    state = Column(Text, nullable=False)
    submitted_at = Column(TIMESTAMP(timezone=True), nullable=False)
    created_at = synonym("submitted_at")


PullRequestReview.created_at.key = "submitted_at"


class PullRequestReviewRequest(Base,
                               IDMixinNG,
                               ):
    __tablename__ = "github_node_review_requested_event"

    created_at = Column(TIMESTAMP(timezone=True))
    pull_request = Column(Text, nullable=False)
    pull_request_node_id = synonym("pull_request")
    # FIXME(vmarkovtsev): set nullable=False when ENG-303 is resolved
    requested_reviewer = Column(Text, nullable=True)


PullRequestReviewRequest.pull_request_node_id.key = "pull_request"
PullRequestReviewRequest.node_id.key = "id"


class PullRequest(Base,
                  IDMixin,
                  RepositoryMixin,
                  UpdatedMixin,
                  UserMixin,
                  ):
    __tablename__ = "github_pull_requests_compat"

    additions = Column(BigInteger)
    base_ref = Column(Text, nullable=False)
    changed_files = Column(BigInteger)
    closed_at = Column(TIMESTAMP(timezone=True))
    deletions = Column(BigInteger)
    head_ref = Column(Text, nullable=False)
    hidden = Column(Boolean)
    htmlurl = Column(Text)
    merge_commit_id = Column(Text)
    merge_commit_sha = Column(Text)
    merged = Column(Boolean)
    merged_at = Column(TIMESTAMP(timezone=True))
    merged_by = Column(Text)
    merged_by_login = Column(Text)
    number = Column(BigInteger, nullable=False)
    title = Column(Text)


class PushCommit(Base,
                 RepositoryMixin):
    __tablename__ = "github_push_commits_compat"

    node_id = Column(Text, primary_key=True)
    message = Column(Text, nullable=False)
    pushed_date = Column(TIMESTAMP(timezone=True))
    author_login = Column(Text)
    author_user = Column(Text)
    author_avatar_url = Column(Text)
    author_email = Column(Text)
    author_name = Column(Text)
    author_date = Column(Text, nullable=False)
    authored_date = Column(TIMESTAMP(timezone=True), nullable=False)
    url = Column(Text)
    sha = Column(Text, nullable=False)
    committer_login = Column(Text)
    committer_user = Column(Text)
    committer_avatar_url = Column(Text)
    committer_email = Column(Text)
    committer_name = Column(Text)
    commit_date = Column(Text, nullable=False)
    committed_date = Column(TIMESTAMP(timezone=True), nullable=False)
    additions = Column(BigInteger, nullable=False)
    deletions = Column(BigInteger, nullable=False)
    changed_files = Column(BigInteger, nullable=False)


class Repository(Base,
                 IDMixin,
                 UpdatedMixin,
                 ):
    __tablename__ = "github_repositories_v2_compat"

    archived = Column(Boolean)
    description = Column(Text)
    disabled = Column(Boolean)
    fork = Column(Boolean)
    full_name = Column(Text, nullable=False)
    html_url = Column(Text)


class User(Base,
           IDMixin,
           UpdatedMixin,
           ):
    __tablename__ = "github_users_v2_compat"

    avatar_url = Column(Text, nullable=False)
    email = Column(Text)
    login = Column(Text, nullable=False)
    name = Column(Text)


class Release(Base,
              RepositoryMixin,
              ):
    __tablename__ = "github_releases_compat"

    id = Column(Text, primary_key=True)
    author = Column(Text, nullable=False)
    name = Column(Text)
    published_at = Column(TIMESTAMP(timezone=True))
    tag = Column(Text)
    url = Column(Text)
    sha = Column(Text, nullable=False)
    commit_id = Column(Text, nullable=False)


class NodeCommit(Base):
    __tablename__ = "github_node_commit"

    id = Column(Text, primary_key=True)
    oid = Column(Text, nullable=False)
    sha = synonym(oid)
    repository = Column(Text, nullable=False)
    committed_date = Column(TIMESTAMP(timezone=True))
    committer_user = Column(Text)
    author_user = Column(Text)


NodeCommit.sha.key = "oid"


class NodeCommitParent(Base,
                       ParentChildMixin,
                       ):
    __tablename__ = "github_node_commit_parents"

    index = Column(Integer, nullable=False)


class NodePullRequestCommit(Base, IDMixinNG):
    __tablename__ = "github_node_pull_request_commit"

    commit = Column(Text, nullable=False)
    pull_request = Column(Text, nullable=False)


class NodeRepository(Base, IDMixinNG):
    __tablename__ = "github_node_repository"

    name_with_owner = Column(Text, nullable=False)


class Branch(Base,
             RepositoryMixin,
             ):
    __tablename__ = "github_branches_compat"

    branch_id = Column(Text, primary_key=True)
    branch_name = Column(Text, nullable=False)
    is_default = Column(Boolean, nullable=False)
    commit_id = Column(Text, nullable=False)
    commit_sha = Column(Text, nullable=False)
    commit_date = "commit_date"


class PullRequestLabel(Base,
                       IDMixin,
                       UpdatedMixin,
                       RepositoryMixin,
                       PullRequestPKMixin,
                       ):
    __tablename__ = "github_pull_request_labels_compat"

    name = Column(Text, nullable=False)
    description = Column(Text)
    color = Column(Text, nullable=False)


class Bot(Base):
    __tablename__ = "github_bots_compat"

    login = Column(Text, primary_key=True)


class NodeRepositoryRef(Base, ParentChildMixin):
    __tablename__ = "github_node_repository_refs"


class NodePullRequestJiraIssues(Base, IDMixin):
    __tablename__ = "node_pull_request_jira_issues"
    __table_args__ = {"schema": "github"}

    jira_acc = Column(BigInteger, nullable=False)
    jira_id = Column(Text, nullable=False)


class NodeUser(Base, IDMixinNG):
    __tablename__ = "github_node_user"

    database_id = Column(BigInteger, unique=True)
    login = Column(Text, nullable=False)


class SchemaMigration(Base):
    __tablename__ = "schema_migrations"

    version = Column(BigInteger, primary_key=True)
    dirty = Column(Boolean, nullable=False)


class Organization(Base, IDMixinNG, AccountMixin):
    __tablename__ = "github_node_organization"

    login = Column(Text, nullable=False)
    name = Column(Text)
    avatar_url = Column(Text)
