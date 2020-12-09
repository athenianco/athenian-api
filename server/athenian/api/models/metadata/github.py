import dateutil.parser
from sqlalchemy import BigInteger, Boolean, Column, Integer, Text, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import synonym


Base = declarative_base()


# -- MIXINS --

class GitHubSchemaMixin:
    __table_args__ = {"schema": "github"}


class AccountMixin:
    acc_id = Column(BigInteger, primary_key=True)


class IDMixin(AccountMixin):
    node_id = Column(Text, primary_key=True)


class IDMixinNG(AccountMixin):
    id = Column(Text, primary_key=True)

    @declared_attr
    def node_id(self):
        """Return 'id' as 'node_id'."""
        return synonym("id")


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


class Account(Base,
              GitHubSchemaMixin,
              UpdatedMixin,
              ):
    __tablename__ = "accounts"

    id = Column(BigInteger, primary_key=True)
    owner_id = Column(BigInteger, nullable=False)
    owner_login = Column(Text, nullable=False)


class AccountRepository(Base,
                        GitHubSchemaMixin,
                        AccountMixin,
                        ):
    __tablename__ = "account_repos_log"

    repo_node_id = Column(Text, primary_key=True)
    repo_full_name = Column(Text, nullable=False)
    event_id = Column(Text, nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False)
    enabled = Column(Boolean, nullable=False)


class OrganizationMember(Base,
                         AccountMixin,
                         ):
    __tablename__ = "github_node_organization_members_with_role"

    parent_id = Column(Text, primary_key=True, comment="Organization ID")
    child_id = Column(Text, primary_key=True, comment="User ID")


class FetchProgress(Base,
                    AccountMixin,
                    UpdatedMixin,
                    ):
    __tablename__ = "github_fetch_progress"

    event_id = Column(Text, primary_key=True)
    node_type = Column(Text, primary_key=True)
    nodes_processed = Column(BigInteger, default=0)
    nodes_total = Column(BigInteger, nullable=False)


class PullRequestComment(Base,
                         GitHubSchemaMixin,
                         IDMixin,
                         UpdatedMixin,
                         UserMixin,
                         RepositoryMixin,
                         PullRequestMixin,
                         ):
    __tablename__ = "api_pull_request_comments"


class PullRequestReviewComment(Base,
                               GitHubSchemaMixin,
                               IDMixin,
                               UpdatedMixin,
                               UserMixin,
                               RepositoryMixin,
                               PullRequestMixin,
                               ):
    __tablename__ = "api_pull_request_review_comments"


class PullRequestCommit(Base,
                        GitHubSchemaMixin,
                        IDMixin,
                        RepositoryMixin,
                        PullRequestPKMixin,
                        ):
    __tablename__ = "api_pull_request_commits"

    sha = Column(Text, nullable=False)
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
                        GitHubSchemaMixin,
                        IDMixin,
                        UserMixin,
                        PullRequestMixin,
                        RepositoryMixin,
                        ):
    __tablename__ = "api_pull_request_reviews"

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
                  GitHubSchemaMixin,
                  IDMixin,
                  RepositoryMixin,
                  UpdatedMixin,
                  UserMixin,
                  ):
    __tablename__ = "api_pull_requests"

    additions = Column(BigInteger)
    base_ref = Column(Text, nullable=False)
    changed_files = Column(BigInteger)
    closed_at = Column(TIMESTAMP(timezone=True))
    closed = Column(Boolean)
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
                 GitHubSchemaMixin,
                 IDMixin,
                 RepositoryMixin):
    __tablename__ = "api_push_commits"

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
                 GitHubSchemaMixin,
                 IDMixin,
                 UpdatedMixin,
                 ):
    __tablename__ = "api_repositories"

    archived = Column(Boolean)
    description = Column(Text)
    disabled = Column(Boolean)
    fork = Column(Boolean)
    full_name = Column(Text, nullable=False)
    html_url = Column(Text)


class User(Base,
           GitHubSchemaMixin,
           IDMixin,
           UpdatedMixin,
           ):
    __tablename__ = "api_users"

    avatar_url = Column(Text, nullable=False)
    email = Column(Text)
    login = Column(Text, nullable=False)
    name = Column(Text)


class Release(Base,
              GitHubSchemaMixin,
              IDMixinNG,
              RepositoryMixin,
              ):
    __tablename__ = "api_releases"

    author = Column(Text, nullable=False)
    name = Column(Text)
    published_at = Column(TIMESTAMP(timezone=True))
    tag = Column(Text)
    url = Column(Text)
    sha = Column(Text, nullable=False)
    commit_id = Column(Text, nullable=False)


class NodeCommit(Base,
                 IDMixinNG,
                 ):
    __tablename__ = "github_node_commit"

    oid = Column(Text, nullable=False)
    sha = synonym(oid)
    repository = Column(Text, nullable=False)
    committed_date = Column(TIMESTAMP(timezone=True))
    committer_user = Column(Text)
    author_user = Column(Text)


NodeCommit.sha.key = "oid"


class NodeCommitParent(Base,
                       ParentChildMixin,
                       AccountMixin,
                       ):
    __tablename__ = "github_node_commit_parents"

    index = Column(Integer, nullable=False)


class NodePullRequestCommit(Base,
                            IDMixinNG,
                            ):
    __tablename__ = "github_node_pull_request_commit"

    commit = Column(Text, nullable=False)
    pull_request = Column(Text, nullable=False)


class NodeRepository(Base,
                     IDMixinNG,
                     ):
    __tablename__ = "github_node_repository"

    name_with_owner = Column(Text, nullable=False)


class Branch(Base,
             GitHubSchemaMixin,
             AccountMixin,
             RepositoryMixin,
             ):
    __tablename__ = "api_branches"

    branch_id = Column(Text, primary_key=True)
    branch_name = Column(Text, nullable=False)
    is_default = Column(Boolean, nullable=False)
    commit_id = Column(Text, nullable=False)
    commit_sha = Column(Text, nullable=False)
    commit_date = "commit_date"


class PullRequestLabel(Base,
                       GitHubSchemaMixin,
                       IDMixin,
                       UpdatedMixin,
                       RepositoryMixin,
                       PullRequestPKMixin,
                       ):
    __tablename__ = "api_pull_request_labels"

    name = Column(Text, nullable=False)
    description = Column(Text)
    color = Column(Text, nullable=False)


class Bot(Base):
    __tablename__ = "github_bots_compat"

    login = Column(Text, primary_key=True)


class NodeRepositoryRef(Base,
                        ParentChildMixin,
                        AccountMixin,
                        ):
    __tablename__ = "github_node_repository_refs"


class NodePullRequestJiraIssues(Base,
                                GitHubSchemaMixin,
                                ):
    __tablename__ = "node_pull_request_jira_issues"

    node_id = Column(Text, primary_key=True)
    node_acc = Column(BigInteger, primary_key=True)
    jira_acc = Column(BigInteger, nullable=False)
    jira_id = Column(Text, nullable=False)


class NodeUser(Base,
               IDMixinNG,
               ):
    __tablename__ = "github_node_user"

    database_id = Column(BigInteger, unique=True)
    login = Column(Text, nullable=False)


class SchemaMigration(Base):
    __tablename__ = "schema_migrations"

    version = Column(BigInteger, primary_key=True)
    dirty = Column(Boolean, nullable=False)


class Organization(Base,
                   IDMixinNG,
                   ):
    __tablename__ = "github_node_organization"

    login = Column(Text, nullable=False)
    name = Column(Text)
    avatar_url = Column(Text)


class Team(Base,
           IDMixinNG,
           ):
    __tablename__ = "github_node_team"

    organization = Column(Text, nullable=False)
    name = Column(Text, nullable=False)
    description = Column(Text)
    parent_team = Column(Text)


class TeamMember(Base,
                 ParentChildMixin,
                 AccountMixin,
                 ):
    __tablename__ = "github_node_team_members"
