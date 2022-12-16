from datetime import datetime, timezone
import enum

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Enum,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects import sqlite
from sqlalchemy.dialects.postgresql import JSONB

from athenian.api.models import always_unequal, create_base

Base = create_base()


JSONType = JSONB().with_variant(JSON(), sqlite.dialect.name)


def create_collection_mixin(name: str) -> type:
    """Create the collections mixin according to the required column name."""

    class CollectionMixin:
        name = Column(String(), nullable=False)

    setattr(
        CollectionMixin,
        name,
        Column(always_unequal(JSONType), nullable=False),
    )

    return CollectionMixin


def create_time_mixin(
    created_at: bool = False,
    updated_at: bool = False,
    nullable: bool = False,
) -> type:
    """Create the mixin accorinding to the required columns."""
    created_at_ = created_at
    updated_at_ = updated_at

    class TimeMixin:
        if created_at_:
            created_at = Column(
                TIMESTAMP(timezone=True),
                nullable=nullable,
                default=lambda: datetime.now(timezone.utc),
                server_default=func.now(),
            )
        if updated_at_:
            updated_at = Column(
                TIMESTAMP(timezone=True),
                nullable=nullable,
                default=lambda: datetime.now(timezone.utc),
                server_default=func.now(),
                onupdate=lambda ctx: datetime.now(timezone.utc),
            )

    return TimeMixin


class RepositorySet(
    create_time_mixin(created_at=True, updated_at=True), create_collection_mixin("items"), Base,
):
    """A group of repositories identified by an integer."""

    __tablename__ = "repository_sets"
    __table_args__ = (
        UniqueConstraint("owner_id", "name", name="uc_owner_name2"),
        {"sqlite_autoincrement": True},
    )
    ALL = "all"  # <<< constant name of the main reposet

    id = Column(Integer(), primary_key=True)
    owner_id = Column(
        Integer(), ForeignKey("accounts.id", name="fk_reposet_owner"), nullable=False,
    )
    updates_count = Column(
        always_unequal(Integer()),
        nullable=False,
        default=1,
        onupdate=lambda ctx: ctx.get_current_parameters()["updates_count"] + 1,
    )
    tracking_re = Column(Text(), nullable=False, default=".*", server_default=".*")
    precomputed = Column(Boolean(), nullable=False, default=False, server_default="false")


class UserAccount(create_time_mixin(created_at=True), Base):
    """User<>account many-to-many relations."""

    __tablename__ = "user_accounts"

    user_id = Column(String(), primary_key=True)
    account_id = Column(
        Integer(),
        ForeignKey("accounts.id", name="fk_user_account"),
        nullable=False,
        primary_key=True,
    )
    is_admin = Column(Boolean(), nullable=False, default=False)


class BanishedUserAccount(create_time_mixin(created_at=True), Base):
    """Deleted user<>account many-to-many relations."""

    __tablename__ = "banished_user_accounts"

    user_id = Column(String(), primary_key=True)
    account_id = Column(
        Integer(),
        ForeignKey("accounts.id", name="fk_banished_user_account"),
        nullable=False,
        primary_key=True,
    )


class Account(create_time_mixin(created_at=True), Base):
    """Group of users, some are admins and some are regular."""

    __tablename__ = "accounts"
    __table_args__ = {"sqlite_autoincrement": True}
    missing_secret = "0" * 8

    id = Column(Integer(), primary_key=True)
    secret_salt = Column(Integer(), nullable=False)
    secret = Column(String(8), nullable=False)
    expires_at = Column(TIMESTAMP(timezone=True), nullable=False)


class Team(
    create_time_mixin(created_at=True, updated_at=True), create_collection_mixin("members"), Base,
):
    """Group of users part of the same team."""

    __tablename__ = "teams"
    __table_args__ = (
        UniqueConstraint("owner_id", "name", name="uc_owner_name"),
        CheckConstraint("id != parent_id", name="cc_parent_self_reference"),
        {"sqlite_autoincrement": True},
    )
    BOTS = "Bots"  # the name of the special team which contains bots
    ROOT = "Root"  # the name of the special artificial unique root team

    id = Column(Integer(), primary_key=True)
    owner_id = Column(
        Integer(), ForeignKey("accounts.id", name="fk_reposet_owner"), nullable=False, index=True,
    )
    parent_id = Column(Integer(), ForeignKey("teams.id", name="fk_team_parent"))
    root_id = "root_id"
    origin_node_id = Column(BigInteger())
    """The id of the corresponding team in metadata DB, pointing github.node_team.graph_id

    Will be NULL for
    - special BOTS and ROOT teams
    - locally created teams
    - teams existing at the time the field was introduced, and for which the migration failed to
      discover the metadata team id

    """


class AccountGitHubAccount(create_time_mixin(created_at=True, nullable=True), Base):
    """Mapping API account -> metadata account, one-to-many."""

    __tablename__ = "account_github_accounts"

    id = Column(BigInteger(), primary_key=True, autoincrement=False)
    account_id = Column(
        Integer(), ForeignKey("accounts.id", name="fk_installation_id_owner"), nullable=False,
    )


class AccountJiraInstallation(create_time_mixin(created_at=True, nullable=True), Base):
    """Mapping account -> installation_id, one-to-many."""

    __tablename__ = "account_jira_installations"
    __table_args__ = (UniqueConstraint("account_id", name="uc_jira_installations_account_id"),)

    id = Column(BigInteger(), primary_key=True, autoincrement=False)
    account_id = Column(
        Integer(), ForeignKey("accounts.id", name="fk_installation_id_owner2"), nullable=False,
    )


class Invitation(create_time_mixin(created_at=True), Base):
    """Account invitations, each maps to a URL that invitees should click."""

    __tablename__ = "invitations"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer(), primary_key=True)
    salt = Column(Integer(), nullable=False)
    account_id = Column(
        Integer(), ForeignKey("accounts.id", name="fk_invitation_account"), nullable=False,
    )
    is_active = Column(Boolean, nullable=False, default=True)
    accepted = Column(Integer(), nullable=False, default=0)
    created_by = Column(String())


class God(create_time_mixin(updated_at=True), Base):
    """Secret user mappings for chosen ones."""

    __tablename__ = "gods"

    user_id = Column(String(), primary_key=True)
    mapped_id = Column(String(), nullable=True)


class ReleaseSetting(create_time_mixin(updated_at=True), Base):
    """Release matching rules per repo."""

    __tablename__ = "release_settings"

    repo_id = Column(BigInteger(), primary_key=True)
    """The identifier of the repository the release setting is about.

    Points to the mdb `Repository.node_id` column.
    """

    logical_name = Column(String(), primary_key=True)
    """
    The logical name of repository, if the release setting is about a logical repository.

    Empty string if the setting is about a physical repo.
    This is the "bare" logical name, without prefix and physical repo name.
    """

    account_id = Column(
        Integer(), ForeignKey("accounts.id", name="fk_release_settings_account"), primary_key=True,
    )
    branches = Column(
        String(), nullable=False, default="{{default}}", server_default="{{default}}",
    )
    tags = Column(String(), nullable=False, default=".*", server_default=".*")
    events = Column(String(), nullable=False, default=".*", server_default=".*")
    match = Column(SmallInteger(), nullable=False, default=2, server_default="2")


class FeatureComponent(enum.IntEnum):
    """Athenian stack parts: the frontend, the backend, etc."""

    webapp = 1
    server = 2


class Feature(create_time_mixin(created_at=True, updated_at=True), Base):
    """Product features."""

    __tablename__ = "features"
    __table_args__ = (
        UniqueConstraint("name", "component", name="uc_feature_name_component"),
        {"sqlite_autoincrement": True},
    )

    USER_ORG_MEMBERSHIP_CHECK = "user_org_membership_check"
    GITHUB_LOGIN_ENABLED = "github_login_enabled"
    QUANTILE_STRIDE = "quantile_stride"
    TEAM_SYNC_ENABLED = "team_sync_enabled"

    id = Column(Integer(), primary_key=True)
    name = Column(String(), nullable=False)
    component = Column(Enum(FeatureComponent), nullable=False)
    enabled = Column(Boolean(), nullable=False, default=False, server_default="false")
    default_parameters = Column(JSONType, nullable=False, default={}, server_default="{}")


class AccountFeature(create_time_mixin(created_at=True, updated_at=True), Base):
    """Product features -> accounts many-to-many mapping."""

    __tablename__ = "account_features"

    account_id = Column(
        Integer(), ForeignKey("accounts.id", name="fk_account_features_account"), primary_key=True,
    )
    feature_id = Column(
        Integer(), ForeignKey("features.id", name="fk_account_features_feature"), primary_key=True,
    )
    enabled = Column(Boolean(), nullable=False, default=False, server_default="false")
    parameters = Column(JSONType)


class UserToken(create_time_mixin(created_at=True, updated_at=True), Base):
    """Personal Access Tokens of the accounts."""

    __tablename__ = "user_tokens"
    __table_args__ = (
        UniqueConstraint("name", "user_id", "account_id", name="uc_token_name"),
        ForeignKeyConstraint(
            ("account_id", "user_id"),
            ("user_accounts.account_id", "user_accounts.user_id"),
            name="fk_account_tokens_user",
            ondelete="CASCADE",
        ),
        {"sqlite_autoincrement": True},
    )

    id = Column(BigInteger().with_variant(Integer(), sqlite.dialect.name), primary_key=True)
    account_id = Column(Integer(), nullable=False)
    user_id = Column(String(), nullable=False)
    name = Column(String(256), nullable=False)
    last_used_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )


class JIRAProjectSetting(create_time_mixin(updated_at=True), Base):
    """JIRA projects enabled/disabled."""

    __tablename__ = "jira_projects"

    account_id = Column(
        Integer(), ForeignKey("accounts.id", name="fk_jira_projects_account"), primary_key=True,
    )
    key = Column(Text(), primary_key=True)
    enabled = Column(Boolean(), nullable=False)


class MappedJIRAIdentity(create_time_mixin(created_at=True, updated_at=True), Base):
    """JIRA identity mapping."""

    __tablename__ = "jira_identity_mapping"
    __table_args__ = (
        Index(
            "jira_identity_mapping_jira_user_id",
            "account_id",
            "jira_user_id",
            postgresql_include=["github_user_id"],
        ),
    )

    account_id = Column(
        Integer(),
        ForeignKey("accounts.id", name="fk_jira_identity_mapping_account"),
        primary_key=True,
    )
    github_user_id = Column(BigInteger(), primary_key=True)
    jira_user_id = Column(Text())
    confidence = Column(Float())


class WorkType(create_time_mixin(created_at=True, updated_at=True), Base):
    """Work type - a set of rules to group the PRs, releases, etc."""

    __tablename__ = "work_types"
    __table_args__ = (
        UniqueConstraint("account_id", "name", name="uc_work_type"),
        {"sqlite_autoincrement": True},
    )

    id = Column(BigInteger().with_variant(Integer(), sqlite.dialect.name), primary_key=True)
    account_id = Column(
        Integer(), ForeignKey("accounts.id", name="fk_work_type_account"), nullable=False,
    )
    name = Column(Text(), nullable=False)
    color = Column(String(6), nullable=False)
    rules = Column(JSONType, nullable=False)


class LogicalRepository(create_time_mixin(created_at=True, updated_at=True), Base):
    """Logical repository - rules to select PRs and releases in a monorepo."""

    __tablename__ = "logical_repositories"
    __table_args__ = (
        UniqueConstraint("account_id", "name", "repository_id", name="uc_logical_repository"),
        {"sqlite_autoincrement": True},
    )

    id = Column(BigInteger().with_variant(Integer(), sqlite.dialect.name), primary_key=True)
    account_id = Column(
        Integer(), ForeignKey("accounts.id", name="fk_logical_repository_account"), nullable=False,
    )
    name = Column(Text(), nullable=False)
    repository_id = Column(BigInteger(), nullable=False)
    prs = Column(JSONType, nullable=False, default={}, server_default="{}")
    deployments = Column(JSONType, nullable=False, default={}, server_default="{}")


class Goal(create_time_mixin(created_at=True, updated_at=True), Base):
    """Goal - A metric target for a team."""

    __tablename__ = "goals"

    id = Column(Integer(), primary_key=True)
    account_id = Column(
        Integer(),
        ForeignKey("accounts.id", name="fk_goal_account"),
        nullable=False,
        index=True,
    )
    name = Column(String, nullable=False)
    metric = Column(String, nullable=False)
    valid_from = Column(TIMESTAMP(timezone=True), nullable=False)
    expires_at = Column(TIMESTAMP(timezone=True), nullable=False)
    archived = Column(Boolean, default=False, nullable=False, server_default="false")
    environments = Column(JSONType, nullable=True)
    """The deployment environments defining the goal scope."""
    # the following are the defaults for TeamGoal
    repositories = Column(JSONType, nullable=True)
    jira_projects = Column(JSONType, nullable=True)
    jira_priorities = Column(JSONType, nullable=True)
    jira_issue_types = Column(JSONType, nullable=True)
    metric_params = Column(JSONType, nullable=True)


class GoalTemplate(create_time_mixin(created_at=True, updated_at=True), Base):
    """A template to generate a Goal."""

    __tablename__ = "goal_templates"
    __table_args__ = (
        UniqueConstraint("account_id", "name", name="uc_goal_templates_account_id_name"),
        {"sqlite_autoincrement": True},
    )

    id = Column(Integer(), primary_key=True)
    account_id = Column(
        Integer(),
        ForeignKey("accounts.id", name="fk_goal_template_account"),
        nullable=False,
        index=True,
    )
    name = Column(String, nullable=False)
    # the following are the defaults for Goal
    metric = Column(String, nullable=False)
    environments = Column(JSONType, nullable=True)
    repositories = Column(JSONType, nullable=True)
    jira_projects = Column(JSONType, nullable=True)
    jira_priorities = Column(JSONType, nullable=True)
    jira_issue_types = Column(JSONType, nullable=True)
    metric_params = Column(JSONType, nullable=True)


class TeamGoal(create_time_mixin(created_at=True, updated_at=True), Base):
    """A Goal applied to a Team, with a specific target."""

    __tablename__ = "team_goals"

    goal_id = Column(
        BigInteger().with_variant(Integer(), sqlite.dialect.name),
        ForeignKey("goals.id", name="fk_team_goal_goal", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )
    team_id = Column(
        Integer(),
        ForeignKey("teams.id", name="fk_team_goal_team", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )
    target = Column(JSONType, nullable=False)
    repositories = Column(JSONType, nullable=True)
    jira_projects = Column(JSONType, nullable=True)
    jira_priorities = Column(JSONType, nullable=True)
    jira_issue_types = Column(JSONType, nullable=True)
    metric_params = Column(JSONType, nullable=True)


class Share(create_time_mixin(created_at=True), Base):
    """Saved UI views state."""

    __tablename__ = "shares"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(BigInteger().with_variant(Integer(), sqlite.dialect.name), primary_key=True)
    created_by = Column(String(), nullable=False)
    divine = Column(Boolean, nullable=False, default=False, server_default="false")
    data = Column(JSONType, nullable=False)


class TeamDashboard(create_time_mixin(created_at=True, updated_at=True), Base):
    """A dashboard displaying information for a team."""

    __tablename__ = "team_dashboards"

    id = Column(Integer(), primary_key=True)
    team_id = Column(
        Integer(),
        ForeignKey("teams.id", name="fk_team_dashboard_team", ondelete="CASCADE"),
        nullable=False,
    )
    """The team this dashboard belongs to."""


class DashboardChart(create_time_mixin(created_at=True, updated_at=True), Base):
    """A chart showing metric values in a dashboard."""

    __tablename__ = "dashboard_charts"

    id = Column(Integer(), primary_key=True)
    dashboard_id = Column(
        Integer(),
        ForeignKey("team_dashboards.id", name="fk_chart_dashboard", ondelete="CASCADE"),
        nullable=False,
    )
    position = Column(Integer(), nullable=False)
    """Position of the chart in the containing dashboard."""

    metric = Column(String, nullable=False)
    """The metric this chart is displaying."""

    time_from = Column(TIMESTAMP(timezone=True), nullable=True)
    time_to = Column(TIMESTAMP(timezone=True), nullable=True)
    """The fixed time interval of the chart.

    If `time_from` and `time_to` are expressed `time_interval` will be NULL.
    """
    time_interval = Column(String, nullable=True)
    """The relative time interval of the chart in ISO-8601 format.

    Examples: "P1W", "P10D".
    If `time_interval` is expressed `time_from` and `time_to` will be NULL.
    """

    name = Column(String, nullable=False)
    description = Column(String, nullable=False)

    # deferred constraint is useful to update positions easier but it's only available on postgres
    __table_args__ = (
        UniqueConstraint(
            "dashboard_id",
            "position",
            name="uc_chart_dashboard_id_position",
            deferrable=True,
            _create_rule=lambda compiler: compiler.dialect.name == "postgresql",
        ),
        UniqueConstraint(
            "dashboard_id",
            "position",
            name="uc_chart_dashboard_id_position",
            _create_rule=lambda compiler: compiler.dialect.name != "postgresql",
        ),
    )
