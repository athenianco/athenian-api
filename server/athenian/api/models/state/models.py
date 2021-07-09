import ctypes
from datetime import datetime, timezone
import enum
import json

from sqlalchemy import BigInteger, Boolean, CheckConstraint, Column, Enum, Float, ForeignKey, \
    ForeignKeyConstraint, func, Integer, JSON, SmallInteger, String, Text, TIMESTAMP, \
    UniqueConstraint
import xxhash

from athenian.api.models import always_unequal, create_base


Base = create_base()


def create_collection_mixin(name: str, with_checksum: bool) -> type:
    """Create the collections mixin according to the required column name."""

    class CollectionMixin:
        name = Column(String(256), nullable=False)

        @staticmethod
        def count_items(ctx):
            """Return the number of items in the collection."""
            return len(ctx.get_current_parameters()[name])

        if with_checksum:
            @staticmethod
            def calc_items_checksum(ctx):
                """Calculate the checksum of the items in the collection."""
                return ctypes.c_longlong(xxhash.xxh64_intdigest(json.dumps(
                    ctx.get_current_parameters()[name]))).value

    setattr(CollectionMixin, name, Column(always_unequal(JSON()), nullable=False))
    setattr(CollectionMixin, f"{name}_count",
            Column(always_unequal(Integer()), nullable=False, default=CollectionMixin.count_items,
                   onupdate=CollectionMixin.count_items))
    if with_checksum:
        setattr(CollectionMixin, f"{name}_checksum",
                Column(always_unequal(BigInteger()), nullable=False,
                       default=CollectionMixin.calc_items_checksum,
                       onupdate=CollectionMixin.calc_items_checksum))

    return CollectionMixin


def create_time_mixin(created_at: bool = False, updated_at: bool = False) -> type:
    """Create the mixin accorinding to the required columns."""
    created_at_ = created_at
    updated_at_ = updated_at

    class TimeMixin:
        if created_at_:
            created_at = Column(TIMESTAMP(timezone=True), nullable=False,
                                default=lambda: datetime.now(timezone.utc),
                                server_default=func.now())
        if updated_at_:
            updated_at = Column(TIMESTAMP(timezone=True), nullable=False,
                                default=lambda: datetime.now(timezone.utc),
                                server_default=func.now(),
                                onupdate=lambda ctx: datetime.now(timezone.utc))

    return TimeMixin


class RepositorySet(create_time_mixin(created_at=True, updated_at=True),
                    create_collection_mixin("items", with_checksum=True), Base):
    """A group of repositories identified by an integer."""

    __tablename__ = "repository_sets"
    __table_args__ = (UniqueConstraint("owner_id", "items_checksum", name="uc_owner_items"),
                      UniqueConstraint("owner_id", "name", name="uc_owner_name2"),
                      {"sqlite_autoincrement": True})
    ALL = "all"  # <<< constant name of the main reposet

    id = Column(Integer(), primary_key=True)
    owner_id = Column(Integer(), ForeignKey("accounts.id", name="fk_reposet_owner"),
                      nullable=False)
    updates_count = Column(always_unequal(Integer()), nullable=False, default=1,
                           onupdate=lambda ctx: ctx.get_current_parameters()["updates_count"] + 1)
    tracking_re = Column(Text(), nullable=False, default=".*", server_default=".*")
    precomputed = Column(Boolean(), nullable=False, default=False, server_default="false")


class UserAccount(create_time_mixin(created_at=True), Base):
    """User<>account many-to-many relations."""

    __tablename__ = "user_accounts"

    user_id = Column(String(256), primary_key=True)
    account_id = Column(Integer(), ForeignKey("accounts.id", name="fk_user_account"),
                        nullable=False, primary_key=True)
    is_admin = Column(Boolean(), nullable=False, default=False)


class Account(create_time_mixin(created_at=True), Base):
    """Group of users, some are admins and some are regular."""

    __tablename__ = "accounts"
    __table_args__ = {"sqlite_autoincrement": True}
    missing_secret = "0" * 8

    id = Column(Integer(), primary_key=True)
    secret_salt = Column(Integer(), nullable=False)
    secret = Column(String(8), nullable=False)
    expires_at = Column(TIMESTAMP(timezone=True), nullable=False)


class Team(create_time_mixin(created_at=True, updated_at=True),
           create_collection_mixin("members", with_checksum=False), Base):
    """Group of users part of the same team."""

    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("owner_id", "name", name="uc_owner_name"),
                      CheckConstraint("id != parent_id", name="cc_parent_self_reference"),
                      {"sqlite_autoincrement": True})
    BOTS = "Bots"  # the name of the special team which contains bots

    id = Column(Integer(), primary_key=True)
    owner_id = Column(Integer(), ForeignKey("accounts.id", name="fk_reposet_owner"),
                      nullable=False)
    parent_id = Column(Integer(), ForeignKey("teams.id", name="fk_team_parent"))


class AccountGitHubAccount(Base):
    """Mapping API account -> metadata account, one-to-many."""

    __tablename__ = "account_github_accounts"

    id = Column(BigInteger(), primary_key=True, autoincrement=False)
    account_id = Column(Integer(), ForeignKey("accounts.id", name="fk_installation_id_owner"),
                        nullable=False)


class AccountJiraInstallation(Base):
    """Mapping account -> installation_id, one-to-many."""

    __tablename__ = "account_jira_installations"

    id = Column(BigInteger(), primary_key=True, autoincrement=False)
    account_id = Column(Integer(), ForeignKey("accounts.id", name="fk_installation_id_owner2"),
                        nullable=False)


class Invitation(create_time_mixin(created_at=True), Base):
    """Account invitations, each maps to a URL that invitees should click."""

    __tablename__ = "invitations"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer(), primary_key=True)
    salt = Column(Integer(), nullable=False)
    account_id = Column(Integer(), ForeignKey(
        "accounts.id", name="fk_invitation_account"), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    accepted = Column(Integer(), nullable=False, default=0)
    created_by = Column(String(256))


class God(create_time_mixin(updated_at=True), Base):
    """Secret user mappings for chosen ones."""

    __tablename__ = "gods"

    user_id = Column(String(256), primary_key=True)
    mapped_id = Column(String(256), nullable=True)


class ReleaseSetting(create_time_mixin(updated_at=True), Base):
    """Release matching rules per repo."""

    __tablename__ = "release_settings"

    repository = Column(String(512), primary_key=True)
    account_id = Column(Integer(), ForeignKey("accounts.id", name="fk_release_settings_account"),
                        primary_key=True)
    branches = Column(String(1024))
    tags = Column(String(1024))
    match = Column(SmallInteger())


class FeatureComponent(enum.IntEnum):
    """Athenian stack parts: the frontend, the backend, etc."""

    webapp = 1
    server = 2


class Feature(create_time_mixin(updated_at=True), Base):
    """Product features."""

    __tablename__ = "features"
    __table_args__ = (UniqueConstraint("name", "component", name="uc_feature_name_component"),
                      {"sqlite_autoincrement": True})

    USER_ORG_MEMBERSHIP_CHECK = "user_org_membership_check"
    QUANTILE_STRIDE = "quantile_stride"

    id = Column(Integer(), primary_key=True)
    name = Column(String(128), nullable=False)
    component = Column(Enum(FeatureComponent), nullable=False)
    enabled = Column(Boolean(), nullable=False, default=False, server_default="false")
    default_parameters = Column(JSON(), nullable=False, default={}, server_default="{}")


class AccountFeature(create_time_mixin(updated_at=True), Base):
    """Product features -> accounts many-to-many mapping."""

    __tablename__ = "account_features"

    account_id = Column(Integer(), ForeignKey(
        "accounts.id", name="fk_account_features_account"), primary_key=True)
    feature_id = Column(Integer(), ForeignKey(
        "features.id", name="fk_account_features_feature"), primary_key=True)
    enabled = Column(Boolean(), nullable=False, default=False, server_default="false")
    parameters = Column(JSON())


class UserToken(create_time_mixin(updated_at=True), Base):
    """Personal Access Tokens of the accounts."""

    __tablename__ = "user_tokens"
    __table_args__ = (UniqueConstraint("name", "user_id", "account_id", name="uc_token_name"),
                      ForeignKeyConstraint(("account_id", "user_id"),
                                           ("user_accounts.account_id", "user_accounts.user_id"),
                                           name="fk_account_tokens_user"),
                      {"sqlite_autoincrement": True})

    id = Column(BigInteger().with_variant(Integer(), "sqlite"), primary_key=True)
    account_id = Column(Integer(), nullable=False)
    user_id = Column(String(256), nullable=False)
    name = Column(String(256), nullable=False)
    last_used_at = Column(TIMESTAMP(timezone=True), nullable=False,
                          default=lambda: datetime.now(timezone.utc), server_default=func.now())


class JIRAProjectSetting(create_time_mixin(updated_at=True), Base):
    """JIRA projects enabled/disabled."""

    __tablename__ = "jira_projects"

    account_id = Column(Integer(), ForeignKey("accounts.id", name="fk_jira_projects_account"),
                        primary_key=True)
    key = Column(Text(), primary_key=True)
    enabled = Column(Boolean(), nullable=False)


class MappedJIRAIdentity(create_time_mixin(created_at=True, updated_at=True), Base):
    """JIRA identity mapping."""

    __tablename__ = "jira_identity_mapping"

    account_id = Column(Integer(),
                        ForeignKey("accounts.id", name="fk_jira_identity_mapping_account"),
                        primary_key=True)
    github_user_id = Column(BigInteger(), primary_key=True)
    jira_user_id = Column(Text())
    confidence = Column(Float())
