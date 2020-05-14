import ctypes
from datetime import datetime, timezone
import json

from sqlalchemy import BigInteger, Boolean, Column, ForeignKey, func, Integer, JSON, \
    SmallInteger, String, TIMESTAMP, UniqueConstraint
import xxhash

from athenian.api.models import always_unequal, create_base


Base = create_base()


def create_collection_mixin(name: str):
    """Create the collections mixin according to the required column name."""
    def count_items(ctx):
        """Return the number of items in the collection."""
        return len(ctx.get_current_parameters()[name])

    def calc_items_checksum(ctx):
        """Calculate the checksum of the items in the collection."""
        return ctypes.c_longlong(xxhash.xxh64_intdigest(json.dumps(
            ctx.get_current_parameters()[name]))).value

    cols = {
        "count_items": staticmethod(count_items),
        "calc_items_checksum": staticmethod(calc_items_checksum),
        name: Column(name, always_unequal(JSON()), nullable=False),
        f"{name}_count": Column(Integer(), nullable=False, default=count_items,
                                onupdate=count_items),
        f"{name}_checksum": Column(always_unequal(BigInteger()), nullable=False,
                                   default=calc_items_checksum, onupdate=calc_items_checksum),
    }

    return type("CollectionMixin", (), cols)


def create_time_mixin(created_at: bool = False, updated_at: bool = False):
    """Create the mixin accorinding to the required columns."""
    cols = {}
    if created_at:
        cols["created_at"] = Column(TIMESTAMP(timezone=True), nullable=False,
                                    default=lambda: datetime.now(timezone.utc),
                                    server_default=func.now())
    if updated_at:
        cols["updated_at"] = Column(TIMESTAMP(timezone=True), nullable=False,
                                    default=lambda: datetime.now(timezone.utc),
                                    server_default=func.now(),
                                    onupdate=lambda ctx: datetime.now(timezone.utc))
    return type("TimeMixin", (), cols)


class RepositorySet(create_time_mixin(created_at=True, updated_at=True),
                    create_collection_mixin("items"), Base):
    """A group of repositories identified by an integer."""

    __tablename__ = "repository_sets"
    __table_args__ = (UniqueConstraint("owner", "items_checksum", name="uc_owner_items"),
                      {"sqlite_autoincrement": True})

    id = Column(Integer(), primary_key=True)
    owner = Column(Integer(), ForeignKey("accounts.id", name="fk_reposet_owner"), nullable=False)
    updates_count = Column(always_unequal(Integer()), nullable=False, default=1,
                           onupdate=lambda ctx: ctx.get_current_parameters()["updates_count"] + 1)


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

    id = Column(Integer(), primary_key=True)


class Team(create_time_mixin(created_at=True, updated_at=True),
           create_collection_mixin("members"), Base):
    """Group of users part of the same team."""

    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("owner", "members_checksum", name="uc_owner_members"),
                      UniqueConstraint("owner", "name", name="uc_owner_name"),
                      {"sqlite_autoincrement": True})

    id = Column(Integer(), primary_key=True)
    name = Column(String(256), nullable=False)
    owner = Column(Integer(), ForeignKey("accounts.id", name="fk_reposet_owner"), nullable=False)


class Installation(Base):
    """Mapping account -> installation_id, one-to-many."""

    __tablename__ = "installations"

    id = Column(BigInteger(), primary_key=True, autoincrement=False)
    account_id = Column(Integer(), ForeignKey("accounts.id", name="fk_installation_id_owner"),
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
    account_id = Column(Integer(), primary_key=True)
    branches = Column(String(1024))
    tags = Column(String(1024))
    match = Column(SmallInteger())
