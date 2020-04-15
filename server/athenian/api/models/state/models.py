import ctypes
from datetime import datetime, timezone
import json

from sqlalchemy import BigInteger, Boolean, Column, ForeignKey, func, Integer, JSON, \
    SmallInteger, String, TIMESTAMP, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
import xxhash


# The following two classes compensate the absent ORM layer in databases.Database.


def always_unequal(coltype):
    """Mark certain attributes to be always included in the execution context."""
    coltype.compare_values = lambda _1, _2: False
    return coltype


class Refresheable:
    """Mixin to invoke default() and onupdate() on all the columns."""

    class Context:
        """Pretend to be a fully-featured SQLAlchemy execution context."""

        def __init__(self, parameters: dict):
            """init"""
            self.current_parameters = parameters

        def get_current_parameters(self):
            """Pretend to be a fully-featured context."""
            return self.current_parameters

    def create_defaults(self) -> "Refresheable":
        """Call default() on all the columns."""
        ctx = self.Context(self.__dict__)
        for k, v in self.__table__.columns.items():
            if getattr(self, k, None) is None and v.default is not None:
                arg = v.default.arg
                if callable(arg):
                    arg = arg(ctx)
                setattr(self, k, arg)
        return self

    def refresh(self) -> "Refresheable":
        """Call onupdate() on all the columns."""
        ctx = self.Context(self.__dict__)
        for k, v in self.__table__.columns.items():
            if v.onupdate is not None:
                setattr(self, k, v.onupdate.arg(ctx))
        return self

    def touch(self, exclude=tuple()) -> "Refresheable":
        """Enable full onupdate() in the next session.flush()."""
        for k in self.__table__.columns.keys():
            if k not in exclude:
                setattr(self, k, getattr(self, k))
        return self


class Explodeable:
    """Convert the model to a dict."""

    def explode(self, with_primary_keys=False):
        """Return a dict of the model data attributes."""
        return {k: getattr(self, k) for k, v in self.__table__.columns.items()
                if not v.primary_key or with_primary_keys}


Base = declarative_base(cls=(Refresheable, Explodeable))


class RepositorySet(Base):
    """A group of repositories identified by an integer."""

    __tablename__ = "repository_sets"
    __table_args__ = (UniqueConstraint("owner", "items_checksum", name="uc_owner_items"),
                      {"sqlite_autoincrement": True})

    def count_items(ctx):
        """Return the number of repositories in a set."""
        return len(ctx.get_current_parameters()["items"])

    def calc_items_checksum(ctx):
        """Calculate the checksum of the reposet items."""
        return ctypes.c_longlong(xxhash.xxh64_intdigest(json.dumps(
            ctx.get_current_parameters()["items"]))).value

    id = Column(Integer(), primary_key=True)
    owner = Column(Integer(), ForeignKey("accounts.id", name="fk_reposet_owner"), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc),
                        server_default=func.now(),
                        onupdate=lambda ctx: datetime.now(timezone.utc))
    created_at = Column(TIMESTAMP(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc),
                        server_default=func.now())
    updates_count = Column(always_unequal(Integer()), nullable=False, default=1,
                           onupdate=lambda ctx: ctx.get_current_parameters()["updates_count"] + 1)
    items = Column(always_unequal(JSON()), nullable=False)
    items_count = Column(Integer(), nullable=False, default=count_items, onupdate=count_items)
    items_checksum = Column(always_unequal(BigInteger()), nullable=False,
                            default=calc_items_checksum, onupdate=calc_items_checksum)

    count_items = staticmethod(count_items)
    calc_items_checksum = staticmethod(calc_items_checksum)


class UserAccount(Base):
    """User<>account many-to-many relations."""

    __tablename__ = "user_accounts"

    user_id = Column(String(256), primary_key=True)
    account_id = Column(Integer(), ForeignKey("accounts.id", name="fk_user_account"),
                        nullable=False, primary_key=True)
    is_admin = Column(Boolean(), nullable=False, default=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc),
                        server_default=func.now())


class Account(Base):
    """Group of users, some are admins and some are regular."""

    __tablename__ = "accounts"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer(), primary_key=True)
    installation_id = Column(BigInteger(), unique=True, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc),
                        server_default=func.now())


class Invitation(Base):
    """Account invitations, each maps to a URL that invitees should click."""

    __tablename__ = "invitations"
    __table_args__ = {"sqlite_autoincrement": True}

    id = Column(Integer(), primary_key=True)
    salt = Column(Integer(), nullable=False)
    account_id = Column(Integer(), ForeignKey(
        "accounts.id", name="fk_invitation_account"), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    accepted = Column(Integer(), nullable=False, default=0)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc),
                        server_default=func.now())
    created_by = Column(String(256))


class God(Base):
    """Secret user mappings for chosen ones."""

    __tablename__ = "gods"

    user_id = Column(String(256), primary_key=True)
    mapped_id = Column(String(256), nullable=True)
    updated_at = Column(always_unequal(TIMESTAMP(timezone=True)), nullable=False,
                        default=lambda: datetime.now(timezone.utc),
                        server_default=func.now(),
                        onupdate=lambda ctx: datetime.now(timezone.utc))


class ReleaseSetting(Base):
    """Release matching rules per repo."""

    __tablename__ = "release_settings"

    repository = Column(String(512), primary_key=True)
    account_id = Column(Integer(), primary_key=True)
    branches = Column(String(1024))
    tags = Column(String(1024))
    match = Column(SmallInteger())
    updated_at = Column(always_unequal(TIMESTAMP(timezone=True)), nullable=False,
                        default=lambda: datetime.now(timezone.utc),
                        server_default=func.now(),
                        onupdate=lambda ctx: datetime.now(timezone.utc))
