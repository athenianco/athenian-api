from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    BigInteger,
    Boolean,
    Column,
    ForeignKeyConstraint,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects import sqlite
from sqlalchemy.dialects.postgresql import JSONB

from athenian.api.models import create_base


class ShardedByAccount:
    """All the tables contain `account_id` primary key."""

    account_id = Column(Integer(), primary_key=True)  # state DB account ID


Base = create_base(ShardedByAccount)
Base.__table_args__ = {"schema": "athenian"}


def create_time_mixin(created_at: bool = False, updated_at: bool = False) -> type:
    """Create the mixin accorinding to the required columns."""
    created_at_ = created_at
    updated_at_ = updated_at

    class TimeMixin:
        if created_at_:
            created_at = Column(
                TIMESTAMP(timezone=True),
                nullable=False,
                default=lambda: datetime.now(timezone.utc),
                server_default=func.now(),
            )
        if updated_at_:
            updated_at = Column(
                TIMESTAMP(timezone=True),
                nullable=False,
                default=lambda: datetime.now(timezone.utc),
                server_default=func.now(),
                onupdate=lambda ctx: datetime.now(timezone.utc),
            )

    return TimeMixin


class ReleaseNotification(create_time_mixin(created_at=True, updated_at=True), Base):
    """Client's pushed release notifications."""

    __tablename__ = "release_notifications"

    repository_node_id = Column(BigInteger(), primary_key=True)
    # registered commit hash 7 or 40 chars
    commit_hash_prefix = Column(Text(), primary_key=True)
    # de-referenced commit hash in metadata DB
    resolved_commit_hash = Column(Text(), info={"dtype": "S40"})
    resolved_commit_node_id = Column(BigInteger())  # de-referenced commit node ID in metadata DB
    resolved_at = Column(TIMESTAMP(timezone=True), nullable=True)
    name = Column(Text())
    author_node_id = Column(BigInteger(), info={"reset_nulls": True})
    url = Column(Text())
    published_at = Column(TIMESTAMP(timezone=True), nullable=False)
    cloned = Column(Boolean(), nullable=False, default=False, server_default="false")


class DeploymentNotification(create_time_mixin(created_at=True, updated_at=True), Base):
    """Client's pushed deployment notifications."""

    __tablename__ = "deployment_notifications"
    CONCLUSION_SUCCESS = b"SUCCESS"
    CONCLUSION_FAILURE = b"FAILURE"
    CONCLUSION_CANCELLED = b"CANCELLED"

    name = Column(Text(), primary_key=True, nullable=False)
    conclusion = Column(Text(), nullable=False, info={"dtype": "S9"})  # CONCLUSION_*
    environment = Column(Text(), nullable=False)  # production, staging, etc.; nothing's enforced
    url = Column(Text())
    started_at = Column(TIMESTAMP(timezone=True), nullable=False)
    finished_at = Column(TIMESTAMP(timezone=True), nullable=False)


class DeployedLabel(Base):
    """Key-values mapped to DeploymentNotification."""

    __tablename__ = "deployed_labels"
    _table_args__ = [
        ForeignKeyConstraint(
            ("account_id", "deployment_name"),
            (DeploymentNotification.account_id, DeploymentNotification.name),
            name="fk_deployed_labels_deployment",
        ),
    ]

    deployment_name = Column(Text(), primary_key=True)
    key = Column(Text(), primary_key=True)
    value = Column(JSONB().with_variant(JSON(), sqlite.dialect.name))


class DeployedComponent(create_time_mixin(created_at=True, updated_at=False), Base):
    """Deployed (repository, Git reference) pairs."""

    __tablename__ = "deployed_components"
    _table_args__ = [
        ForeignKeyConstraint(
            ("account_id", "deployment_name"),
            (DeploymentNotification.account_id, DeploymentNotification.name),
            name="fk_deployed_components_deployment",
        ),
    ]

    deployment_name = Column(Text(), primary_key=True)
    repository_node_id = Column(BigInteger(), primary_key=True)
    repository_full_name = "repository_full_name"
    reference = Column(Text(), primary_key=True)  # tag, commit hash 7-char prefix or full
    # de-referenced commit node ID in metadata DB
    resolved_commit_node_id = Column(BigInteger(), info={"reset_nulls": True})
    resolved_at = Column(TIMESTAMP(timezone=True), nullable=True)
