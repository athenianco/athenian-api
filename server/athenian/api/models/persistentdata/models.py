from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, Column, ForeignKeyConstraint, func, Integer, Text, \
    TIMESTAMP

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
            created_at = Column(TIMESTAMP(timezone=True), nullable=False,
                                default=lambda: datetime.now(timezone.utc),
                                server_default=func.now())
        if updated_at_:
            updated_at = Column(TIMESTAMP(timezone=True), nullable=False,
                                default=lambda: datetime.now(timezone.utc),
                                server_default=func.now(),
                                onupdate=lambda ctx: datetime.now(timezone.utc))

    return TimeMixin


class ReleaseNotification(create_time_mixin(created_at=True, updated_at=True), Base):
    """Client's pushed release notifications."""

    __tablename__ = "release_notifications"

    repository_node_id = Column(Text(), primary_key=True)
    commit_hash_prefix = Column(Text(), primary_key=True)  # registered commit hash 7 or 40 chars
    resolved_commit_hash = Column(Text())  # de-referenced commit hash in metadata DB
    resolved_commit_node_id = Column(Text())  # de-referenced commit node ID in metadata DB
    name = Column(Text())
    author_node_id = Column(Text())
    url = Column(Text())
    published_at = Column(TIMESTAMP(timezone=True), nullable=False)
    cloned = Column(Boolean(), nullable=False, default=False, server_default="false")


class DeploymentNotification(create_time_mixin(created_at=True, updated_at=True), Base):
    """Client's pushed deployment notifications."""

    __tablename__ = "deployment_notifications"

    id = Column(BigInteger(), primary_key=True)  # deployment ID
    conclusion = Column(Text())  # SUCCESS, FAILURE, CANCELLED
    environment = Column(Text(), nullable=False)  # production, staging, etc.; nothing's enforced
    name = Column(Text(), nullable=False)  # arbitrary but must exist
    url = Column(Text())
    started_at = Column(TIMESTAMP(timezone=True), nullable=False)
    finished_at = Column(TIMESTAMP(timezone=True))  # is nullable to support 2-step notifications


class DeployedLabel(Base):
    """Key-values mapped to DeploymentNotification."""

    __tablename__ = "deployed_labels"
    _table_args__ = [
        ForeignKeyConstraint(
            ("account_id", "deployment_id"),
            (DeploymentNotification.account_id, DeploymentNotification.id),
            name="fk_deployed_labels_deployment",
        ),
    ]

    deployment_id = Column("deployment_id", BigInteger(), primary_key=True)
    key = Column(Text(), primary_key=True)
    value = Column(Text())


class DeployedComponent(create_time_mixin(created_at=True, updated_at=False), Base):
    """Deployed (repository, Git reference) pairs."""

    __tablename__ = "deployed_components"
    _table_args__ = [
        ForeignKeyConstraint(
            ("account_id", "deployment_id"),
            (DeploymentNotification.account_id, DeploymentNotification.id),
            name="fk_deployed_components_deployment",
        ),
    ]

    deployment_id = Column("deployment_id", BigInteger(), primary_key=True)
    repository_node_id = Column(Text(), primary_key=True)
    reference = Column(Text(), primary_key=True)  # tag, commit hash 7-char prefix or full
    resolved_commit_node_id = Column(Text())  # de-referenced commit node ID in metadata DB
