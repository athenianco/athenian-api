from datetime import datetime, timezone

from sqlalchemy import Column, func, Integer, Text, TIMESTAMP

from athenian.api.models import create_base


Base = create_base()
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

    account_id = Column(Integer(), primary_key=True)  # state DB account ID
    repository_node_id = Column(Text(), primary_key=True)
    commit_hash_prefix = Column(Text(), primary_key=True)  # registered commit hash 7 or 40 chars
    resolved_commit_hash = Column(Text())  # de-referenced commit hash in metadata DB
    resolved_commit_node_id = Column(Text())  # de-referenced commit node ID in metadata DB
    name = Column(Text())
    author = Column(Text())
    url = Column(Text())
    published_at = Column(TIMESTAMP(timezone=True), nullable=False)
