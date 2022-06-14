from datetime import datetime

import pytz
from sqlalchemy import ARRAY, TIMESTAMP, BigInteger, Column, Integer, Text, func

from athenian.api.models import create_base

Base = create_base()


class PullRequestStatus(Base):
    """Table with the aggregation metrics status."""

    __tablename__ = "pull_requests_status"

    account = Column(Integer(), nullable=False)
    repository_node_id = Column(Text(), nullable=False)
    repository_full_name = Column(Text(), nullable=False)
    node_id = Column(Text(), primary_key=True)
    number = Column(BigInteger(), nullable=False)
    last_aggregation_timestamp = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc),
        server_default=func.to_timestamp(0),
    )


class DimensionColumn(Column):
    """Column representing an aggregate dimension."""

    def __init__(self, *args, **kwargs):
        """Create a dimension column."""
        super().__init__(*args, nullable=False, **kwargs)


class MetricColumn(Column):
    """Column representing an aggregate metric."""

    def __init__(self, *args, **kwargs):
        """Create a metric column."""
        super().__init__(BigInteger(), *args, nullable=False, default=0, **kwargs)


class PullRequestEvent(Base):
    """Table with the aggregate metrics as events."""

    __tablename__ = "pull_requests_events"

    # dimensions
    repository_full_name = DimensionColumn(Text(), primary_key=True)
    number = DimensionColumn(BigInteger(), primary_key=True)
    event_type = DimensionColumn(Text(), primary_key=True)
    release_setting = DimensionColumn(Text(), primary_key=True)

    account = DimensionColumn(Integer(), index=True)
    init = DimensionColumn(TIMESTAMP(timezone=True), index=True)
    timestamp = DimensionColumn(TIMESTAMP(timezone=True), index=True)
    event_owners = DimensionColumn(ARRAY(Text()), index=True)

    # metrics
    wip_time = MetricColumn()
    wip_count = MetricColumn()

    review_time = MetricColumn()
    review_count = MetricColumn()

    merging_time = MetricColumn()
    merging_count = MetricColumn()

    release_time = MetricColumn()
    release_count = MetricColumn()

    lead_time = MetricColumn()
    lead_count = MetricColumn()

    wait_first_review_time = MetricColumn()
    wait_first_review_count = MetricColumn()

    opened = MetricColumn()
    merged = MetricColumn()
    rejected = MetricColumn()
    closed = MetricColumn()
    released = MetricColumn()

    size_added = MetricColumn()
    size_removed = MetricColumn()
