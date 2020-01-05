from datetime import datetime

from sqlalchemy import Column, Integer, JSON, String, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class RepositorySet(Base):
    """A group of repositories identified by an integer."""

    __tablename__ = "repository_sets"

    def count_items(ctx):
        """Return the number of repositories in a set."""
        return len(ctx.current_parameters["items"])

    id = Column("id", Integer(), primary_key=True)
    owner = Column("owner", String(256), nullable=False)
    updated_at = Column("updated_at", TIMESTAMP(), nullable=False, default=datetime.utcnow,
                        onupdate=datetime.utcnow)
    created_at = Column("created_at", TIMESTAMP(), nullable=False, default=datetime.utcnow)
    updates_count = Column("updates_count", Integer(), nullable=False, default=1,
                           onupdate=lambda ctx: ctx.current_parameters["updates_count"] + 1)
    items = Column("items", JSON())
    items_count = Column("items_count", Integer(), nullable=False, default=count_items,
                         onupdate=count_items)

    count_items = staticmethod(count_items)
