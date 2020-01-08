from datetime import datetime

from sqlalchemy import Column, Integer, JSON, String, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base


# The following two classes compensate the absent ORM layer in databases.Database.

class Refresheable:
    """Mixin to invoke default() and onupdate() on all the columns."""

    class Context:
        """Pretend to be a fully-featured SQLAlchemy execution context."""

        def __init__(self, parameters: dict):
            """init"""
            self.current_parameters = parameters

    def create_defaults(self):
        """Call default() on all the columns."""
        ctx = self.Context(self.__dict__)
        for k, v in self.__table__.columns.items():
            if getattr(self, k, None) is None and v.default is not None:
                arg = v.default.arg
                if callable(arg):
                    arg = arg(ctx)
                setattr(self, k, arg)

    def refresh(self):
        """Call onupdate() on all the columns."""
        ctx = self.Context(self.__dict__)
        for k, v in self.__table__.columns.items():
            if v.onupdate is not None:
                setattr(self, k, v.onupdate.arg(ctx))


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

    def count_items(ctx):
        """Return the number of repositories in a set."""
        return len(ctx.current_parameters["items"])

    id = Column("id", Integer(), primary_key=True)
    owner = Column("owner", String(256), nullable=False)
    updated_at = Column("updated_at", TIMESTAMP(), nullable=False, default=datetime.utcnow,
                        onupdate=lambda ctx: datetime.utcnow())
    created_at = Column("created_at", TIMESTAMP(), nullable=False, default=datetime.utcnow)
    updates_count = Column("updates_count", Integer(), nullable=False, default=1,
                           onupdate=lambda ctx: ctx.current_parameters["updates_count"] + 1)
    items = Column("items", JSON())
    items_count = Column("items_count", Integer(), nullable=False, default=count_items,
                         onupdate=count_items)

    count_items = staticmethod(count_items)
