"""The following two classes compensate the absent ORM layer in databases.Database."""
import logging
import os
import subprocess
from typing import Union

from alembic import script
from alembic.migration import MigrationContext
import jinja2
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.sql.expression import all_, any_

from athenian.api import slogging


def always_unequal(coltype):
    """Mark certain attributes to be always included in the execution context."""
    coltype.compare_values = lambda _1, _2: False
    return coltype


class Refreshable:
    """Mixin to invoke default() and onupdate() on all the columns."""

    class Context:
        """Pretend to be a fully-featured SQLAlchemy execution context."""

        def __init__(self, parameters: dict):
            """init"""
            self.current_parameters = parameters

        def get_current_parameters(self):
            """Pretend to be a fully-featured context."""
            return self.current_parameters

    def create_defaults(self) -> "Refreshable":
        """Call default() on all the columns."""
        ctx = self.Context(self.__dict__)
        for k, v in self.__table__.columns.items():
            if getattr(self, k, None) is None and v.default is not None:
                arg = v.default.arg
                if callable(arg):
                    arg = arg(ctx)
                setattr(self, k, arg)
        return self

    def refresh(self) -> "Refreshable":
        """Call onupdate() on all the columns."""
        ctx = self.Context(self.__dict__)
        for k, v in self.__table__.columns.items():
            if v.onupdate is not None:
                setattr(self, k, v.onupdate.arg(ctx))
        return self

    def touch(self, exclude=tuple()) -> "Refreshable":
        """Enable full onupdate() in the next session.flush()."""
        for k in self.__table__.columns.keys():
            if k not in exclude:
                setattr(self, k, getattr(self, k))
        return self


class Explodable:
    """Convert the model to a dict."""

    def explode(self, with_primary_keys=False):
        """Return a dict of the model data attributes."""
        return {k: getattr(self, k) for k, v in self.__table__.columns.items()
                if not v.primary_key or with_primary_keys}


BaseType = Union[DeclarativeMeta, Refreshable, Explodable]


def create_base() -> BaseType:
    """Create the declarative base class type."""
    return declarative_base(cls=(Refreshable, Explodable))


class AutoInAnyColumn(Column):
    """Column that automatically converts `IN` and `NOT IN` to `ANY` and `!= ALL` repsecitvely"""

    # 32767 is the maximum value for postgres:
    #   - https://sentry.io/organizations/athenianco/issues/1785993425/
    MAX_ALLOWED_QUERY_ARGUMENTS = 32767
    # Anyway we cannot use exactly the upper limit because the length of the values used for
    # `IN` and `NOT IN` could be near the the maximum allowed, but it often comes with other
    # clauses that could make the total number of arguments bindings overflow the limit.
    # This switch also affects performance. `IN` is better for smaller values and `ANY` for larger
    # ones:
    #   - https://blog.jooq.org/2017/03/30/sql-in-predicate-with-in-list-or-with-array-which-is-faster/ # noqa: E501
    IN_AUTO_ANY_THRESHOLD = 100  # This specific value doesn't have any specific meaning

    assert IN_AUTO_ANY_THRESHOLD < MAX_ALLOWED_QUERY_ARGUMENTS

    log = logging.getLogger("%s.AutoInAnyColumn" % __name__)

    def in_(self, other):
        """Change the `in` operator into an `any` if too many values"""
        if len(other) > self.IN_AUTO_ANY_THRESHOLD:
            self.log.info("automatically switching `IN` clause to `ANY`")
            # `in_` works if you pass a `dict` for example, but `any_` does not
            return self == any_(list(other))

        return super(AutoInAnyColumn, self).in_(other)

    def notin_(self, other):
        """Change the `notin` operator into a `!= all` if too many values"""
        if len(other) > self.IN_AUTO_ANY_THRESHOLD:
            self.log.info("automatically switching `NOT IN` clause to `!= ALL`")
            # `notin_` works if you pass a `dict` for example, but `all_` does not
            return self != all_(list(other))

        return super(AutoInAnyColumn, self).notin_(other)


slogging.trailing_dot_exceptions.add("alembic.runtime.migration")


class DBSchemaMismatchError(Exception):
    """Error raised if the DB schema versions do not match."""


def check_schema_version(name: str, conn_str: str, log: logging.Logger) -> None:
    """Raise DBSchemaVersionMismatchError if the real (connected) DB schema version \
    does not match the required (declared in the code) version."""
    directory = script.ScriptDirectory(os.path.join(os.path.dirname(__file__), name))
    engine = create_engine(conn_str.split("?", 1)[0])
    with engine.begin() as conn:
        context = MigrationContext.configure(conn)
        real_rev = context.get_current_revision()
    req_rev = directory.get_current_head()
    if real_rev != req_rev:
        raise DBSchemaMismatchError(
            "%s version: required: %s connected: %s" % (conn_str, req_rev, real_rev))
    log.info("%s DB schema version: %s", name, real_rev)


def check_collation(conn_str: str) -> None:
    """Force the PostgreSQL collation to be "C"."""
    engine = create_engine(conn_str.split("?", 1)[0])
    if engine.dialect.name not in ("postgres", "postgresql"):
        return
    collation = engine.scalar(
        "select datcollate from pg_database where datname='%s';" % engine.url.database)
    if collation.lower() != "c.utf-8":
        raise DBSchemaMismatchError(
            "%s collation: required: C.UTF-8 connected: %s" % (conn_str, collation))


def migrate(name: str, url=None, exec=True):
    """
    Migrate a database with alembic.

    This script creates all the tables if they don't exist and migrates the DB to the most
    recent version. It is to simplify the deployment.

    As a bonus, you obtain a functional Alembic INI config for any `alembic` commands.
    """
    root = os.path.join(os.path.dirname(__file__), name)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(root))
    t = env.get_template("alembic.ini.jinja2")
    path = os.path.relpath(root)
    with open("alembic.ini", "w") as fout:
        fout.write(t.render(url=url, path=path))
    args = ["alembic", "alembic", "upgrade", "head"]
    if os.getenv("OFFLINE"):
        args.append("--sql")
    if exec:
        os.execlp(*args)
    else:
        subprocess.run(" ".join(args[1:]), check=True, shell=True)


def hack_sqlite_arrays():
    """Hack SQLite compiler to handle ARRAY fields."""
    SQLiteTypeCompiler.visit_ARRAY = lambda self, type_, **kw: "JSON"


def hack_sqlite_hstore():
    """Hack SQLite compiler to handle HSTORE fields."""
    SQLiteTypeCompiler.visit_HSTORE = lambda self, type_, **kw: "JSON"
