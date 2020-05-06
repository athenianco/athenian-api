"""The following two classes compensate the absent ORM layer in databases.Database."""
import logging
import os
import subprocess
from typing import Union

from alembic import script
from alembic.migration import MigrationContext
import jinja2
from sqlalchemy import create_engine
from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta

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


slogging.trailing_dot_exceptions.add("alembic.runtime.migration")


class DBSchemaVersionMismatchError(Exception):
    """Error raised if the DB schema versions do not match."""


def check_schema_version(name: str, conn_str: str, log: logging.Logger) -> None:
    """Raise DBSchemaVersionMismatchError if the real (connected) DB schema version \
    does not match the required (declared in the code) version."""
    directory = script.ScriptDirectory(os.path.join(os.path.dirname(__file__), name))
    engine = create_engine(conn_str)
    with engine.begin() as conn:
        context = MigrationContext.configure(conn)
        real_rev = context.get_current_revision()
    req_rev = directory.get_current_head()
    if real_rev != req_rev:
        raise DBSchemaVersionMismatchError(
            "%s: required: %s connected: %s" % (conn_str, req_rev, real_rev))
    log.info("%s DB schema version: %s", name, real_rev)


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
