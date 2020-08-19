"""The following two classes compensate the absent ORM layer in databases.Database."""
import logging
import os
import subprocess
import sys
from typing import Union

from alembic import script
from alembic.migration import MigrationContext
import jinja2
from sqlalchemy import any_, create_engine
from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.sql import operators
from sqlalchemy.sql.compiler import OPERATORS
from sqlalchemy.sql.elements import BinaryExpression, ClauseList, Grouping, UnaryExpression
from sqlalchemy.sql.operators import ColumnOperators, custom_op, in_op, notin_op

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


class values(UnaryExpression):
    """PostgreSQL "VALUES (...), (...), ..."."""

    def __init__(self, element):
        """Plug in UnaryExpression to provide the required syntax: there is no outer grouping, \
        but each element is grouped separately."""
        if isinstance(element, Grouping):
            element = element.element
        contents = ClauseList(*element.clauses, group_contents=False, group=False)
        for i, clause in enumerate(contents.clauses):
            contents.clauses[i] = Grouping(clause)
        super().__init__(element=contents, operator=custom_op("VALUES "))


@compiles(BinaryExpression)
def compile_binary(binary, compiler, override_operator=None, **kw):
    """
    If there are more than 10 elements in the `IN` set, inline them to avoid hitting the limit of \
    the number of query arguments in Postgres (1<<15).
    """  # noqa: D200
    operator = override_operator or binary.operator

    try:
        right_len = len(binary.right)
    except TypeError:
        if isinstance(binary.right, Grouping):
            right_len = len(binary.right.element.clauses)
        else:
            right_len = 0
    if (operator is in_op or operator is notin_op) and right_len >= 10:
        left = compiler.process(binary.left, **kw)
        kw["literal_binds"] = True
        use_any = getattr(binary, "any_values", False) and \
            compiler.dialect.name in ("postgres", "postgresql")
        negate = use_any and operator is notin_op
        if use_any:
            # ANY(VALUES ...) seems to be performing the best among these three:
            # 1. IN (...)
            # 2. IN(ARRAY[...])
            # 3. IN(VALUES ...)
            right = any_(values(binary.right))
            operator = operators.eq
        else:
            right = binary.right
        right = compiler.process(right, **kw)
        sql = left + OPERATORS[operator] + right
        if negate:
            sql = "NOT (%s)" % sql
        return sql

    return compiler.visit_binary(binary, override_operator=override_operator, **kw)


def in_any_values(self: ColumnOperators, other):
    """Implement = ANY(VALUES (...), (...), ...) PostgreSQL operator."""
    expr = self.in_(other)
    expr.any_values = True
    return expr


def notin_any_values(self: ColumnOperators, other):
    """Implement NOT = ANY(VALUES (...), (...), ...) PostgreSQL operator."""
    expr = self.notin_(other)
    expr.any_values = True
    return expr


ColumnOperators.in_any_values = in_any_values
ColumnOperators.notin_any_values = notin_any_values


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
    args = [sys.executable, sys.executable, "-m", "athenian.api.sentry_wrapper",
            "alembic.config", "upgrade", "head"]
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
