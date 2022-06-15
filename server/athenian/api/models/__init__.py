from importlib import import_module
import logging
import os
import subprocess
import sys
from typing import Any, Iterable, Sequence

from alembic import script
from alembic.migration import MigrationContext
from flogging import flogging
from mako.template import Template
import numpy as np
from sqlalchemy import any_, create_engine, text
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import Values, operators
from sqlalchemy.sql.compiler import OPERATORS
from sqlalchemy.sql.elements import BinaryExpression, BindParameter, Grouping
from sqlalchemy.sql.operators import ColumnOperators, in_op, not_in_op

from athenian.api.models.sql_builders import in_any_values_inline, in_inline
from athenian.precomputer.db import always_unequal, create_base  # noqa: F401


class TupleWrapper(Sequence):
    """Pretend to be a sequence, wrap each element in a tuple."""

    __slots__ = ("_items",)

    def __init__(self, items: Sequence):
        """Initialize a new instance of TupleWrapper over `items`."""
        self._items = items

    def __len__(self):
        """Return the length of the underlying sequence."""
        return len(self._items)

    def __getitem__(self, item: int) -> Any:
        """Return element by index wrapped in a tuple."""
        return (self._items[item],)


@compiles(BinaryExpression)
def compile_binary(binary, compiler, override_operator=None, **kw):
    """
    If there are more than 10 elements in the `IN` set, inline them to avoid hitting the limit of \
    the number of query arguments in Postgres (1<<15).
    """  # noqa: D200
    operator = override_operator or binary.operator

    if operator is not in_op and operator is not not_in_op:
        return compiler.visit_binary(binary, override_operator=override_operator, **kw)

    if isinstance(binary.right, BindParameter):
        values = binary.right.value
        right_len = len(values)
    else:
        values = []
        right_len = 0
    if is_array := right_len == 1 and isinstance(values[0], np.ndarray):
        binary.right.value = values = values[0]
        right_len = len(values)
    render_any_values = (
        getattr(binary, "any_values", False)
        and compiler.dialect.name == "postgresql"
        and right_len > 0  # = ANY(VALUES) is invalid syntax
    )
    if right_len >= 10:
        # bypass the limit of the number of arguments
        kw["literal_binds"] = True
    if render_any_values:
        # = ANY(VALUES ...)
        if is_array and (
            values.dtype.kind in ("S", "U")
            or values.dtype.kind in ("i", "u")
            and values.dtype.itemsize == 8
        ):
            right = any_(Grouping(text(in_any_values_inline(values))))
        else:
            right = any_(
                Grouping(Values(binary.left, literal_binds=True).data(TupleWrapper(values))),
            )
        left = compiler.process(binary.left, **kw)
        right = compiler.process(right, **kw)
        sql = left + OPERATORS[operators.eq] + right
        if operator is not_in_op:
            sql = f"NOT ({sql})"
        return sql
    else:
        # IN (...)
        if operator is in_op and right_len == 1:
            # IN (<value>) -> = <value>
            value = values[0]
            if is_array and values.dtype.kind == "S":
                value = value.decode()
            return compiler.process(binary.left == value, **kw)
        if is_array:
            if (
                values.dtype.kind in ("S", "U")
                or values.dtype.kind in ("i", "u")
                and values.dtype.itemsize == 8
            ):
                binary.right = Grouping(text(in_inline(values)))
            else:
                binary.right.value = values.tolist()
        return compiler.visit_binary(binary, override_operator=override_operator, **kw)


_original_in_ = ColumnOperators.in_
_original_notin_ = ColumnOperators.notin_


def _in_(self: ColumnOperators, other: Iterable, any_: bool = False):
    """Override IN (...) PostgreSQL operator."""
    if isinstance(other, np.ndarray) and (any_ or other.dtype != object):
        other = [other]  # performance hack to avoid conversion to list
    return _original_in_(self, other)


def _notin_(self: ColumnOperators, other: Iterable, any_: bool = False):
    """Override NOT IN (...) PostgreSQL operator."""
    if isinstance(other, np.ndarray) and (any_ or other.dtype != object):
        other = [other]  # performance hack to avoid conversion to list
    return _original_notin_(self, other)


ColumnOperators.in_ = _in_
ColumnOperators.notin_ = _notin_


def _in_any_values(self: ColumnOperators, other: Iterable):
    """Implement = ANY(VALUES (...), (...), ...) PostgreSQL operator."""
    expr = self.in_(other, any_=True)
    expr.any_values = True
    return expr


def _notin_any_values(self: ColumnOperators, other: Iterable):
    """Implement NOT = ANY(VALUES (...), (...), ...) PostgreSQL operator."""
    expr = self.notin_(other, any_=True)
    expr.any_values = True
    return expr


ColumnOperators.in_any_values = _in_any_values
ColumnOperators.notin_any_values = _notin_any_values


flogging.trailing_dot_exceptions.add("alembic.runtime.migration")


class DBSchemaMismatchError(Exception):
    """Error raised if the DB schema versions do not match."""


def check_alembic_schema_version(name: str, conn_str: str, log: logging.Logger) -> None:
    """Raise DBSchemaVersionMismatchError if the real (connected) DB schema version \
    does not match the required (declared in the code) version."""
    template = import_module("%s.%s" % (__package__, name)).template
    directory = script.ScriptDirectory(str(template.parent))
    engine = create_engine(conn_str.split("?", 1)[0])
    with engine.begin() as conn:
        context = MigrationContext.configure(conn)
        real_rev = context.get_current_revision()
    req_rev = directory.get_current_head()
    if real_rev != req_rev:
        raise DBSchemaMismatchError(
            "%s version: required: %s connected: %s" % (conn_str, req_rev, real_rev),
        )
    log.info("%s DB schema version: %s", name, real_rev)


def check_collation(conn_str: str) -> None:
    """Force the PostgreSQL collation to be "C"."""
    engine = create_engine(conn_str.split("?", 1)[0])
    if engine.dialect.name != "postgresql":
        return
    collation = engine.scalar(
        "select datcollate from pg_database where datname='%s';" % engine.url.database,
    )
    if collation.lower() != "c.utf-8":
        raise DBSchemaMismatchError(
            "%s collation: required: C.UTF-8 connected: %s" % (conn_str, collation),
        )


def migrate(name: str, url=None, exec=True):
    """
    Migrate a database with alembic.

    This script creates all the tables if they don't exist and migrates the DB to the most
    recent version. It is to simplify the deployment.

    As a bonus, you obtain a functional Alembic INI config for any `alembic` commands.
    """
    root = import_module("%s.%s" % (__package__, name))
    template_file_name = root.template
    path = template_file_name.parent
    with open("alembic.ini", "w") as fout:
        fout.write(Template(filename=str(template_file_name)).render(url=url, path=path))
    args = [
        sys.executable,
        sys.executable,
        "-m",
        "athenian.api.sentry_wrapper",
        "alembic.config",
        "upgrade",
        "head",
    ]
    if os.getenv("OFFLINE"):
        args.append("--sql")
    if exec:
        os.execlp(*args)
    else:
        subprocess.run(args[1:], check=True)
