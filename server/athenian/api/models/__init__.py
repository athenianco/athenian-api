from importlib import import_module
import logging
import os
import subprocess
import sys
from typing import Any, Sequence

from alembic import script
from alembic.migration import MigrationContext
from flogging import flogging
from mako.template import Template
import numpy as np
from sqlalchemy import all_, any_, ARRAY, cast, create_engine, Text
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import operators, Values
from sqlalchemy.sql.compiler import OPERATORS
from sqlalchemy.sql.elements import BinaryExpression, BindParameter, Grouping, literal
from sqlalchemy.sql.operators import ColumnOperators, in_op, not_in_op
from sqlalchemy.sql.sqltypes import BigInteger, NullType

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
        right_len = len(binary.right.value)
    else:
        right_len = 0
    if right_len >= 10:
        postgres = compiler.dialect.name == "postgresql"
        left = compiler.process(binary.left, **kw)
        use_any = getattr(binary, "any_values", False) and postgres
        negate = use_any and operator is not_in_op
        if use_any:
            # ANY(VALUES ...) seems to be performing the best among these three:
            # 1. IN (...)
            # 2. IN(ARRAY[...])
            # 3. ANY(VALUES ...)
            kw["literal_binds"] = True
            operator = operators.eq
            right = any_(Grouping(Values(
                binary.left, literal_binds=True,
            ).data(TupleWrapper(binary.right.value))))
        elif postgres:
            if operator is not_in_op:
                operator = operators.ne
                agg = all_
            else:
                operator = operators.eq
                agg = any_
            values = ",".join(str(i) for i in binary.right.value)
            if values.count(",") >= len(binary.right.value) or "\\" in values:
                # we have to escape commas as \, and backslashes as \\
                values = ",".join(i.replace("\\", "\\\\").replace(",", "\\,")
                                  for i in binary.right.value)
            # not exactly correct: \,None, will match
            # but faster than checking each element
            values = values.replace(",None,", ",")
            if isinstance((left_type := binary.left.type), NullType):
                if isinstance(values[0], (int, np.integer)):
                    left_type = BigInteger
                else:
                    left_type = Text
            right = agg(cast(cast(literal(f"{{{values}}}"), Text), ARRAY(left_type)))
        else:
            right = binary.right
        right = compiler.process(right, **kw)
        sql = left + OPERATORS[operator] + right
        if negate:
            sql = "NOT (%s)" % sql
        return sql
    elif operator is in_op and right_len == 1:
        # IN (<value>) -> = <value>
        return compiler.process(binary.left == binary.right.value[0], **kw)

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
            "%s version: required: %s connected: %s" % (conn_str, req_rev, real_rev))
    log.info("%s DB schema version: %s", name, real_rev)


def check_collation(conn_str: str) -> None:
    """Force the PostgreSQL collation to be "C"."""
    engine = create_engine(conn_str.split("?", 1)[0])
    if engine.dialect.name != "postgresql":
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
    root = import_module("%s.%s" % (__package__, name))
    template_file_name = root.template
    path = template_file_name.parent
    with open("alembic.ini", "w") as fout:
        fout.write(Template(filename=str(template_file_name)).render(url=url, path=path))
    args = [sys.executable, sys.executable, "-m", "athenian.api.sentry_wrapper",
            "alembic.config", "upgrade", "head"]
    if os.getenv("OFFLINE"):
        args.append("--sql")
    if exec:
        os.execlp(*args)
    else:
        subprocess.run(args[1:], check=True)
