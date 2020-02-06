import logging
import os

from alembic import script
from alembic.migration import MigrationContext
from sqlalchemy import create_engine

from athenian.api import slogging


slogging.trailing_dot_exceptions.add("alembic.runtime.migration")


class StateDBSchemaVersionMismatchError(Exception):
    """Error raised if the DB schema versions do not match."""


def check_schema_version(conn_str: str, log: logging.Logger) -> None:
    """Raise StateDBSchemaVersionMismatchError if the real (connected) DB schema version \
    does not match the required (declared in the code) version."""
    directory = script.ScriptDirectory(os.path.dirname(__file__))
    engine = create_engine(conn_str)
    with engine.begin() as conn:
        context = MigrationContext.configure(conn)
        real_rev = context.get_current_revision()
    req_rev = directory.get_current_head()
    if real_rev != req_rev:
        raise StateDBSchemaVersionMismatchError("Required: %s Connected: %s" % (req_rev, real_rev))
    log.info("State DB schema version: %s", real_rev)
