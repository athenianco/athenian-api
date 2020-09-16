from itertools import chain
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from athenian.api.models import DBSchemaMismatchError
from athenian.api.models.metadata.github import Base as GithubBase, SchemaMigration
from athenian.api.models.metadata.jira import Base as JiraBase


# Canonical repository URL prefixes.
PREFIXES = {
    "github": "github.com/",
}


__min_version__ = 13
__version__ = None


def dereference_schemas():
    """Move table schema to the name prefix for the DBs that do not support it."""
    for table in chain(GithubBase.metadata.tables.values(),
                       JiraBase.metadata.tables.values()):
        if table.schema is not None:
            table.name = ".".join([table.schema, table.name])
            table.schema = None


def check_schema_version(conn_str: str, log: logging.Logger) -> None:
    """Validate the metadata DB schema version."""
    engine = create_engine(conn_str.split("?", 1)[0])
    global __version__
    session = sessionmaker(bind=engine)()
    try:
        __version__ = session.query(SchemaMigration.version).scalar()
    finally:
        session.close()
    if __version__ < __min_version__:
        raise DBSchemaMismatchError(
            "%s version: required: %s connected: %s" % (conn_str, __min_version__, __version__))
    log.info("metadata DB schema version: %s", __version__)
