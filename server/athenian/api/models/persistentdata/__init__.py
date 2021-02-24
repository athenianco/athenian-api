from pathlib import Path

from athenian.api.models.persistentdata.models import Base

template = Path(__file__).with_name("alembic.ini.mako")


def dereference_schemas():
    """Move table schema to the name prefix for the DBs that do not support it."""
    for table in Base.metadata.tables.values():
        if table.schema is not None:
            table.name = ".".join([table.schema, table.name])
            table.schema = None
