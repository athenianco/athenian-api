from itertools import chain

from athenian.api.models.metadata.github import Base as GithubBase
from athenian.api.models.metadata.jira import Base as JiraBase


# Canonical repository URL prefixes.
PREFIXES = {
    "github": "github.com/",
}


def dereference_schemas():
    """Move table schema to the name prefix for the DBs that do not support it."""
    for table in chain(GithubBase.metadata.tables.values(),
                       JiraBase.metadata.tables.values()):
        if table.schema is not None:
            table.name = ".".join([table.schema, table.name])
            table.schema = None
