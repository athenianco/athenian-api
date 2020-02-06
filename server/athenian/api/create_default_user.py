#!/usr/bin/python3

import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from athenian.api import Auth0
from athenian.api.models.state.models import Account, RepositorySet, UserAccount


def main():
    """Make sure that /user and /reposets work without authorization."""
    engine = create_engine(sys.argv[1])
    session = sessionmaker(bind=engine)()
    acc = Account()
    session.add(acc)
    session.flush()
    session.add(UserAccount(
        user_id=Auth0.DEFAULT_USER,
        account_id=acc.id,
        is_admin=False,
    ))
    session.add(RepositorySet(owner=acc.id, items=[
        "github.com/athenianco/athenian-api",
        "github.com/athenianco/metadata",
        "github.com/athenianco/athenian-webapp",
        "github.com/athenianco/infrastructure",
    ]))
    session.commit()


if __name__ == "__main__":
    exit(main())
