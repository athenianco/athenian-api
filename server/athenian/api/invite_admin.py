#!/usr/bin/python3

from random import randint
import sys

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from athenian.api.controllers import invitation_controller
from athenian.api.models.state.models import Account, Invitation


def main():
    """Add an admin invitation DB record and print the invitation URL."""
    if invitation_controller.ikey is None:
        raise EnvironmentError("ATHENIAN_INVITATION_KEY environment variable must be defined")
    engine = create_engine(sys.argv[1])
    session = sessionmaker(bind=engine)()
    salt = randint(0, (1 << 16) - 1)
    admin_backdoor = invitation_controller.admin_backdoor
    if not session.query(Account).filter(Account.id == admin_backdoor).all():
        session.add(Account(id=invitation_controller.admin_backdoor))
        session.commit()
        max_id = session.query(func.max(Account.id)).filter(Account.id < admin_backdoor).first()
        if max_id is None or max_id[0] is None:
            max_id = 0
        else:
            max_id = max_id[0]
        max_id += 1
        if engine.url.drivername in ("postgres", "postgresql"):
            engine.execute("ALTER SEQUENCE accounts_id_seq RESTART WITH %d;" % max_id)
        elif engine.url.drivername == "sqlite":
            engine.execute("UPDATE sqlite_sequence SET seq=%d WHERE NAME='accounts';" % max_id)
        else:
            raise NotImplementedError(
                "Cannot reset the primary key counter for " + engine.url.drivername)
    try:
        inv = Invitation(salt=salt, account_id=admin_backdoor)
        session.add(inv)
        session.flush()
        url_prefix = invitation_controller.url_prefix
        encode_slug = invitation_controller.encode_slug
        print(url_prefix + encode_slug(inv.id, inv.salt))
        session.commit()
    finally:
        session.close()


if __name__ == "__main__":
    exit(main())
