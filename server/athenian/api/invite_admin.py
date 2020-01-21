from random import randint
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from athenian.api.controllers import invitation_controller
from athenian.api.models.state.models import Invitation


def main():
    """Add an admin invitation DB record and print the invitation URL."""
    if invitation_controller.ikey is None:
        raise EnvironmentError("ATHENIAN_INVITATION_KEY environment variable must be defined")
    engine = create_engine(sys.argv[1])
    session = sessionmaker(bind=engine)()
    salt = randint(0, (1 << 16) - 1)
    try:
        inv = Invitation(salt=salt, account_id=invitation_controller.admin_backdoor)
        session.add(inv)
        session.flush()
        print(invitation_controller.prefix + invitation_controller.encode_slug(inv.id, inv.salt))
        session.commit()
    finally:
        session.close()


if __name__ == "__main__":
    exit(main())
