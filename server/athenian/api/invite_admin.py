#!/usr/bin/python3
import argparse
from datetime import datetime, timezone
import logging
from random import randint

from sqlalchemy import and_, create_engine, func
from sqlalchemy.orm import sessionmaker

from athenian.api.auth import Auth0
from athenian.api.controllers import invitation_controller
from athenian.api.models import check_alembic_schema_version
from athenian.api.models.state.models import Account, Invitation
from athenian.api.slogging import add_logging_args


def parse_args():
    """Parse the cmdline arguments."""
    parser = argparse.ArgumentParser()
    add_logging_args(parser)
    parser.add_argument("conn_str", help="SQLAlchemy connection string")
    parser.add_argument(
        "--force-new", action="store_true", help="Always generate a new invitation."
    )
    return parser.parse_args()


def main(conn_str: str, force_new: bool = False) -> None:
    """Add an admin invitation DB record and print the invitation URL."""
    invitation_controller.validate_env()
    log = logging.getLogger("invite_admin")
    check_alembic_schema_version("state", conn_str, log)
    engine = create_engine(conn_str)
    session = sessionmaker(bind=engine)()
    salt = randint(0, (1 << 16) - 1)
    admin_backdoor = invitation_controller.admin_backdoor
    if not session.query(Account).filter(Account.id == admin_backdoor).all():
        session.add(
            Account(
                id=invitation_controller.admin_backdoor,
                secret_salt=0,
                secret=Account.missing_secret,
                expires_at=datetime.now(timezone.utc) + invitation_controller.TRIAL_PERIOD,
            )
        )
        session.commit()
        max_id = session.query(func.max(Account.id)).filter(Account.id < admin_backdoor).first()
        if max_id is None or max_id[0] is None:
            max_id = 0
        else:
            max_id = max_id[0]
        max_id += 1
        if engine.url.drivername == "postgresql":
            session.execute("ALTER SEQUENCE accounts_id_seq RESTART WITH %d;" % max_id)
        elif engine.url.drivername == "sqlite":
            pass
            # This will not help, unfortunately.
            # session.execute("UPDATE sqlite_sequence SET seq=%d WHERE name='accounts';" % max_id)
        else:
            raise NotImplementedError(
                "Cannot reset the primary key counter for " + engine.url.drivername
            )
    if not force_new:
        issued = (
            session.query(Invitation)
            .filter(and_(Invitation.account_id == admin_backdoor, Invitation.is_active))
            .order_by(Invitation.created_by)
            .all()
        )
    else:
        issued = []
    try:
        if not issued:
            inv = Invitation(salt=salt, account_id=admin_backdoor)
            session.add(inv)
            session.flush()
        else:
            inv = issued[-1]
        url_prefix = invitation_controller.url_prefix
        encode_slug = invitation_controller.encode_slug
        print(url_prefix + encode_slug(inv.id, inv.salt, Auth0.KEY))
        session.commit()
    finally:
        session.close()


if __name__ == "__main__":
    exit(main(**vars(parse_args())))
