import argparse
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

try:
    import pytest

    def fixture(*args, **kwargs):
        if args and not kwargs:
            return args[0]
        return lambda fn: fn

    pytest.fixture = fixture
except ImportError:
    pass

from tests.conftest import db_dir, metadata_db
from tests.sample_db_data import fill_persistentdata_session, fill_state_session
# this must go *after* tests to let tests.conftest check ATHENIAN_INVITATION_KEY
from athenian.api.models import migrate  # noqa: I100
from athenian.api.models.persistentdata import \
    dereference_schemas as dereference_persistentdata_schemas


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DB fixtures.")
    parser.add_argument("--no-state-samples", action="store_true",
                        help="Leave the initialized state DB empty.")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_db("master")
    for letter, name in (("s", "state"), ("p", "precomputed"), ("r", "persistentdata")):
        db_path = db_dir / ("%sdb-master.sqlite" % letter)
        if db_path.exists():
            db_path.unlink()
        conn_str = os.getenv(
            "OVERRIDE_%sDB" % letter.upper(), "sqlite:///%s" % db_path,
        ).rsplit("?", 1)[0]
        migrate(name, conn_str, exec=False)

        if letter == "s" and not args.no_state_samples:
            engine = create_engine(conn_str)
            session = sessionmaker(bind=engine)()
            try:
                fill_state_session(session)
                session.commit()
            finally:
                session.close()
        if letter == "r":
            engine = create_engine(conn_str)
            if engine.name == "sqlite":
                dereference_persistentdata_schemas()
            session = sessionmaker(bind=engine)()
            try:
                fill_persistentdata_session(session)
                session.commit()
            finally:
                session.close()
        if db_path.exists():
            os.chmod(db_path, 0o666)


if __name__ == "__main__":
    exit(main())
