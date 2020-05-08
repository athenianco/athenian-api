import argparse
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

try:
    import pytest

    def fixture(*args, **kwargs):
        if not kwargs:
            return args[0]
        return lambda fn: fn

    pytest.fixture = fixture
except ImportError:
    pass

from athenian.api.models import migrate
from tests.conftest import db_dir, metadata_db
from tests.sample_db_data import fill_state_session


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DB fixtures.")
    parser.add_argument("--no-state-samples", action="store_true",
                        help="Leave the initialized state DB empty.")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_db()
    for letter, name in (("s", "state"), ("p", "precomputed")):
        db_path = db_dir / ("%sdb.sqlite" % letter)
        if db_path.exists():
            db_path.unlink()
        conn_str = "sqlite:///%s" % db_path
        migrate(name, conn_str, exec=False)

        if letter == "s" and not args.no_state_samples:
            engine = create_engine(conn_str)
            session = sessionmaker(bind=engine)()
            try:
                fill_state_session(session)
                session.commit()
            finally:
                session.close()
        os.chmod(db_path, 0o666)


if __name__ == "__main__":
    exit(main())
