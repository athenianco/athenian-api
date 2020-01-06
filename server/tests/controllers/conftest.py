import logging
import os
from pathlib import Path

try:
    import pytest
except ImportError:
    class pytest:
        @staticmethod
        def fixture(fn):
            return fn
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from athenian.api import AthenianApp
from athenian.api.models.metadata import hack_sqlite_arrays
from athenian.api.models.metadata.github import Base
from tests.sample_db_data import fill_session


db_path = Path(os.getenv("DB_DIR", os.path.dirname(os.path.dirname(__file__)))) / "db.sqlite"


@pytest.fixture
def client(loop, aiohttp_client, sqlite_db):
    logging.getLogger("connexion.operation").setLevel("ERROR")
    app = AthenianApp(mdb_conn="sqlite:///%s" % db_path, sdb_conn="sqlite://", ui=False)
    return loop.run_until_complete(aiohttp_client(app.app))


@pytest.fixture
def sqlite_db():
    hack_sqlite_arrays()
    if db_path.exists():
        return
    engine = create_engine("sqlite:///%s" % db_path, echo=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        fill_session(session)
        session.commit()
    finally:
        session.close()
