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
from athenian.api.models.metadata.github import Base as MetadataBase
from athenian.api.models.state.models import Base as StateBase
from tests.sample_db_data import fill_metadata_session, fill_state_session


db_dir = Path(os.getenv("DB_DIR", os.path.dirname(os.path.dirname(__file__))))





@pytest.fixture
def client(loop, aiohttp_client, metadata_db, state_db):
    logging.getLogger("connexion.operation").setLevel("ERROR")
    app = AthenianApp(mdb_conn=metadata_db, sdb_conn=state_db, ui=False)
    return loop.run_until_complete(aiohttp_client(app.app))


@pytest.fixture
def metadata_db():
    hack_sqlite_arrays()
    metadata_db_path = db_dir / "mdb.sqlite"
    if metadata_db_path.exists():
        return
    conn_str = "sqlite:///%s" % metadata_db_path
    engine = create_engine(conn_str, echo=True)
    MetadataBase.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        fill_metadata_session(session)
        session.commit()
    finally:
        session.close()
    return conn_str


@pytest.fixture
def state_db():
    state_db_path = db_dir / "sdb.sqlite"
    if state_db_path.exists():
        return
    conn_str = "sqlite:///%s" % state_db_path
    engine = create_engine(conn_str, echo=True)
    StateBase.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        fill_state_session(session)
        session.commit()
    finally:
        session.close()
    return conn_str
