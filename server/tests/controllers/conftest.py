import asyncio
from datetime import datetime
import logging
import os
from pathlib import Path
import re

import aiohttp.web

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
from athenian.api.auth import Auth0, User
from athenian.api.models.metadata import hack_sqlite_arrays
from athenian.api.models.metadata.github import Base as MetadataBase
from athenian.api.models.state.models import Base as StateBase
from tests.sample_db_data import fill_metadata_session, fill_state_session


db_dir = Path(os.getenv("DB_DIR", os.path.dirname(os.path.dirname(__file__))))


class TestAuth0(Auth0):
    def __init__(self, whitelist):
        super().__init__(whitelist=whitelist, lazy_mgmt=True)
        self.user = User(
            id="auth0|5e1f6dfb57bc640ea390557b",
            email="vadim@athenian.co",
            name="Vadim Markovtsev",
            picture="https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
            updated=datetime.utcnow(),
        )

    async def _set_user(self, request) -> None:
        request.user = self.user


@pytest.fixture(scope="function")
async def eiso(app) -> User:
    user = User(
        id="auth0|5e1f6e2e8bfa520ea5290741",
        email="eiso@athenian.co",
        name="Eiso Kant",
        picture="https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
        updated=datetime.utcnow(),
    )
    app._auth0.user = user
    return user


@pytest.fixture(scope="function")
async def app(metadata_db, state_db) -> AthenianApp:
    logging.getLogger("connexion.operation").setLevel("WARNING")
    return AthenianApp(mdb_conn=metadata_db, sdb_conn=state_db, ui=False,  auth0_cls=TestAuth0)


@pytest.fixture(scope="function")
def client(loop, aiohttp_client, app):
    return loop.run_until_complete(aiohttp_client(app.app))


@pytest.fixture(scope="module")
def metadata_db() -> str:
    hack_sqlite_arrays()
    metadata_db_path = db_dir / "mdb.sqlite"
    conn_str = "sqlite:///%s" % metadata_db_path
    if metadata_db_path.exists():
        return conn_str
    engine = create_engine(conn_str)
    MetadataBase.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        fill_metadata_session(session)
        session.commit()
    finally:
        session.close()
    return conn_str


@pytest.fixture(scope="function")
def state_db() -> str:
    state_db_path = db_dir / "sdb.sqlite"
    conn_str = "sqlite:///%s" % state_db_path
    if state_db_path.exists():
        state_db_path.unlink()
    engine = create_engine(conn_str)
    StateBase.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        fill_state_session(session)
        session.commit()
    finally:
        session.close()
    return conn_str
