from datetime import datetime
import logging
import os
from pathlib import Path
import time
from typing import Dict, Optional, Union

import databases
try:
    import pytest
except ImportError:
    class pytest:
        @staticmethod
        def fixture(*args, **kwargs):
            if not kwargs:
                return args[0]
            return lambda fn: fn
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from athenian.api import AthenianApp
from athenian.api.auth import Auth0, User
from athenian.api.controllers import invitation_controller
from athenian.api.models.metadata import hack_sqlite_arrays
from athenian.api.models.metadata.github import Base as MetadataBase
from athenian.api.models.state.models import Base as StateBase
from tests.sample_db_data import fill_metadata_session, fill_state_session


db_dir = Path(os.getenv("DB_DIR", os.path.dirname(__file__)))
invitation_controller.ikey = "vadim"
invitation_controller.url_prefix = "https://app.athenian.co/i/"
override_mdb = os.getenv("OVERRIDE_MDB")


class FakeCache:
    def __init__(self):
        self.mem = {}

    async def get(self, key: bytes, default: Optional[bytes] = None) -> Optional[bytes]:
        assert isinstance(key, bytes)
        assert default is None or isinstance(default, bytes)
        if key not in self.mem:
            return default
        value, start, exp = self.mem[key]
        if exp < 0 or 0 < exp < time.time() - start:
            return default
        return value

    async def set(self, key: bytes, value: Union[bytes, memoryview], exptime: int = 0) -> bool:
        assert isinstance(key, bytes)
        assert isinstance(value, (bytes, memoryview))
        assert isinstance(exptime, int)
        self.mem[key] = value, time.time(), exptime
        return True

    async def close(self):
        pass


@pytest.fixture(scope="function")
def cache():
    return FakeCache()


class TestAuth0(Auth0):
    def __init__(self, whitelist, cache=None):
        super().__init__(
            whitelist=whitelist, default_user="auth0|5e1f6dfb57bc640ea390557b", lazy=True)
        self._default_user = User(
            id="auth0|5e1f6dfb57bc640ea390557b",
            email="vadim@athenian.co",
            name="Vadim Markovtsev",
            native_id="5e1f6dfb57bc640ea390557b",
            picture="https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
            updated=datetime.utcnow(),
        )


@pytest.fixture(scope="function")
async def eiso(app) -> User:
    user = User(
        id="auth0|5e1f6e2e8bfa520ea5290741",
        email="eiso@athenian.co",
        name="Eiso Kant",
        native_id="5e1f6e2e8bfa520ea5290741",
        picture="https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
        updated=datetime.utcnow(),
    )
    app._auth0._default_user_id = "auth0|5e1f6e2e8bfa520ea5290741"
    app._auth0._default_user = user
    return user


@pytest.fixture(scope="function")
async def gkwillie(app) -> User:
    app._auth0._default_user_id = "github|60340680"
    app._auth0._default_user = None


@pytest.fixture(scope="function")
def headers() -> Dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Origin": "http://localhost",
    }


@pytest.fixture(scope="function")
async def app(metadata_db, state_db) -> AthenianApp:
    logging.getLogger("connexion.operation").setLevel("WARNING")
    return AthenianApp(mdb_conn=metadata_db, sdb_conn=state_db, ui=False, auth0_cls=TestAuth0)


@pytest.fixture(scope="function")
async def xapp(app: AthenianApp, request, loop) -> AthenianApp:
    def shutdown():
        loop.run_until_complete(app.shutdown(app))

    request.addfinalizer(shutdown)
    return app


@pytest.fixture(scope="function")
def client(loop, aiohttp_client, app):
    return loop.run_until_complete(aiohttp_client(app.app))


@pytest.fixture(scope="module")
def metadata_db() -> str:
    if override_mdb:
        return override_mdb
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
    os.chmod(metadata_db_path, 0o666)
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
    os.chmod(state_db_path, 0o666)
    return conn_str


@pytest.fixture(scope="function")
async def mdb(metadata_db, loop):
    db = databases.Database(metadata_db)
    await db.connect()
    return db


@pytest.fixture(scope="function")
async def sdb(state_db, loop):
    state_db_path = db_dir / "sdb.sqlite"
    conn_str = "sqlite:///%s" % state_db_path
    db = databases.Database(conn_str)
    await db.connect()
    return db
