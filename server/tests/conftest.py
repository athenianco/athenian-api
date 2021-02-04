import asyncio
import base64
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Dict, List, Optional, Union

from databases import Database
from filelock import FileLock
try:
    import nest_asyncio
except ImportError:
    class nest_asyncio:
        @staticmethod
        def apply():
            pass
import numpy as np
from prometheus_client import CollectorRegistry
try:
    import pytest
except ImportError:
    class pytest:
        @staticmethod
        def fixture(*args, **kwargs):
            if not kwargs:
                return args[0]
            return lambda fn: fn
from sqlalchemy import create_engine, func
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import sessionmaker
import uvloop

# This must stay before any athenian.api import to override any external ATHENIAN_INVITATION_KEY
os.environ["ATHENIAN_INVITATION_KEY"] = "vadim"

from athenian.api import create_memcached, create_slack, patch_pandas
from athenian.api.auth import Auth0, User
from athenian.api.cache import setup_cache_metrics
from athenian.api.connexion import AthenianApp
from athenian.api.controllers import account, invitation_controller
from athenian.api.db import ParallelDatabase
from athenian.api.metadata import __package__ as package
from athenian.api.models import check_collation, metadata
from athenian.api.models.metadata.github import Base as GithubBase, PullRequest
from athenian.api.models.metadata.jira import Base as JiraBase
from athenian.api.models.precomputed.models import Base as PrecomputedBase
from athenian.api.models.state.models import Base as StateBase
from tests.sample_db_data import fill_metadata_session, fill_state_session


if os.getenv("NEST_ASYNCIO"):
    nest_asyncio.apply()
uvloop.install()
np.seterr(all="raise")
patch_pandas()
db_dir = Path(os.getenv("DB_DIR", os.path.dirname(__file__)))
sdb_backup = tempfile.NamedTemporaryFile(prefix="athenian.api.state.", suffix=".sqlite")
pdb_backup = tempfile.NamedTemporaryFile(prefix="athenian.api.precomputed.", suffix=".sqlite")
assert Auth0.KEY == os.environ["ATHENIAN_INVITATION_KEY"], "athenian.api was imported before tests"
invitation_controller.url_prefix = "https://app.athenian.co/i/"
account.jira_url_template = invitation_controller.jira_url_template = \
    "https://installation.athenian.co/jira/%s/atlassian-connect.json"
override_mdb = os.getenv("OVERRIDE_MDB")
override_sdb = os.getenv("OVERRIDE_SDB")
override_pdb = os.getenv("OVERRIDE_PDB")
override_memcached = os.getenv("OVERRIDE_MEMCACHED")
logging.getLogger("aiosqlite").setLevel(logging.CRITICAL)


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

    async def multi_get(self, *keys: bytes) -> List[Optional[bytes]]:
        return [await self.get(k) for k in keys]

    async def set(self, key: bytes, value: Union[bytes, memoryview], exptime: int = 0) -> bool:
        assert isinstance(key, bytes)
        assert isinstance(value, (bytes, memoryview))
        assert isinstance(exptime, int)
        self.mem[key] = value, time.time(), exptime
        return True

    async def delete(self, key: bytes) -> bool:
        try:
            del self.mem[key]
            return True
        except KeyError:
            return False

    async def close(self):
        pass

    async def touch(self, key: bytes, exptime: int):
        pass


@pytest.fixture(scope="function")
def cache(loop, xapp):
    fc = FakeCache()
    setup_cache_metrics(fc, xapp.app, CollectorRegistry(auto_describe=True))
    for v in fc.metrics["context"].values():
        v.set(defaultdict(int))
    return fc


@pytest.fixture(scope="function")
def client_cache(loop, app):
    fc = FakeCache()
    setup_cache_metrics(fc, app.app, CollectorRegistry(auto_describe=True))
    for v in fc.metrics["context"].values():
        v.set(defaultdict(int))
    return fc


@pytest.fixture(scope="function")
def memcached(loop, xapp, request):
    client = create_memcached(override_memcached or "0.0.0.0:11211", logging.getLogger("pytest"))
    client.version_future.cancel()
    trash = []
    set = client.set

    async def tracked_set(key, value, exptime=0):
        trash.append(key)
        return await set(key, value, exptime=exptime)

    client.set = tracked_set

    def shutdown():
        async def delete_trash():
            for key in trash:
                await client.delete(key)
            await client.close()

        loop.run_until_complete(delete_trash())

    request.addfinalizer(shutdown)
    setup_cache_metrics(client, xapp.app, CollectorRegistry(auto_describe=True))
    for v in client.metrics["context"].values():
        v.set(defaultdict(int))
    return client


def check_memcached():
    client = create_memcached(override_memcached or "0.0.0.0:11211", logging.getLogger("pytest"))
    client.version_future.cancel()

    async def probe():
        try:
            await client.get(b"0")
            return True
        except ConnectionRefusedError:
            return False
        finally:
            await client.close()

    return asyncio.get_event_loop().run_until_complete(probe())


has_memcached = check_memcached()


class TestAuth0(Auth0):
    def __init__(self, whitelist, cache=None):
        super().__init__(
            whitelist=whitelist, default_user="auth0|5e1f6dfb57bc640ea390557b", lazy=True)
        self._default_user = User(
            id="auth0|5e1f6dfb57bc640ea390557b",
            login="vadim",
            email="vadim@athenian.co",
            name="Vadim Markovtsev",
            native_id="5e1f6dfb57bc640ea390557b",
            picture="https://s.gravatar.com/avatar/d7fb46e4e35ecf7c22a1275dd5dbd303?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fva.png",  # noqa
            updated=datetime.now(timezone.utc),
        )


class FakeKMS:
    async def encrypt(self, plaintext: Union[bytes, str]) -> str:
        if isinstance(plaintext, str):
            plaintext = plaintext.encode()
        return base64.b64encode(plaintext).decode()

    async def decrypt(self, ciphertext: str) -> bytes:
        return base64.b64decode(ciphertext.encode())

    async def close(self):
        pass


@pytest.fixture(scope="module")
def eiso_user() -> User:
    return User(
        id="auth0|5e1f6e2e8bfa520ea5290741",
        login="eiso",
        email="eiso@athenian.co",
        name="Eiso Kant",
        native_id="5e1f6e2e8bfa520ea5290741",
        picture="https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
        updated=datetime.now(timezone.utc),
    )


@pytest.fixture(scope="function")
async def eiso(app, eiso_user) -> User:
    app._auth0._default_user_id = eiso_user.id
    app._auth0._default_user = eiso_user
    return eiso_user


@pytest.fixture(scope="function")
async def gkwillie(app) -> User:
    app._auth0._default_user_id = "github|60340680"
    app._auth0._default_user = User(
        id="github|60340680",
        login="gkwillie",
        email="bot@athenian.co",
        name="Groundskeeper Willie",
        native_id="60340680",
        picture="https://s.gravatar.com/avatar/dfe23533b671f82d2932e713b0477c75?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fei.png",  # noqa
        updated=datetime.now(timezone.utc),
    )


@pytest.fixture(scope="function")
async def lazy_gkwillie(app) -> User:
    app._auth0._default_user_id = "github|60340680"
    app._auth0._default_user = None


@pytest.fixture(scope="function")
def disable_default_user(app):
    _extract_token = app._auth0._extract_bearer_token
    default_user_id = app._auth0._default_user_id

    async def hacked_extract_token(token: str):
        app._auth0._default_user_id = default_user_id
        r = await _extract_token(token)
        app._auth0._default_user_id = "xxx"
        return r

    app._auth0._extract_bearer_token = hacked_extract_token


@pytest.fixture(scope="function")
def headers() -> Dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Origin": "http://localhost",
    }


@pytest.fixture(scope="module")
def slack():
    return create_slack(logging.getLogger("pytest"))


@pytest.fixture(scope="function")
async def app(metadata_db, state_db, precomputed_db, slack) -> AthenianApp:
    logging.getLogger("connexion.operation").setLevel("WARNING")
    return AthenianApp(mdb_conn=metadata_db, sdb_conn=state_db, pdb_conn=precomputed_db,
                       ui=False, auth0_cls=TestAuth0, kms_cls=FakeKMS, slack=slack,
                       client_max_size=256 * 1024, max_load=15)


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
def metadata_db(worker_id) -> str:
    metadata.__version__ = metadata.__min_version__
    if override_mdb:
        conn_str = override_mdb % worker_id
    else:
        metadata_db_path = db_dir / ("mdb-%s.sqlite" % worker_id)
        conn_str = "sqlite:///%s" % metadata_db_path
    engine = create_engine(conn_str.rsplit("?", 1)[0])
    if engine.url.drivername == "postgresql":
        engine.execute("CREATE SCHEMA IF NOT EXISTS github;")
        engine.execute("CREATE SCHEMA IF NOT EXISTS jira;")
    else:
        metadata.dereference_schemas()
    GithubBase.metadata.create_all(engine)
    JiraBase.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    if session.query(func.count(PullRequest.node_id)).one()[0] == 0:
        try:
            fill_metadata_session(session)
            session.commit()
        finally:
            session.close()
        if not override_mdb:
            os.chmod(metadata_db_path, 0o666)
    check_collation(conn_str)
    return conn_str


def init_own_db(letter: str,
                base: DeclarativeMeta,
                worker_id: str,
                init_sql: Optional[dict] = None):
    override_db = globals()["override_%sdb" % letter]
    backup_path = globals()["%sdb_backup" % letter].name
    if override_db:
        conn_str = override_db % worker_id
        db_path = None
    else:
        db_path = db_dir / ("%sdb-%s.sqlite" % (letter, worker_id))
        conn_str = "sqlite:///%s" % db_path
        if db_path.exists():
            db_path.unlink()
        if Path(backup_path).stat().st_size > 0:
            shutil.copy(backup_path, db_path)
            return conn_str
    engine = create_engine(conn_str.rsplit("?", 1)[0])
    base.metadata.drop_all(engine)
    if init_sql:
        driver = engine.url.drivername
        if driver == "postgres":
            driver = "postgresql"
        try:
            init_sql = init_sql[driver]
        except KeyError:
            pass
        else:
            engine.execute(init_sql)
    base.metadata.create_all(engine)
    if letter == "s":
        session = sessionmaker(bind=engine)()
        try:
            fill_state_session(session)
            session.commit()
        finally:
            session.close()
    if not override_db:
        os.chmod(db_path, 0o666)
        if not Path(backup_path).exists():
            shutil.copy(db_path, backup_path)
    return conn_str


@pytest.fixture(scope="function")
def state_db(worker_id) -> str:
    return init_own_db("s", StateBase, worker_id)


@pytest.fixture(scope="function")
def precomputed_db(worker_id) -> str:
    return init_own_db("p", PrecomputedBase, worker_id, {
        "postgresql": "create extension if not exists hstore;",
    })


@pytest.fixture(scope="function")
async def mdb(metadata_db, loop, request):
    db = ParallelDatabase(metadata_db)
    await db.connect()

    def shutdown():
        loop.run_until_complete(db.disconnect())

    request.addfinalizer(shutdown)
    return db


@pytest.fixture(scope="function")
async def sdb(state_db, loop, request):
    db = ParallelDatabase(state_db)
    await db.connect()

    def shutdown():
        loop.run_until_complete(db.disconnect())

    request.addfinalizer(shutdown)
    return db


@pytest.fixture(scope="function")
async def pdb(precomputed_db, loop, request):
    if precomputed_db.startswith("sqlite"):
        db = Database(precomputed_db)
    else:
        db = ParallelDatabase(precomputed_db)
    db.metrics = {
        "hits": ContextVar("pdb_hits", default=defaultdict(int)),
        "misses": ContextVar("pdb_misses", default=defaultdict(int)),
    }
    await db.connect()
    if precomputed_db.startswith("sqlite"):
        db._global_connection = db.connection()

    def shutdown():
        loop.run_until_complete(db.disconnect())

    request.addfinalizer(shutdown)
    return db


@pytest.fixture(scope="session")
def locked_migrations(request):
    fn = os.path.join(tempfile.gettempdir(), "%s.migrations.lock" % package)

    def cleanup():
        try:
            os.remove(fn)
        except FileNotFoundError:
            return

    request.addfinalizer(cleanup)
    return FileLock(fn)


def pytest_addoption(parser):
    parser.addoption("--limit", action="store", default=-1, type=float,
                     help="Max number of tests to run.")


def pytest_collection_modifyitems(session, config, items):
    if (limit := config.getoption("--limit")) >= 0:
        random.seed(time.time() // 60)
        if limit < 1:
            limit = len(items) * limit
        limit = int(limit)
        indexes = list(range(len(items)))
        indexes = sorted(random.sample(indexes, limit))
        items[:] = [items[i] for i in indexes]
