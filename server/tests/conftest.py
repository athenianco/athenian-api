import asyncio
import base64
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import random
import shutil
from sqlite3 import OperationalError
import sys
import tempfile
import time
import traceback
from typing import Dict, List, Optional, Union, cast
import warnings

import aiomcache
from filelock import FileLock
import sentry_sdk
import sqlalchemy as sa

# FIXME(vmarkovtsev): if you remove this, test_integrations_controller.py is going to SIGSEGV
# The root cause is https://github.com/apache/arrow/issues/15054
# We should be able to remove this import once pyarrow 12.0 releases a fix
import athenian.api  # noqa: F401

try:
    import nest_asyncio
except ImportError:

    class nest_asyncio:
        @staticmethod
        def apply():
            pass


import numpy as np

try:
    import pytest
except ImportError:

    class pytest:
        @staticmethod
        def fixture(*args, **kwargs):
            if not kwargs:
                return args[0]
            return lambda fn: fn


import pytest_sentry
from sqlalchemy import create_engine, delete, func, insert, select
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import sessionmaker
import uvloop

# This must stay before any athenian.api import to override any external ATHENIAN_INVITATION_KEY
os.environ["ATHENIAN_INVITATION_KEY"] = "vadim"

from athenian.api.__main__ import (
    _ApplicationEnvironment,
    _init_sentry,
    create_mandrill,
    create_memcached,
    create_slack,
)
from athenian.api.aiohttp_addons import create_aiohttp_closed_event
from athenian.api.application import AthenianApp
from athenian.api.async_utils import read_sql_query
from athenian.api.auth import Auth0, User
from athenian.api.cache import CACHE_VAR_NAME, setup_cache_metrics
from athenian.api.controllers import invitation_controller
from athenian.api.db import Connection, Database, db_retry_intervals, measure_db_overhead_and_retry
from athenian.api.defer import with_defer
from athenian.api.internal import account
from athenian.api.internal.features.entries import MetricEntriesCalculator, PRFactsCalculator
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github import deployment_light
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.commit import _empty_dag, _fetch_commit_history_edges
from athenian.api.internal.miners.github.dag_accelerated import join_dags
from athenian.api.internal.miners.github.deployment import mine_deployments
from athenian.api.internal.miners.github.precomputed_prs import (
    DonePRFactsLoader,
    MergedPRFactsLoader,
    OpenPRFactsLoader,
)
from athenian.api.internal.miners.github.pull_request import PullRequestMiner
from athenian.api.internal.miners.github.release_load import ReleaseLoader
from athenian.api.internal.miners.github.release_match import ReleaseToPullRequestMapper
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import (
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseMatchSetting,
    ReleaseSettings,
    default_branch_alias,
)
from athenian.api.metadata import __package__ as package
from athenian.api.models import check_collation, metadata, persistentdata
from athenian.api.models.metadata.github import (
    Account,
    Base as GithubBase,
    NodeCommit,
    PullRequest,
    ShadowBase as ShadowGithubBase,
)
from athenian.api.models.metadata.jira import Base as JiraBase
from athenian.api.models.persistentdata.models import (
    Base as PersistentdataBase,
    DeployedComponent,
    DeployedLabel,
    DeploymentNotification,
)
from athenian.api.models.precomputed.models import GitHubBase as PrecomputedBase
from athenian.api.models.state.models import (
    Base as StateBase,
    God,
    LogicalRepository,
    RepositorySet,
)
from athenian.api.prometheus import PROMETHEUS_REGISTRY_VAR_NAME
from athenian.api.request import AthenianWebRequest
from athenian.precomputer.db import dereference_schemas as dereference_precomputed_schemas
from tests.sample_db_data import (
    fill_metadata_session,
    fill_persistentdata_session,
    fill_state_session,
)
from tests.testutils.db import models_insert
from tests.testutils.factory.miners import PullRequestFactsFactory
from tests.testutils.factory.state import ReleaseSettingFactory

if os.getenv("NEST_ASYNCIO"):
    nest_asyncio.apply()
uvloop.install()
np.seterr(all="raise")
db_dir = Path(os.getenv("DB_DIR", os.path.dirname(__file__)))
sdb_backup = tempfile.NamedTemporaryFile(prefix="athenian.api.state.", suffix=".sqlite")
pdb_backup = tempfile.NamedTemporaryFile(prefix="athenian.api.precomputed.", suffix=".sqlite")
rdb_backup = tempfile.NamedTemporaryFile(prefix="athenian.api.persistentdata.", suffix=".sqlite")
assert Auth0.KEY == os.environ["ATHENIAN_INVITATION_KEY"], "athenian.api was imported before tests"
invitation_controller.url_prefix = "https://app.athenian.co/i/"
account.jira_url_template = (
    invitation_controller.jira_url_template
) = "https://installation.athenian.co/jira/%s/atlassian-connect.json"
override_mdb = os.getenv("OVERRIDE_MDB")
override_sdb = os.getenv("OVERRIDE_SDB")
override_pdb = os.getenv("OVERRIDE_PDB")
override_rdb = os.getenv("OVERRIDE_RDB")
override_memcached = os.getenv("OVERRIDE_MEMCACHED")
logging.getLogger("aiosqlite").setLevel(logging.CRITICAL)
db_retry_intervals.insert(-2, 5)  # reduce the probability of TooManyConnectionsError in Postgres
os.environ["SENTRY_SAMPLING_RATE"] = "0.001"  # ~20 transactions per day
# do not forget to update PYTEST_SENTRY_TRACES_SAMPLE_RATE in GHA
os.environ["SENTRY_ENV"] = "test"
# do not forget to update SENTRY_ENVIRONMENT in GHA
if _init_sentry(
    log := logging.getLogger("test-api"),
    _ApplicationEnvironment.discover(log),
    extra_integrations=[pytest_sentry.PytestIntegration(always_report=True)],
):
    pytest_sentry.DEFAULT_HUB = sentry_sdk.Hub.current


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


def build_fake_cache() -> aiomcache.Client:
    cache = FakeCache()
    app = {CACHE_VAR_NAME: cache, PROMETHEUS_REGISTRY_VAR_NAME: None}
    setup_cache_metrics(app)
    for v in cache.metrics["context"].values():
        v.set(defaultdict(int))
    cache.metrics["context"] = app["cache_context"]
    return cast(aiomcache.Client, cache)


@pytest.fixture(scope="function")
def cache(event_loop):
    return build_fake_cache()


@pytest.fixture(scope="function")
def client_cache(event_loop, app):
    app.app[CACHE_VAR_NAME] = fc = FakeCache()
    setup_cache_metrics(app.app)
    for v in fc.metrics["context"].values():
        v.set(defaultdict(int))
    app.app[CACHE_VAR_NAME] = fc
    return fc


@pytest.fixture(scope="function")
def memcached(event_loop, xapp, request):
    old_cache = xapp.app[CACHE_VAR_NAME]
    xapp.app[CACHE_VAR_NAME] = client = create_memcached(
        override_memcached or "0.0.0.0:11211", logging.getLogger("pytest"),
    )
    client.version_future.cancel()
    trash = []
    set_ = client.set

    async def tracked_set(key, value, exptime=0):
        trash.append(key)
        return await set_(key, value, exptime=exptime)

    client.set = tracked_set

    def shutdown():
        async def delete_trash():
            for key in trash:
                await client.delete(key)
            await client.close()

        event_loop.run_until_complete(delete_trash())

    request.addfinalizer(shutdown)
    try:
        setup_cache_metrics(xapp.app)
    except ValueError:
        # double metrics registration is harmless
        client.metrics = old_cache.metrics
    for v in client.metrics["context"].values():
        v.set(defaultdict(int))
    return client


def check_memcached():
    loop = asyncio.new_event_loop()

    client = create_memcached(
        override_memcached or "0.0.0.0:11211",
        logging.getLogger("pytest"),
        loop=loop,
    )
    client.version_future.cancel()

    async def probe():
        try:
            await client.get(b"0")
            return True
        except ConnectionRefusedError:
            return False
        finally:
            await client.close()

    try:
        result = loop.run_until_complete(probe())
    finally:
        loop.close()
    return result


has_memcached = check_memcached()


class TestAuth0(Auth0):
    def __init__(self, whitelist, cache=None):
        super().__init__(
            whitelist=whitelist, default_user="auth0|62a1ae88b6bba16c6dbc6870", lazy=True,
        )
        self._default_user = User(
            id="auth0|62a1ae88b6bba16c6dbc6870",
            login="vadim",
            email="vadim@athenian.co",
            name="Vadim Markovtsev",
            native_id="62a1ae88b6bba16c6dbc6870",
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


@pytest.fixture(scope="session")
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
    _extract_api_key = app._auth0._extract_api_key
    default_user_id = None

    async def hacked_extract_token(token: str):
        nonlocal default_user_id
        if default_user_id is None:
            default_user_id = app._auth0._default_user_id
        else:
            app._auth0._default_user_id = default_user_id
        r = await _extract_token(token)
        app._auth0._default_user_id = "xxx"
        return r

    async def hacked_extract_api_key(token: str, request: AthenianWebRequest):
        nonlocal default_user_id
        if default_user_id is None:
            default_user_id = app._auth0._default_user_id
        else:
            app._auth0._default_user_id = default_user_id
        r = await _extract_api_key(token, request)
        app._auth0._default_user_id = "xxx"
        return r

    app._auth0._extract_bearer_token = hacked_extract_token
    app._auth0._extract_api_key = hacked_extract_api_key


@pytest.fixture(scope="function")
async def god(sdb) -> None:
    await sdb.execute(
        insert(God).values(
            God(user_id="auth0|62a1ae88b6bba16c6dbc6870")
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )


@pytest.fixture(scope="function")
def headers() -> Dict[str, str]:
    # DO NOT MAKE THIS A GLOBAL SCOPE VARIABLE
    # THIS MUST BE A FIXTURE
    # Otherwise, it is too easy to mutate.
    return {
        # DO NOT
        "Accept": "application/json",
        # MAKE THIS
        "Content-Type": "application/json",
        # A GLOBAL
        "Origin": "http://localhost",
        # VARIABLE
    }


@pytest.fixture(scope="function")
def slack(request, event_loop):
    client = create_slack(logging.getLogger("pytest"))
    if client is not None:

        async def close_slack():
            client.session_future.cancel()
            if client.session is not None:
                close_event = create_aiohttp_closed_event(client.session)
                await client.session.close()
                await close_event.wait()

        def shutdown():
            event_loop.run_until_complete(close_slack())

        request.addfinalizer(shutdown)
    return client


@pytest.fixture(scope="function")
async def mandrill(request, event_loop):
    if (client := create_mandrill()) is not None:

        def shutdown():
            event_loop.run_until_complete(client.close())

        request.addfinalizer(shutdown)
    return client


@pytest.fixture(scope="function")
async def app(
    metadata_db,
    state_db,
    precomputed_db,
    persistentdata_db,
    slack,
    request,
) -> AthenianApp:
    """Build the especifico App to be used during tests

    By default handler responses will be validated against oas spec,
    `app_validate_response` mark can be applied to disable this behavior:

    ```
    @pytest.mark.app_validate_response(False)
    def my_test(app):
        ...
    ```
    """
    logging.getLogger("especifico.operation").setLevel("WARNING")

    validate_responses_mark = request.node.get_closest_marker("app_validate_responses")
    if validate_responses_mark is None:
        validate_responses = True
    else:
        validate_responses = validate_responses_mark.args[0]
    pgdb_opts = {"pgbouncer_statement": True}

    app = AthenianApp(
        mdb_conn=metadata_db,
        sdb_conn=state_db,
        pdb_conn=precomputed_db,
        rdb_conn=persistentdata_db,
        mdb_options=pgdb_opts if metadata_db.startswith("postgresql") else None,
        sdb_options=pgdb_opts if state_db.startswith("postgresql") else None,
        pdb_options=pgdb_opts if precomputed_db.startswith("postgresql") else None,
        rdb_options=pgdb_opts if persistentdata_db.startswith("postgresql") else None,
        ui=False,
        auth0_cls=TestAuth0,
        kms_cls=FakeKMS,
        slack=slack,
        client_max_size=256 * 1024,
        max_load=15,
        with_pdb_schema_checks=False,
        validate_responses=validate_responses,
    )
    await app.ready()
    return app


@pytest.fixture(scope="function")
async def xapp(app: AthenianApp, request, event_loop) -> AthenianApp:
    def shutdown():
        event_loop.run_until_complete(app.shutdown())

    request.addfinalizer(shutdown)
    return app


@pytest.fixture(scope="function")
def client(event_loop, aiohttp_client, app):
    return event_loop.run_until_complete(aiohttp_client(app.app))


@pytest.fixture(scope="session")
def metadata_db(worker_id) -> str:
    return _metadata_db(worker_id, False)


def _metadata_db(worker_id: str, force_reset: bool) -> str:
    metadata.__version__ = metadata.__min_version__
    if override_mdb:
        conn_str = override_mdb % worker_id
    else:
        metadata_db_path = db_dir / f"mdb-{worker_id}.sqlite"
        if force_reset:
            metadata_db_path.unlink(missing_ok=True)
        conn_str = f"sqlite:///{metadata_db_path}"
    engine = create_engine(conn_str.rsplit("?", 1)[0])
    if engine.url.drivername == "postgresql":
        with engine.begin() as conn:
            conn.execute(sa.text("CREATE SCHEMA IF NOT EXISTS github;"))
            conn.execute(sa.text("CREATE SCHEMA IF NOT EXISTS jira;"))
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS hstore;"))
    else:
        metadata.dereference_schemas()
    ShadowGithubBase.metadata.create_all(engine)
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


def _init_own_db(
    letter: str,
    base: DeclarativeMeta,
    worker_id: str,
    init_sql: Optional[dict] = None,
) -> str:
    try:
        return _init_own_db_unchecked(letter, base, worker_id, init_sql)
    except Exception:
        print(f"Unable to initialize {letter}db", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def _init_own_db_unchecked(
    letter: str,
    base: DeclarativeMeta,
    worker_id: str,
    init_sql: Optional[dict] = None,
) -> str:
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
    driver = engine.url.drivername
    if letter in ("r", "p") and driver == "sqlite":
        if letter == "r":
            persistentdata.dereference_schemas()
        if letter == "p":
            dereference_precomputed_schemas()
    base.metadata.drop_all(engine)
    if init_sql:
        try:
            init_sql_text = init_sql[driver]
        except KeyError:
            pass
        else:
            with engine.begin() as conn:
                conn.execute(sa.text(init_sql_text))
    base.metadata.create_all(engine)
    if letter == "s":
        session = sessionmaker(bind=engine)()
        try:
            fill_state_session(session)
            session.commit()
        finally:
            session.close()
    if letter == "r":
        session = sessionmaker(bind=engine)()
        try:
            fill_persistentdata_session(session)
            session.commit()
        finally:
            session.close()
    engine.dispose()
    if not override_db:
        assert db_path
        os.chmod(db_path, 0o666)
        if not Path(backup_path).exists():
            shutil.copy(db_path, backup_path)
    return conn_str


@pytest.fixture(scope="function")
def state_db(worker_id) -> str:
    return _init_own_db("s", StateBase, worker_id)


@pytest.fixture(scope="function")
def persistentdata_db(worker_id) -> str:
    return _init_own_db(
        "r",
        PersistentdataBase,
        worker_id,
        {
            "postgresql": "create schema if not exists athenian;",
        },
    )


@pytest.fixture(scope="function")
def precomputed_db(worker_id) -> str:
    return _init_own_db(
        "p",
        PrecomputedBase,
        worker_id,
        {
            "postgresql": (
                "create extension if not exists hstore; create schema if not exists github;"
            ),
        },
    )


async def connect_to_db(addr, event_loop, request):
    opts = {"pgbouncer_statement": True} if addr.startswith("postgresql") else {}
    db = measure_db_overhead_and_retry(Database(addr, **opts), None, None)
    try:
        await db.connect()
    except Exception:
        print(f"Unable to connect to {addr}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    def shutdown():
        event_loop.run_until_complete(db.disconnect())

    request.addfinalizer(shutdown)
    return db


@pytest.fixture(scope="function")
async def _mdb(metadata_db, event_loop, request):
    return await connect_to_db(metadata_db, event_loop, request)


@pytest.fixture(scope="function")
async def mdb(_mdb, worker_id, event_loop, request):
    if _mdb.url.dialect != "sqlite":
        return _mdb
    # check whether the database is sane
    # IDK why it happens in the CI sometimes
    while True:
        try:
            # a canary query
            await _mdb.fetch_val(select(func.count(NodeCommit.graph_id)))
            break
        except OperationalError:
            metadata_db = _metadata_db(worker_id, True)
            _mdb = await connect_to_db(metadata_db, event_loop, request)
    _mdb.is_rw = False
    return _mdb


@pytest.fixture(scope="function")
async def mdb_rw(mdb, event_loop, worker_id, request):
    if mdb.url.dialect != "sqlite":
        mdb.is_rw = True
        return mdb
    # check whether the database is locked
    # IDK why it happens in the CI sometimes
    url = "https://github.com/organizations/athenianco/settings/installations/777"
    while True:
        try:
            # a canary query
            await mdb.execute(
                insert(Account).values(
                    {
                        Account.id: 777,
                        Account.owner_id: 1,
                        Account.owner_login: "xxx",
                        Account.name: "src-d",
                        Account.install_url: url,
                        Account.created_at: datetime.now(timezone.utc),
                        Account.updated_at: datetime.now(timezone.utc),
                    },
                ),
            )
            break
        except OperationalError:
            metadata_db = _metadata_db(worker_id, True)
            mdb = await connect_to_db(metadata_db, event_loop, request)
        finally:
            await mdb.execute(delete(Account).where(Account.id == 777))
    mdb.is_rw = True
    return mdb


@pytest.fixture(scope="function")
async def sdb(state_db, event_loop, request):
    return await connect_to_db(state_db, event_loop, request)


@pytest.fixture(scope="function")
async def sdb_conn(sdb: Database) -> Connection:
    async with sdb.connection() as sdb_conn:
        yield sdb_conn


@pytest.fixture(scope="function")
async def pdb(precomputed_db, event_loop, request):
    db = await connect_to_db(precomputed_db, event_loop, request)
    db.metrics = {
        "hits": ContextVar("pdb_hits", default=defaultdict(int)),
        "misses": ContextVar("pdb_misses", default=defaultdict(int)),
    }
    return db


@pytest.fixture(scope="function")
async def rdb(persistentdata_db, event_loop, request):
    return await connect_to_db(persistentdata_db, event_loop, request)


@pytest.fixture(scope="function")
def branch_miner():
    return BranchMiner


@pytest.fixture(scope="function")
def release_loader():
    return ReleaseLoader


@pytest.fixture(scope="function")
def releases_to_prs_mapper():
    return ReleaseToPullRequestMapper


@pytest.fixture(scope="function")
def done_prs_facts_loader():
    return DonePRFactsLoader


@pytest.fixture(scope="function")
def open_prs_facts_loader():
    return OpenPRFactsLoader


@pytest.fixture(scope="function")
def merged_prs_facts_loader():
    return MergedPRFactsLoader


@pytest.fixture(scope="function")
def pr_miner():
    return PullRequestMiner


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
    parser.addoption(
        "--limit", action="store", default=-1, type=float, help="Ratio of suffled tests to run.",
    )
    parser.addoption(
        "--seed",
        action="store",
        default=0,
        type=int,
        help="Random seed to shuffle the tests, requires --limit.",
    )
    parser.addoption(
        "--offset",
        action="store",
        default=0,
        type=int,
        help="0-based index of the executed batch of tests, requires --limit.",
    )


def pytest_collection_modifyitems(session, config, items):
    offset = config.getoption("--offset")
    limit = config.getoption("--limit")
    seed = config.getoption("--seed")
    if limit >= 0:
        if seed == 0:
            seed = time.time() // 60
        random.seed(seed)
        offset = int(len(items) * limit * offset)
        limit = int(len(items) * limit) + 1
        indexes = list(range(len(items)))
        random.shuffle(indexes)
        indexes = sorted(indexes[offset : offset + limit])
        items[:] = [items[i] for i in indexes]


@pytest.fixture(scope="function")
async def sample_deployments(rdb):
    await rdb.execute(delete(DeployedLabel))
    await rdb.execute(delete(DeployedComponent))
    await rdb.execute(delete(DeploymentNotification))
    for year, month, day, conclusion, tag, commit in (
        (2019, 11, 1, "SUCCESS", "v4.13.1", 2755244),
        (2018, 12, 1, "SUCCESS", "4.8.1", 2755046),
        (2018, 12, 2, "SUCCESS", "4.8.1", 2755046),
        (2018, 8, 1, "SUCCESS", "4.5.0", 2755028),
        (2016, 12, 1, "SUCCESS", "3.2.0", 2755108),
        (2018, 1, 12, "FAILURE", "4.0.0", 2757510),
        (2018, 1, 11, "SUCCESS", "4.0.0", 2757510),
        (2018, 1, 10, "FAILURE", "4.0.0", 2757510),
        (2016, 7, 6, "SUCCESS", "3.1.0", 2756224),
    ):
        for env in ("production", "staging", "canary"):
            name = "%s_%d_%02d_%02d" % (env, year, month, day)
            await rdb.execute(
                insert(DeploymentNotification).values(
                    account_id=1,
                    name=name,
                    conclusion=conclusion,
                    environment=env,
                    started_at=datetime(year, month, day, tzinfo=timezone.utc),
                    finished_at=datetime(year, month, day, 0, 10, tzinfo=timezone.utc),
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                ),
            )
            await rdb.execute(
                insert(DeployedComponent).values(
                    account_id=1,
                    deployment_name=name,
                    repository_node_id=40550,
                    reference=tag,
                    resolved_commit_node_id=commit,
                    created_at=datetime.now(timezone.utc),
                ),
            )


@pytest.fixture(scope="session")
def release_match_setting_tag_or_branch():
    return ReleaseSettings(
        {
            "github.com/src-d/go-git": ReleaseMatchSetting(
                branches="master", tags=".*", events=".*", match=ReleaseMatch.tag_or_branch,
            ),
        },
    )


def get_release_match_setting_tag() -> ReleaseSettings:
    return ReleaseSettings(
        {
            "github.com/src-d/go-git": ReleaseMatchSetting(
                branches="", tags=".*", events=".*", match=ReleaseMatch.tag,
            ),
        },
    )


@pytest.fixture(scope="session")
def release_match_setting_tag():
    return get_release_match_setting_tag()


@pytest.fixture(scope="session")
def release_match_setting_branch():
    return ReleaseSettings(
        {
            "github.com/src-d/go-git": ReleaseMatchSetting(
                branches=default_branch_alias, tags=".*", events=".*", match=ReleaseMatch.branch,
            ),
        },
    )


@pytest.fixture(scope="session")
def release_match_setting_event():
    return ReleaseSettings(
        {
            "github.com/src-d/go-git": ReleaseMatchSetting(
                branches="", tags=".*", events=".*", match=ReleaseMatch.event,
            ),
        },
    )


@pytest.fixture(scope="session")
def release_match_setting_tag_logical(release_match_setting_tag):
    return ReleaseSettings(
        {
            **release_match_setting_tag.prefixed,
            "github.com/src-d/go-git/alpha": ReleaseMatchSetting(
                branches="", tags=".*", events=".*", match=ReleaseMatch.tag,
            ),
            "github.com/src-d/go-git/beta": ReleaseMatchSetting(
                branches="", tags=r"v4\..*", events=".*", match=ReleaseMatch.tag,
            ),
        },
    )


@pytest.fixture(scope="session")
def meta_ids() -> tuple[int, ...]:
    return (6366825,)


def get_default_branches() -> dict:
    return {
        "src-d/go-git": "master",
        "src-d/gitbase": "master",
        "src-d/hercules": "master",
    }


@pytest.fixture(scope="session")
def default_branches():
    return get_default_branches()


_branches = None


@pytest.fixture(scope="function")
async def branches(mdb, branch_miner, prefixer, meta_ids):
    # we have to hack this because mdb has to be function scope
    global _branches
    if _branches is None:
        _branches, _ = await branch_miner.load_branches(
            ["src-d/go-git"], prefixer, 1, meta_ids, mdb, None, None,
        )
    return _branches


@pytest.fixture(scope="function")
@with_defer
async def prefixer(mdb, meta_ids):
    return await Prefixer.load(meta_ids, mdb, None)


@pytest.fixture(scope="session")
def logical_settings_full():
    return LogicalRepositorySettings(
        {
            "src-d/go-git/alpha": {"title": ".*[Ff]ix"},
            "src-d/go-git/beta": {"title": ".*[Aa]dd"},
        },
        {
            "src-d/go-git/alpha": {"title": ".*(2016|2019)"},
            "src-d/go-git/beta": {"title": "prod|.*2019"},
        },
    )


_dag = None


async def fetch_dag(meta_ids, mdb, heads=None):
    if heads is None:
        heads = [
            2755363,
        ]
    edges = await _fetch_commit_history_edges(heads, [], meta_ids, mdb)
    return {"src-d/go-git": (True, join_dags(*_empty_dag(), edges))}


@pytest.fixture(scope="function")  # we cannot declare it "module" because of mdb's scope
async def dag(mdb, meta_ids):
    global _dag
    if _dag is not None:
        return _dag
    _dag = await fetch_dag(meta_ids, mdb)
    return _dag  # _dag is global # noqa: PIE781


@pytest.fixture(scope="function")
@with_defer
async def precomputed_dead_prs(mdb, pdb, branches, dag, meta_ids) -> None:
    prs = await read_sql_query(
        select(PullRequest).order_by(PullRequest.node_id),
        mdb,
        PullRequest,
        index=PullRequest.node_id,
    )
    await PullRequestMiner.mark_dead_prs(prs, branches, dag, 1, meta_ids, mdb, pdb)


@pytest.fixture(scope="function")
async def metrics_calculator_factory(mdb, pdb, rdb, cache):
    def build(account_id, meta_ids, with_cache=False, cache_only=False):
        if cache_only:
            return MetricEntriesCalculator(account_id, meta_ids, 28, None, None, None, cache)
        if with_cache:
            c = cache
        else:
            c = None

        return MetricEntriesCalculator(account_id, meta_ids, 28, mdb, pdb, rdb, c)

    return build


@pytest.fixture(scope="function")
async def pr_facts_calculator_factory(mdb, pdb, rdb, cache):
    def build(account_id, meta_ids, with_cache=False, cache_only=False) -> PRFactsCalculator:
        if cache_only:
            return PRFactsCalculator(account_id, meta_ids, None, None, None, cache=cache)
        if with_cache:
            c = cache
        else:
            c = None

        return PRFactsCalculator(account_id, meta_ids, mdb, pdb, rdb, cache=c)

    return build


@pytest.fixture(scope="function")
async def metrics_calculator_factory_memcached(mdb, pdb, rdb, memcached):
    def build(account_id, meta_ids, with_cache=False, cache_only=False):
        if cache_only:
            return MetricEntriesCalculator(account_id, meta_ids, 28, None, None, None, memcached)
        if with_cache:
            c = memcached
        else:
            c = None

        return MetricEntriesCalculator(account_id, meta_ids, 28, mdb, pdb, rdb, c)

    return build


SAMPLE_BOTS = {
    "login",
    "similar-code-searcher",
    "prettierci",
    "pull",
    "dependabot",
    "changeset-bot",
    "jira",
    "depfu",
    "codecov-io",
    "linear-app",
    "pull-assistant",
    "stale",
    "codecov",
    "sentry-io",
    "minimum-review-bot",
    "sonarcloud",
    "thehub-integration",
    "release-drafter",
    "netlify",
    "height",
    "allcontributors",
    "linc",
    "cla-checker-service",
    "unfurl-links",
    "probot-auto-merge",
    "snyk-bot",
    "slash-commands",
    "greenkeeper",
    "cypress",
    "gally-bot",
    "commitlint",
    "monocodus",
    "dependabot-preview",
    "vercel",
    "codecov-commenter",
    "botelastic",
    "renovate",
    "markdownify",
    "coveralls",
    "github-actions",
    "codeclimate",
    "zube",
}


@pytest.fixture(scope="session")
def bots() -> set[str]:
    return SAMPLE_BOTS


def generate_pr_samples(n):
    return [PullRequestFactsFactory() for _ in range(n)]


@pytest.fixture(scope="session")
def pr_samples():
    return generate_pr_samples


@pytest.fixture(scope="function")
async def logical_settings_db(sdb):
    await sdb.execute(
        insert(LogicalRepository).values(
            LogicalRepository(
                account_id=1,
                name="alpha",
                repository_id=40550,
                prs={"title": ".*[Ff]ix"},
            )
            .create_defaults()
            .explode(),
        ),
    )
    await sdb.execute(
        insert(LogicalRepository).values(
            LogicalRepository(
                account_id=1,
                name="beta",
                repository_id=40550,
                prs={"title": ".*[Aa]dd"},
            )
            .create_defaults()
            .explode(),
        ),
    )
    await sdb.execute(
        sa.update(RepositorySet)
        .where(RepositorySet.owner_id == 1)
        .values(
            {
                RepositorySet.items: [
                    ["github.com", 40550, ""],
                    ["github.com", 40550, "alpha"],
                    ["github.com", 40550, "beta"],
                    ["github.com", 39652769, ""],
                ],
                RepositorySet.updates_count: RepositorySet.updates_count + 1,
                RepositorySet.updated_at: datetime.now(timezone.utc),
            },
        ),
    )


@pytest.fixture(scope="function")
async def release_match_setting_tag_logical_db(sdb):
    await models_insert(
        sdb,
        ReleaseSettingFactory(
            logical_name="alpha",
            repo_id=40550,
            branches="master",
            match=ReleaseMatch.tag,
        ),
        ReleaseSettingFactory(
            logical_name="beta",
            repo_id=40550,
            branches="master",
            tags=r"v4\..*",
            match=ReleaseMatch.tag,
        ),
    )


@pytest.fixture(scope="function")
def no_deprecation_warnings():
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


@pytest.fixture(scope="session")
def logical_settings():
    return LogicalRepositorySettings(
        {
            "src-d/go-git/alpha": {"title": ".*[Ff]ix"},
            "src-d/go-git/beta": {"title": ".*[Aa]dd"},
        },
        {},
    )


@pytest.fixture(scope="function")
@with_defer
async def precomputed_deployments(
    release_match_setting_tag_or_branch,
    prefixer,
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
):
    await _precompute_deployments(
        release_match_setting_tag_or_branch, prefixer, branches, default_branches, mdb, pdb, rdb,
    )


@pytest.fixture(scope="function")
@with_defer
async def precomputed_sample_deployments(
    release_match_setting_tag_or_branch,
    prefixer,
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    sample_deployments,
):
    await _precompute_deployments(
        release_match_setting_tag_or_branch, prefixer, branches, default_branches, mdb, pdb, rdb,
    )


async def _precompute_deployments(
    release_match_setting_tag_or_branch,
    prefixer,
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
):
    deps = await mine_deployments(
        ["src-d/go-git"],
        {},
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        ["production", "staging"],
        [],
        {},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        branches,
        default_branches,
        prefixer,
        1,
        None,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    log = logging.getLogger(f"{package}.precomputed_deployments")
    log.info("Mined %d deployments", len(deps))
    log.info("Mined %d release deployments", sum(len(df) for df in deps["releases"]))


@pytest.fixture(scope="function")
def detect_deployments(request):
    repository_environment_threshold = deployment_light.repository_environment_threshold
    deployment_light.repository_environment_threshold = timedelta(days=100 * 365)

    def restore_repository_environment_threshold():
        deployment_light.repository_environment_threshold = repository_environment_threshold

    request.addfinalizer(restore_repository_environment_threshold)
