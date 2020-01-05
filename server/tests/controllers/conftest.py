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
from athenian.api.auth import User
from athenian.api.models.metadata import hack_sqlite_arrays
from athenian.api.models.metadata.github import Base as MetadataBase
from athenian.api.models.state.models import Base as StateBase
from tests.sample_db_data import fill_metadata_session, fill_state_session


db_dir = Path(os.getenv("DB_DIR", os.path.dirname(os.path.dirname(__file__))))


class FakeAuth0:
    @staticmethod
    def ensure_static_configuration():
        pass

    def __init__(self, whitelist):
        self.whitelist = whitelist

    def _is_whitelisted(self, request: aiohttp.web.Request) -> bool:
        for pattern in self.whitelist:
            if re.match(pattern, request.path):
                return True
        return False

    @aiohttp.web.middleware
    async def middleware(self, request, handler):
        """Middleware function compatible with aiohttp."""
        if self._is_whitelisted(request):
            return await handler(request)
        request.user = User(nickname="vmarkovtsev", name="Vadim Markovtsev",
                            picture="", updated_at=str(datetime.utcnow()))
        return await handler(request)


@pytest.fixture
def client(loop, aiohttp_client, metadata_db, state_db):
    logging.getLogger("connexion.operation").setLevel("ERROR")
    app = AthenianApp(mdb_conn=metadata_db, sdb_conn=state_db, ui=False, auth0_cls=FakeAuth0)
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
