import os
import subprocess
import sys
import tempfile

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from athenian.api.db import extract_registered_models
from athenian.api.models.state.models import Base


def test_migrations(worker_id):
    with tempfile.TemporaryDirectory() as tmpdir:
        cs = "sqlite:///%s/sdb-%s.sqlite" % (tmpdir, worker_id)
        env = {**os.environ, "PYTHONPATH": os.getcwd(), "ATHENIAN_INVITATION_KEY": "vadim"}
        try:
            subprocess.run(
                [sys.executable, "-m", "athenian.api.models.state", cs],
                capture_output=True,
                check=True,
                cwd=tmpdir,
                env=env,
            )
        except subprocess.CalledProcessError as err:
            raise RuntimeError(err.stderr.decode("utf8")) from err

        engine = create_engine(cs)
        session = sessionmaker(bind=engine)()
        try:
            for model in extract_registered_models(Base).values():
                try:
                    model.__name__, model.__tablename__
                except AttributeError:
                    continue
                if model.__table__.info.get("test", False):
                    continue
                session.query(model).first()
        finally:
            session.close()
