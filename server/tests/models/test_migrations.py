import subprocess
import sys
import tempfile

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from athenian.api.models.state.models import Base


def test_migrations():
    with tempfile.TemporaryDirectory() as tmpdir:
        cs = "sqlite:///%s/sdb.sqlite" % tmpdir
        subprocess.run([sys.executable, "-m", "athenian.api.models.state", cs],
                       check=True, cwd=tmpdir)
        engine = create_engine(cs)
        session = sessionmaker(bind=engine)()
        try:
            for model in Base._decl_class_registry.values():
                try:
                    model.__name__, model.__tablename__
                except AttributeError:
                    continue
                session.query(model).first()
        finally:
            session.close()
