from pathlib import Path

from athenian.precomputer import db

template = Path(db.__file__).with_name("alembic.ini.mako")
del db
