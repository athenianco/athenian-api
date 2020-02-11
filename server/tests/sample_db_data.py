import datetime
from lzma import LZMAFile
import os
from pathlib import Path

from sqlalchemy.cprocessors import str_to_date, str_to_datetime
import sqlalchemy.orm

from athenian.api.models.metadata.github import Base
from athenian.api.models.state.models import Account, God, Invitation, RepositorySet, UserAccount


def fill_metadata_session(session: sqlalchemy.orm.Session):
    models = {}
    for cls in Base._decl_class_registry.values():
        table = getattr(cls, "__table__", None)
        if table is not None:
            models[table.fullname] = cls
    data_file = os.getenv("DB_DATA")
    if data_file is None:
        data_file = Path(__file__).with_name("test_data.sql.xz")
    else:
        data_file = Path(data_file)
    if data_file.suffix == ".xz":
        opener = lambda: LZMAFile(data_file)  # noqa:E731
    else:
        opener = lambda: open(data_file, "rb")  # noqa:E731
    with opener() as fin:
        stdin = False
        for line in fin:
            if not stdin and line.startswith(b"COPY "):
                stdin = True
                parts = line[5:].split(b" ")
                table = parts[0].decode()
                if table.startswith("public."):
                    table = table[7:]
                model = models[table]
                columns = {}
                for c in Base.metadata.tables[table].columns:
                    pt = c.type.python_type
                    if pt is datetime.date:
                        ctor = str_to_date
                    elif pt is datetime.datetime:
                        ctor = str_to_datetime
                    elif pt is bool:
                        ctor = lambda x: x == "t"  # noqa:E731
                    else:
                        ctor = lambda x: x  # noqa:E731
                    columns[c.name] = ctor
                keys = [p.strip(b'(),"').decode() for p in parts[1:-2]]
                continue
            if stdin:
                if line == b"\\.\n":
                    stdin = False
                    continue
                kwargs = {}
                vals = line[:-1].split(b"\t")
                for k, p in zip(keys, vals):
                    p = p.replace(b"\\t", b"\t").replace(b"\\n", b"\n").decode()
                    if p == r"\N":
                        kwargs[k] = None
                    else:
                        try:
                            kwargs[k] = columns[k](p)
                        except Exception as e:
                            print("%s.%s" % (table, k), p)
                            for k, p in zip(keys, vals):
                                print(k, '"%s"' % p.decode())
                            raise e from None
                session.add(model(**kwargs))


def fill_state_session(session: sqlalchemy.orm.Session):
    session.add(Account(id=1))
    session.add(Account(id=2))
    session.add(Account(id=3))
    session.add(UserAccount(
        user_id="auth0|5e1f6dfb57bc640ea390557b", account_id=1, is_admin=True))
    session.add(UserAccount(
        user_id="auth0|5e1f6dfb57bc640ea390557b", account_id=2, is_admin=False))
    session.add(UserAccount(
        user_id="auth0|5e1f6e2e8bfa520ea5290741", account_id=3, is_admin=True))
    session.add(UserAccount(
        user_id="auth0|5e1f6e2e8bfa520ea5290741", account_id=1, is_admin=False))
    session.add(RepositorySet(
        owner=1,
        items=["github.com/src-d/go-git", "github.com/athenianco/athenian-api"]))
    session.add(RepositorySet(
        owner=2,
        items=["github.com/src-d/hercules", "github.com/athenianco/athenian-api"]))
    session.add(RepositorySet(
        owner=3,
        items=["github.com/athenianco/athenian-webapp", "github.com/athenianco/athenian-api"]))
    session.add(Invitation(salt=777, account_id=3, created_by="auth0|5e1f6e2e8bfa520ea5290741"))
    session.add(God(user_id="auth0|5e1f6dfb57bc640ea390557b"))
