import datetime
from lzma import LZMAFile
import os
from pathlib import Path

import sqlalchemy.orm
from sqlalchemy.cprocessors import str_to_date, str_to_datetime

from athenian.api.models.metadata.github import Base
from athenian.api.models.state.models import RepositorySet


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
        opener = lambda: LZMAFile(data_file)
    else:
        opener = lambda: open(data_file, "rb")
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
                        ctor = lambda x: x == "t"
                    else:
                        ctor = lambda x: x
                    columns[c.name] = ctor
                keys = [p.strip(b'(),"').decode() for p in parts[1:-2]]
                continue
            if stdin:
                if line == b"\\.\n":
                    stdin = False
                    continue
                kwargs = {}
                for k, p in zip(keys, line[:-1].split(b"\t")):
                    p = p.replace(b"\\t", b"\t").replace(b"\\n", b"\n").decode()
                    if p == r"\N":
                        kwargs[k] = None
                    else:
                        kwargs[k] = columns[k](p)
                session.add(model(**kwargs))


def fill_state_session(session: sqlalchemy.orm.Session):
    session.add(RepositorySet(
        owner="github.com/vmarkovtsev",
        items=["github.com/src-d/go-git", "github.com/athenianco/athenian-api"]))
