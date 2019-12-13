import datetime
from lzma import LZMAFile
from pathlib import Path

import sqlalchemy.orm
from sqlalchemy.cprocessors import str_to_date, str_to_datetime

from athenian.api.models.db.github import Base


def fill_session(session: sqlalchemy.orm.Session):
    models = {}
    for cls in Base._decl_class_registry.values():
        table = getattr(cls, "__table__", None)
        if table is not None:
            models[table.fullname] = cls
    with LZMAFile(Path(__file__).with_name("test_data.sql.xz")) as fin:
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
