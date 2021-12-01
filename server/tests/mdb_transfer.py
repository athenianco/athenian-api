#!/usr/bin/python3
import json
import sys

from sqlalchemy import ARRAY, create_engine, JSON
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import elements
from tqdm import tqdm

from athenian.api.db import extract_registered_models
from athenian.api.models.metadata.github import Base


def main():
    engine = create_engine(sys.argv[1])
    session = sessionmaker(bind=engine)()
    print("Dumping the models...", file=sys.stderr)
    json_col = JSON()
    dialect = sqlite.dialect()
    type(dialect.type_descriptor(json_col)).literal_processor = lambda self, _: json.dumps
    for model in tqdm(list(extract_registered_models(Base).values()), file=sys.stderr):
        try:
            model.__name__, model.__tablename__
        except AttributeError:
            continue
        columns = model.__table__.columns
        compilers = {}
        print("COPY public.%s (%s) FROM stdin;" % (
            model.__table__.fullname, ", ".join([c.name for c in columns])))
        for obj in session.query(model):
            vals = []
            for col in columns:
                ct = col.type
                if isinstance(ct, ARRAY):
                    ct = json_col
                val = getattr(obj, col.name)
                if val is None:
                    vals.append(r"\N")
                    continue
                try:
                    compiler = compilers[col.name]
                except KeyError:
                    bindparam = elements.BindParameter(
                        col.name, getattr(obj, col.name), type_=ct, required=True,
                    )
                    compilers[col.name] = compiler = bindparam.compile(None, dialect=dialect)
                rendered = compiler.render_literal_value(val, ct)
                rendered = rendered.replace("\t", r"\t").replace("\n", r"\n").strip("'")
                vals.append(rendered)
            print("\t".join(vals))
        print("\\.\n")


if __name__ == "__main__":
    exit(main())
