import sys
import traceback

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from athenian.api.models.metadata.github import Base


def main():
    """Try to fetch one example of each model in the metadata schema from a real DB instance."""
    engine = create_engine(sys.argv[1])
    print("Checking the metadata schema...", flush=True)
    errors = []
    for model in Base._decl_class_registry.values():
        session = sessionmaker(bind=engine)()
        try:
            try:
                model.__name__, model.__tablename__
            except AttributeError:
                continue
            try:
                session.query(model).first()
            except Exception:
                errors.append((model, traceback.format_exc()))
                status = "❌"
            else:
                status = "✔️"
            print("%s  %s / %s" % (status, model.__name__, model.__tablename__), flush=True)
        finally:
            session.close()
    for model, exc in errors:
        print("=" * 80, file=sys.stderr)
        print("%s / %s\n" % (model.__name__, model.__tablename__), file=sys.stderr)
        print(exc, file=sys.stderr)
        print(file=sys.stderr)


if __name__ == "__main__":
    exit(main())
