import sqlalchemy.orm

from athenian.api.models.db.github import Repository


def fill_session(session: sqlalchemy.orm.Session):
    session.add(Repository(sum256="7" * 64, id=1, name="src-d/hercules", language="Go",
                           owner_id=1, owner_login="vmarkovtsev", owner_type="cool",
                           topics='["git", "mloncode"]'))