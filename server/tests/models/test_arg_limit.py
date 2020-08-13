from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from athenian.api.models.metadata.github import Repository


async def test_query_argument_limit(mdb):
    sql = select([Repository]).where(Repository.full_name.in_(
        ["r%d" % i for i in range(1 << 3)] + ["src-d/go-git"]))
    postgres_sql = str(sql.compile(dialect=postgresql.dialect()))
    assert "= ANY (VALUES  ('r0'), ('r1')," not in postgres_sql
    sql = select([Repository]).where(Repository.full_name.in_(
        ["r%d" % i for i in range(1 << 16)] + ["src-d/go-git"]))
    postgres_sql = str(sql.compile(dialect=postgresql.dialect()))
    assert "= ANY (VALUES  ('r0'), ('r1')," in postgres_sql
    rows = await mdb.fetch_all(sql)
    assert rows
