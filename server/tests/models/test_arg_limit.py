from sqlalchemy import select
from sqlalchemy.dialects import postgresql, sqlite

from athenian.api.models.metadata.github import Repository


async def test_query_argument_limit_in(mdb):
    rows = await mdb.fetch_all(select([Repository]).where(Repository.full_name.in_(
        ["r%d" % i for i in range(1 << 16)] + ["src-d/go-git"])))
    assert rows


async def test_in_inlining():
    check_any_values = "= ANY (VALUES ('r0'), ('r1'),"
    sql = select([Repository]).where(Repository.full_name.in_(
        ["r%d" % i for i in range(1 << 3)] + ["src-d/go-git"]))
    postgres_sql = str(sql.compile(dialect=postgresql.dialect()))
    assert check_any_values not in postgres_sql
    sql = select([Repository]).where(Repository.full_name.in_(
        ["r%d" % i for i in range(1 << 16)] + ["src-d/go-git"]))
    postgres_sql = str(sql.compile(dialect=postgresql.dialect()))
    assert check_any_values not in postgres_sql
    sql = select([Repository]).where(Repository.full_name.in_any_values(
        ["r%d" % i for i in range(1 << 3)] + ["src-d/go-git"]))
    postgres_sql = str(sql.compile(dialect=postgresql.dialect()))
    assert check_any_values not in postgres_sql
    sql = select([Repository]).where(Repository.full_name.in_any_values(
        ["r%d" % i for i in range(1 << 16)] + ["src-d/go-git"]))
    postgres_sql = str(sql.compile(dialect=postgresql.dialect()))
    assert check_any_values in postgres_sql
    sql = select([Repository]).where(Repository.full_name.in_any_values(
        ["r%d" % i for i in range(1 << 16)] + ["src-d/go-git"]))
    postgres_sql = str(sql.compile(dialect=sqlite.dialect()))
    assert check_any_values not in postgres_sql
