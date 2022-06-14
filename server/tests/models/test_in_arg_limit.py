import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql, sqlite

from athenian.api.models.metadata.github import Repository


async def test_query_argument_limit_in(mdb):
    rows = await mdb.fetch_all(
        select([Repository]).where(
            Repository.full_name.in_(["r%d" % i for i in range(1 << 16)] + ["src-d/go-git"])
        )
    )
    assert rows


render_postcompile = {
    "compile_kwargs": {
        "literal_binds": True,
        "render_postcompile": True,
    },
}


@pytest.mark.parametrize("dtype", [None, object, "S", "U"])
async def test_in_inlining(dtype):
    def wrap_list(vals):
        if dtype is None:
            return vals
        else:
            return np.array(vals, dtype=dtype)

    if dtype is None:
        check_any_values = "= ANY (VALUES ('r0'), ('r1'),"
    else:
        check_any_values = "= ANY (VALUES ('r0'"
    sql = select([Repository]).where(
        Repository.full_name.in_(wrap_list(["r%d" % i for i in range(1 << 3)] + ["src-d/go-git"]))
    )
    postgres_sql = str(sql.compile(dialect=postgresql.dialect(), **render_postcompile))
    assert check_any_values not in postgres_sql
    sql = select([Repository]).where(
        Repository.full_name.in_(wrap_list(["r%d" % i for i in range(1 << 16)] + ["src-d/go-git"]))
    )
    postgres_sql = str(sql.compile(dialect=postgresql.dialect(), **render_postcompile))
    assert check_any_values not in postgres_sql
    assert "'r0'" in postgres_sql
    assert ",'r1'" in postgres_sql or ", 'r1'" in postgres_sql
    sql = select([Repository]).where(
        Repository.full_name.in_any_values(
            wrap_list(["r%d" % i for i in range(1 << 3)] + ["src-d/go-git"])
        )
    )
    postgres_sql = str(sql.compile(dialect=postgresql.dialect(), **render_postcompile))
    assert check_any_values in postgres_sql
    sql = select([Repository]).where(
        Repository.full_name.in_any_values(
            wrap_list(["r%d" % i for i in range(1 << 16)] + ["src-d/go-git"])
        )
    )
    postgres_sql = str(sql.compile(dialect=postgresql.dialect(), **render_postcompile))
    assert check_any_values in postgres_sql
    sql = select([Repository]).where(
        Repository.full_name.in_any_values(
            wrap_list(["r%d" % i for i in range(1 << 16)] + ["src-d/go-git"])
        )
    )
    postgres_sql = str(sql.compile(dialect=sqlite.dialect(), **render_postcompile))
    assert check_any_values not in postgres_sql


async def test_in_any_values_null():
    sql = select([Repository]).where(
        Repository.full_name.in_any_values(np.array([None, ""], dtype=object))
    )
    postgres_sql = str(sql.compile(dialect=postgresql.dialect(), **render_postcompile))
    assert postgres_sql.endswith("ANY (VALUES (NULL), (''))")


async def test_in_null():
    sql = select([Repository]).where(Repository.full_name.in_(np.array([None, ""], dtype=object)))
    postgres_sql = str(sql.compile(dialect=postgresql.dialect(), **render_postcompile))
    assert postgres_sql.endswith("IN (NULL, '')")
