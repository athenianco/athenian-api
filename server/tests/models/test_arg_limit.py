from sqlalchemy import select

from athenian.api.models.metadata.github import Repository


async def test_query_argument_limit(mdb):
    rows = await mdb.fetch_all(
        select([Repository]).where(Repository.full_name.in_(
            ["r%d" % i for i in range(1 << 16)] + ["src-d/go-git"])))
    assert rows
