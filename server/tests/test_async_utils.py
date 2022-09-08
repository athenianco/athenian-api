import numpy as np
from numpy.testing import assert_array_equal
import sqlalchemy as sa
from sqlalchemy import BigInteger, Column, Integer, insert, select

from athenian.api.async_utils import read_sql_query
from athenian.api.db import Database
from athenian.api.models.state.models import Base
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory


class NullsModel(Base):
    __tablename__ = "int_nulls_test"
    __table_args__ = {"info": {"test": True}}

    id = Column(BigInteger, primary_key=True)
    int_col = Column(Integer)


async def test_erase_integer_nulls(sdb):
    await sdb.execute(insert(NullsModel).values({"id": 1, "int_col": None}))
    df = await read_sql_query(select([NullsModel]), sdb, NullsModel)
    assert len(df) == 1
    df = await read_sql_query(
        select([NullsModel]),
        sdb,
        [
            NullsModel.id,
            Column(Integer, name="int_col", nullable=False, info={"erase_nulls": True}),
        ],
    )
    assert len(df) == 0


async def test_reset_integer_nulls(sdb):
    await sdb.execute(insert(NullsModel).values({"id": 1, "int_col": None}))
    df = await read_sql_query(select([NullsModel]), sdb, NullsModel)
    assert len(df) == 1
    df = await read_sql_query(
        select([NullsModel]),
        sdb,
        [
            NullsModel.id,
            Column(Integer, name="int_col", nullable=False, info={"reset_nulls": True}),
        ],
    )
    assert len(df) == 1
    assert df.iloc[0]["int_col"] == 0


class TestReadSQLQuery:
    async def test_table_alias(self, mdb_rw: Database) -> None:
        from athenian.api.models.metadata.jira import Issue

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.JIRAIssueFactory(id="1", project_id="100"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            _issue = sa.orm.aliased(Issue, name="issue_alias")
            df_alias = await read_sql_query(
                sa.select(_issue.project_id), mdb_rw, columns=[_issue.project_id],
            )

            df_non_alias = await read_sql_query(
                sa.select(Issue.project_id), mdb_rw, columns=[Issue.project_id],
            )
        assert df_alias.project_id.values.dtype.type == np.dtype("S").type
        assert df_non_alias.project_id.values.dtype.type == np.dtype("S").type

        assert_array_equal(df_alias, df_non_alias)

    async def test_column_label(self, mdb_rw: Database) -> None:
        from athenian.api.models.metadata.jira import Issue

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.JIRAIssueFactory(id="1", project_id="100"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            stmt = sa.select(Issue.project_id.label("foo")).where(Issue.id == "1")
            df = await read_sql_query(stmt, mdb_rw, columns=[Issue.project_id.label("foo")])

        assert list(df.columns) == ["foo"]
        assert list(df.foo.values) == [b"100"]

    async def test_nullable_bytes(self, mdb_rw: Database) -> None:
        from athenian.api.models.metadata.jira import Issue

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.JIRAIssueFactory(id="1", priority_id=None),
                md_factory.JIRAIssueFactory(id="2", priority_id="100"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            stmt = sa.select(Issue.priority_id).where(Issue.id.in_(["1", "2"])).order_by(Issue.id)
            df = await read_sql_query(stmt, mdb_rw, columns=[Issue.priority_id])

        assert list(df.columns) == ["priority_id"]
        assert_array_equal(df.priority_id.values, np.array([b"", b"100"], dtype="S"))
