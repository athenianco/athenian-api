from sqlalchemy import BigInteger, Column, insert, Integer, select

from athenian.api.async_utils import read_sql_query
from athenian.api.models.state.models import Base


class NullsModel(Base):
    __tablename__ = "erase_nulls_test"
    __table_args__ = {"info": {"test": True}}

    id = Column(BigInteger, primary_key=True)
    int_col = Column(Integer)


async def test_erase_integer_nulls(sdb):
    await sdb.execute(insert(NullsModel).values({"id": 1, "int_col": None}))
    df = await read_sql_query(select([NullsModel]), sdb, NullsModel)
    assert len(df) == 1
    df = await read_sql_query(select([NullsModel]), sdb, [
        NullsModel.id, Column(Integer, name="int_col", nullable=False, info={"erase_nulls": True}),
    ])
    assert len(df) == 0
