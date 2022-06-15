from athenian.api.db import Database, conn_in_transaction


class TestConnInTransaction:
    async def test_inside_transaction(self, state_db: str) -> None:
        async with Database(state_db) as db:
            async with db.connection() as conn:
                async with conn.transaction():
                    assert await conn_in_transaction(conn)

    async def test_outside_transaction(self, state_db: str) -> None:
        async with Database(state_db) as db:
            async with db.connection() as conn:
                assert not await conn_in_transaction(conn)
