from athenian.api.db import conn_in_transaction, Database


class TestConnInTransaction:
    async def test_inside_transaction(self, state_db: str) -> None:
        async with Database(state_db) as db:
            async with db.connection() as conn:
                async with conn.transaction():
                    assert conn_in_transaction(conn)

    async def test_outside_transaction(self, state_db: str) -> None:
        async with Database(state_db) as db:
            async with db.connection() as conn:
                assert not conn_in_transaction(conn)
