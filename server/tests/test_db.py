from datetime import datetime, timezone

from athenian.api.db import Database, conn_in_transaction, ensure_db_datetime_tz


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


class TestEnsureDBDatetimeTZ:
    async def test_sqlite(self, state_db) -> None:
        dt = datetime(2002, 10, 23)
        async with Database("sqlite:////") as db:
            assert ensure_db_datetime_tz(dt, db) == datetime(2002, 10, 23).replace(
                tzinfo=timezone.utc,
            )
