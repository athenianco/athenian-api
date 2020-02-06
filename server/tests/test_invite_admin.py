import databases

from athenian.api.controllers import invitation_controller
from athenian.api.invite_admin import main


async def test_reset_sequence(state_db):
    main(state_db)
    db = databases.Database(state_db)
    await db.connect()
    if db.url.dialect == "sqlite":
        assert await invitation_controller._create_new_account_slow(db) == 4
    else:
        assert await invitation_controller._create_new_account_fast(db) == 4
