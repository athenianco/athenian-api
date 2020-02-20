import databases

from athenian.api.controllers import invitation_controller
from athenian.api.invite_admin import main


async def test_reset_sequence(state_db):
    main(state_db)
    db = databases.Database(state_db)
    await db.connect()
    assert await invitation_controller.create_new_account(db) == 4
