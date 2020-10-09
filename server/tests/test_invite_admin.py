import os

import databases
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from athenian.api.controllers import invitation_controller
from athenian.api.invite_admin import main as main_invite
from athenian.api.models import migrate
from athenian.api.models.state.models import Base
from tests.sample_db_data import fill_state_session


async def test_reset_sequence(state_db):
    engine = create_engine(state_db)
    Base.metadata.drop_all(engine)
    os.putenv("ATHENIAN_INVITATION_KEY", "whatever")
    migrate("state", url=state_db, exec=False)
    session = sessionmaker(bind=engine)()
    try:
        fill_state_session(session)
        session.commit()
    finally:
        session.close()
    main_invite(state_db)
    db = databases.Database(state_db)
    await db.connect()
    assert await invitation_controller.create_new_account(db) == 4
