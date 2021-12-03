import morcilla
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from athenian.api.controllers import invitation_controller
from athenian.api.invite_admin import main as main_invite
from athenian.api.models import migrate
from athenian.api.models.state.models import Base
from tests.sample_db_data import fill_state_session


async def test_reset_sequence(state_db, locked_migrations):
    with locked_migrations:
        await _test_reset_sequence(state_db)


async def _test_reset_sequence(state_db):
    sqla_conn_str = state_db.rsplit("?", 1)[0]
    engine = create_engine(sqla_conn_str)
    Base.metadata.drop_all(engine)
    migrate("state", url=sqla_conn_str, exec=False)
    session = sessionmaker(bind=engine)()
    try:
        fill_state_session(session)
        session.commit()
    finally:
        session.close()
    main_invite(sqla_conn_str)
    db = morcilla.Database(state_db)
    await db.connect()
    assert await invitation_controller.create_new_account(db, "whatever") == 4
