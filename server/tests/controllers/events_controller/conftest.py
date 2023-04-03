import pytest
import sqlalchemy as sa

from athenian.api.models.state.models import UserToken


@pytest.fixture(scope="function")
async def token(sdb):
    await sdb.execute(
        sa.insert(UserToken).values(
            UserToken(account_id=1, user_id="auth0|62a1ae88b6bba16c6dbc6870", name="xxx")
            .create_defaults()
            .explode(),
        ),
    )
    return "AQAAAAAAAAA="  # unencrypted
