from aiohttp.web_runner import GracefulExit
import pytest

from athenian.api import AthenianApp


async def test_default_user(metadata_db, state_db, loop):
    app = AthenianApp(mdb_conn=metadata_db, sdb_conn=state_db, ui=False)
    user = await app.auth0.default_user()
    assert user.name == "Groundskeeper Willie"
    app.auth0._default_user = None
    app.auth0._default_user_id = "abracadabra"
    with pytest.raises(GracefulExit):
        await app.auth0.default_user()
