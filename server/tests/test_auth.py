from aiohttp.web_runner import GracefulExit
import pytest

from athenian.api import AthenianApp, Auth0


class DefaultAuth0(Auth0):
    def __init__(self, whitelist):
        super().__init__(whitelist=whitelist, default_user="github|60340680")


async def test_default_user(loop):
    auth0 = Auth0(default_user="github|60340680")
    user = await auth0.default_user()
    assert user.name == "Groundskeeper Willie"
    auth0._default_user = None
    auth0._default_user_id = "abracadabra"
    with pytest.raises(GracefulExit):
        await auth0.default_user()
