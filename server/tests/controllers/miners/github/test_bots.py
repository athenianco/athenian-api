from athenian.api.controllers.miners.github.bots import bots


async def test_bots_fetch(mdb):
    bs = await bots(mdb)
    assert "codecov" in bs
    assert "dependabot" in bs
