from athenian.api.controllers.miners.github.bots import bots


async def test_bots_fetch(mdb, sdb):
    bs = await bots(1, (6366825,), mdb, sdb, None)
    assert "codecov" in bs
    assert "dependabot" in bs
    assert "coveralls" in bs
