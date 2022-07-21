from athenian.api.internal.miners.github.bots import bots


class TestBots:
    async def test_call(self, mdb, sdb):
        bs = await bots(1, (6366825,), mdb, sdb, None)
        assert "codecov" in bs
        assert "dependabot" in bs
        assert "coveralls" in bs
