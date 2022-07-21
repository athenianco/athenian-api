from athenian.api.internal.miners.github.bots import bots


class TestBots:
    async def test_call(self, mdb, sdb):
        bs = await bots(1, (6366825,), mdb, sdb, None)
        assert "codecov" in bs
        assert "dependabot" in bs
        assert "coveralls" in bs

    async def test_get_account_bots(self, mdb, sdb) -> None:
        account_bots = await bots.get_account_bots(1, (6366825,), mdb, sdb, None)
        all_bots = await bots(1, (6366825,), mdb, sdb, None)

        assert not (account_bots.global_bots & account_bots.local_bots)
        assert account_bots.all_bots == all_bots
        assert (account_bots.global_bots | account_bots.local_bots) == all_bots
        assert "coveralls" in account_bots.global_bots
        assert "greenkeeper" in account_bots.local_bots
