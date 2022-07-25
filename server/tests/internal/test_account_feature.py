from athenian.api.db import Database
from athenian.api.internal.account_feature import is_feature_enabled
from tests.testutils.db import models_insert
from tests.testutils.factory.state import AccountFeatureFactory, FeatureFactory


class TestFeatureEnabled:
    async def test_not_existing(self, sdb: Database) -> None:
        assert not await is_feature_enabled(1, "FT", sdb)

    async def test_default_enabled(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            FeatureFactory(name="FT0", id=100, enabled=True),
            AccountFeatureFactory(account_id=1, feature_id=100, enabled=False),
            AccountFeatureFactory(account_id=2, feature_id=100, enabled=True),
        )

        for account_id, expected in ((1, False), (2, True), (3, True)):
            assert (await is_feature_enabled(account_id, "FT0", sdb)) is expected

    async def test_default_disabled(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            FeatureFactory(name="FT0", id=100, enabled=False),
            AccountFeatureFactory(account_id=1, feature_id=100, enabled=False),
            AccountFeatureFactory(account_id=2, feature_id=100, enabled=True),
        )

        for account_id, expected in ((1, False), (2, True), (3, False)):
            assert (await is_feature_enabled(account_id, "FT0", sdb)) is expected
