from athenian.api.db import Database
from athenian.api.internal.account import get_multiple_metadata_account_ids
from tests.testutils.db import models_insert
from tests.testutils.factory.state import AccountFactory, AccountGitHubAccountFactory


class TestGetMultipleMetadataMetadataIds:
    async def test(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            AccountFactory(id=10),
            AccountFactory(id=11),
            AccountFactory(id=12),
            AccountGitHubAccountFactory(account_id=10, id=20),
            AccountGitHubAccountFactory(account_id=10, id=21),
            AccountGitHubAccountFactory(account_id=11, id=22),
        )

        accounts_meta_ids = await get_multiple_metadata_account_ids([10, 11, 12, 13], sdb, None)

        assert sorted(accounts_meta_ids[10]) == [20, 21]
        assert sorted(accounts_meta_ids[11]) == [22]
        assert sorted(accounts_meta_ids[12]) == []
        assert sorted(accounts_meta_ids[13]) == []
