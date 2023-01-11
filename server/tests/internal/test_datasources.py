import pytest

from athenian.api.db import Database
from athenian.api.internal.datasources import AccountDatasources


class TestAccountDatasources:
    def test_used_as_container(self) -> None:
        datasources = AccountDatasources((AccountDatasources.GITHUB,))
        assert AccountDatasources.GITHUB in datasources
        assert AccountDatasources.JIRA not in datasources
        assert "foobar" not in datasources

    def test_invalid_datasources(self) -> None:
        with pytest.raises(ValueError):
            AccountDatasources(["foo"])

        with pytest.raises(ValueError):
            AccountDatasources([AccountDatasources.JIRA, "bar"])

    async def test_build_for_account_jira_available(self, sdb: Database) -> None:
        datasources = await AccountDatasources.build_for_account(1, sdb)
        assert AccountDatasources.GITHUB in datasources
        assert AccountDatasources.JIRA in datasources

    async def test_build_for_account_jira_not_available(self, sdb: Database) -> None:
        datasources = await AccountDatasources.build_for_account(2, sdb)
        assert AccountDatasources.GITHUB in datasources
        assert AccountDatasources.JIRA not in datasources
