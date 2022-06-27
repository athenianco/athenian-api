from argparse import Namespace
from unittest import mock

from freezegun import freeze_time
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.models.state.models import AccountGitHubAccount
from athenian.api.precompute.notify_almost_expired_accounts import main
from tests.testutils.db import models_insert
from tests.testutils.factory.state import (
    AccountFactory,
    AccountGitHubAccountFactory,
    UserAccountFactory,
)
from tests.testutils.time import dt

from .conftest import build_context


class TestMain:
    @freeze_time("2022-01-03T15:00:00")
    async def test_base(self, sdb: Database, mdb: Database) -> None:
        await sdb.execute(sa.delete(AccountGitHubAccount))
        await models_insert(
            sdb,
            AccountFactory(id=5, expires_at=dt(2022, 1, 4, 14, 30)),
            UserAccountFactory(user_id="u500", account_id=5),
            AccountGitHubAccountFactory(id=6366825, account_id=5),  # <- id found in mdb fixture
            AccountFactory(id=6, expires_at=dt(2022, 1, 4, 18, 0)),
        )
        slack_mock = self._slack_mock()
        ctx = build_context(sdb=sdb, mdb=mdb, slack=slack_mock)
        await main(ctx, Namespace())

        slack_mock.post_account.assert_called_once_with(
            "almost_expired.jinja2",
            account=5,
            name="athenianco",
            user="u500",
            expires=dt(2022, 1, 4, 14, 30),
        )

    @freeze_time("2022-01-03T15:00:00")
    async def test_account_not_found_in_mdb(self, sdb, mdb) -> None:
        await sdb.execute(sa.delete(AccountGitHubAccount))
        await models_insert(
            sdb,
            AccountFactory(id=5, expires_at=dt(2022, 1, 4, 14, 30)),
            UserAccountFactory(user_id="u500", account_id=5),
            AccountGitHubAccountFactory(id=9890, account_id=5),  # <- id not found in mdb fixture
        )
        slack_mock = self._slack_mock()
        ctx = build_context(sdb=sdb, mdb=mdb, slack=slack_mock)
        await main(ctx, Namespace())

        slack_mock.post_account.assert_called_once_with(
            "almost_expired.jinja2",
            account=5,
            name="<uninstalled>",
            user="u500",
            expires=dt(2022, 1, 4, 14, 30),
        )

    @classmethod
    def _slack_mock(cls) -> mock.Mock:
        slack_mock = mock.Mock(spec=SlackWebClient)
        slack_mock.post_account = mock.AsyncMock()
        return slack_mock
