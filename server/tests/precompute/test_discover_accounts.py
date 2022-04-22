from argparse import Namespace
from datetime import datetime, timezone
from functools import partialmethod
import logging
from typing import Any, Iterator
from unittest import mock

import pytest
from sqlalchemy import delete, insert, update
import tqdm

from athenian.api.db import Database
from athenian.api.models.metadata.github import Account as MetaAccount, \
    AccountRepository as MetaAccountRepository, FetchProgress
from athenian.api.models.state.models import Account, AccountFeature, AccountGitHubAccount, \
    AccountJiraInstallation, Invitation, RepositorySet, UserAccount, WorkType
from athenian.api.precompute.context import PrecomputeContext
from athenian.api.precompute.discover_accounts import main
from tests.testutils.db import model_insert_stmt
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.state import AccountFactory, RepositorySetFactory


@pytest.fixture
def tqdm_disable() -> Iterator[None]:
    orig_init = tqdm.tqdm.__init__
    tqdm.tqdm.__init__ = partialmethod(orig_init, disable=True)
    yield
    tqdm.tqdm.__init__ = orig_init


async def test_no_accounts(sdb, mdb_rw, tqdm_disable) -> None:
    await _clear_all_accounts(sdb)
    ctx = _build_context(sdb=sdb, mdb=mdb_rw)
    accounts = await main(ctx, Namespace(partition=False))
    assert accounts == []


async def test_some_accounts(sdb, mdb_rw, tqdm_disable) -> None:
    await _clear_all_accounts(sdb)
    await sdb.execute(model_insert_stmt(AccountFactory(id=11)))

    try:
        await _make_installed_account(sdb, mdb_rw, 10, 110)
        ctx = _build_context(sdb=sdb, mdb=mdb_rw)
        accounts = await main(ctx, Namespace(partition=False))
    finally:
        await _clear_account_from_meta(mdb_rw, 110)

    # account 10 is installed, account 12 not
    assert accounts == [10]


async def test_partition(sdb, mdb_rw, tqdm_disable) -> None:
    await _clear_all_accounts(sdb)
    await sdb.execute(model_insert_stmt(AccountFactory(id=10)))

    try:
        await _make_installed_account(sdb, mdb_rw, 11, 111)
        await _make_installed_account(sdb, mdb_rw, 12, 112)
        await _make_installed_account(sdb, mdb_rw, 13, 113)
        # make all reposets precomputed for account 13
        update_stmt = update(RepositorySet).where(RepositorySet.owner_id == 13).values(
            precomputed=True, updated_at=datetime.now(timezone.utc), updates_count=1,
        )
        await sdb.execute(update_stmt)
        ctx = _build_context(sdb=sdb, mdb=mdb_rw)
        accounts = await main(ctx, Namespace(partition=True))
    finally:
        await _clear_account_from_meta(mdb_rw, 111)
        await _clear_account_from_meta(mdb_rw, 112)
        await _clear_account_from_meta(mdb_rw, 113)

    # account 13 is "precomputed", 11 and 12 are installed but fresh, 10 is not installed
    assert accounts == {"fresh": [11, 12], "precomputed": [13]}


async def _make_installed_account(sdb, mdb, account_id, meta_acc_id) -> None:
    """Create DB objects so that account_id is fully installed."""

    for state_model in (
        AccountFactory(id=account_id),
        RepositorySetFactory(owner_id=account_id),
    ):
        await sdb.execute(model_insert_stmt(state_model))
    await sdb.execute(insert(AccountGitHubAccount).values(id=meta_acc_id, account_id=account_id))

    for md_model in (
        md_factory.AccountFactory(id=meta_acc_id),
        md_factory.AccountRepositoryFactory(acc_id=meta_acc_id, event_id="repo-event-00"),
        md_factory.FetchProgressFactory(
            event_id="repo-event-00", acc_id=meta_acc_id, nodes_total=100, nodes_processed=100,
        ),
    ):
        await mdb.execute(model_insert_stmt(md_model))


async def _clear_account_from_meta(mdb, meta_acc_id) -> None:
    """Drop from metadata DB objects relative to meta_acc_id."""
    await mdb.execute(
        delete(MetaAccountRepository).where(MetaAccountRepository.acc_id == meta_acc_id),
    )
    await mdb.execute(delete(MetaAccount).where(MetaAccount.id == meta_acc_id))
    await mdb.execute(delete(FetchProgress).where(FetchProgress.acc_id == meta_acc_id))


async def _clear_all_accounts(sdb: Database) -> None:
    """Drop from state DB all accounts and related objects."""
    for model in (
        RepositorySet,
        UserAccount,
        AccountGitHubAccount,
        AccountJiraInstallation,
        AccountFeature,
        WorkType,
        Invitation,
        Account,
    ):
        await sdb.execute(delete(model))


def _build_context(**kwargs: Any) -> PrecomputeContext:
    kwargs.setdefault("log", logging.getLogger(__name__))
    for db_field in ("sdb", "mdb", "pdb", "rdb"):
        if db_field not in kwargs:
            kwargs[db_field] = mock.Mock(Database)
    kwargs.setdefault("cache", None)
    kwargs.setdefault("slack", None)
    return PrecomputeContext(**kwargs)
