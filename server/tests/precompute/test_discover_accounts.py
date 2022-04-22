from argparse import Namespace
from functools import partialmethod
import logging
from typing import Any, Iterator
from unittest import mock

import pytest
from sqlalchemy import delete, insert
import tqdm

from athenian.api.db import Database
from athenian.api.models.metadata.github import Account as MetaAccount, \
    AccountRepository as MetaAccountRepository, FetchProgress
from athenian.api.models.state.models import Account, AccountGitHubAccount
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


async def test_no_accounts(sdb, tqdm_disable) -> None:
    await sdb.execute(delete(Account))
    ctx = _build_context(sdb=sdb)
    accounts = await main(ctx, Namespace())
    assert accounts == []


async def test_some_accounts(sdb, mdb_rw, tqdm_disable, request) -> None:
    await sdb.execute(delete(Account))
    await sdb.execute(model_insert_stmt(AccountFactory(id=10)))
    await sdb.execute(model_insert_stmt(AccountFactory(id=11)))

    await sdb.execute(insert(AccountGitHubAccount).values(id=110, account_id=10))
    await sdb.execute(model_insert_stmt(RepositorySetFactory(owner_id=10)))

    try:
        await mdb_rw.execute(model_insert_stmt(
            md_factory.AccountRepositoryFactory(acc_id=110, event_id="repo-event-00"),
        ))
        await mdb_rw.execute(model_insert_stmt(md_factory.AccountFactory(id=110)))
        await mdb_rw.execute(model_insert_stmt(md_factory.FetchProgressFactory(
            event_id="repo-event-00", acc_id=110, nodes_total=100, nodes_processed=100,
        )))
        ctx = _build_context(sdb=sdb, mdb=mdb_rw)
        accounts = await main(ctx, Namespace())
    finally:
        await mdb_rw.execute(
            delete(MetaAccountRepository).where(MetaAccountRepository.acc_id == 110),
        )
        await mdb_rw.execute(delete(MetaAccount).where(MetaAccount.id == 110))
        await mdb_rw.execute(delete(FetchProgress).where(FetchProgress.acc_id == 110))

    assert accounts == [10]


def _build_context(**kwargs: Any) -> PrecomputeContext:
    kwargs.setdefault("log", logging.getLogger(__name__))
    for db_field in ("sdb", "mdb", "pdb", "rdb"):
        if db_field not in kwargs:
            kwargs[db_field] = mock.Mock(Database)
    kwargs.setdefault("cache", None)
    kwargs.setdefault("slack", None)
    return PrecomputeContext(**kwargs)
