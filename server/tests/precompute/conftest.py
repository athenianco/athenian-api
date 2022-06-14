from functools import partialmethod
import logging
from typing import Any, Iterator
from unittest import mock

import pytest
import sqlalchemy as sa
import tqdm

from athenian.api.db import Database
from athenian.api.models.state.models import (
    Account,
    AccountFeature,
    AccountGitHubAccount,
    AccountJiraInstallation,
    Invitation,
    RepositorySet,
    UserAccount,
    WorkType,
)
from athenian.api.precompute.context import PrecomputeContext


@pytest.fixture
def tqdm_disable() -> Iterator[None]:
    orig_init = tqdm.tqdm.__init__
    tqdm.tqdm.__init__ = partialmethod(orig_init, disable=True)
    yield
    tqdm.tqdm.__init__ = orig_init


def build_context(**kwargs: Any) -> PrecomputeContext:
    """Build the context usable by precomputer commands."""
    kwargs.setdefault("log", logging.getLogger(__name__))
    for db_field in ("sdb", "mdb", "pdb", "rdb"):
        if db_field not in kwargs:
            kwargs[db_field] = mock.Mock(Database)
    kwargs.setdefault("cache", None)
    kwargs.setdefault("slack", None)
    return PrecomputeContext(**kwargs)


async def clear_all_accounts(sdb: Database) -> None:
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
        await sdb.execute(sa.delete(model))
