from argparse import Namespace
from unittest import mock

from athenian.api.models.state.models import RepositorySet
from athenian.api.precompute.accounts import main, precompute_reposet
from tests.testutils.db import model_insert_stmt
from tests.testutils.factory.state import AccountFactory, RepositorySetFactory

from .conftest import build_context, clear_all_accounts


async def test_reposets_grouping(sdb, mdb, tqdm_disable) -> None:
    await clear_all_accounts(sdb)
    for model in (
        AccountFactory(id=11),
        RepositorySetFactory(owner_id=11),
        AccountFactory(id=12),
        RepositorySetFactory(owner_id=12),
        RepositorySetFactory(owner_id=12, name="another-reposet"),
    ):
        await sdb.execute(model_insert_stmt(model))

    ctx = build_context(sdb=sdb, mdb=mdb)
    namespace = Namespace(skip_jira_identity_map=True, account=["11", "12"])

    with mock.patch(
        f"{main.__module__}.precompute_reposet", wraps=precompute_reposet,
    ) as precompute_mock:
        await main(ctx, namespace)

    assert precompute_mock.call_count == 2
    # real precompute_reposet calls order is undefined, order calls by account id
    call_args_list = sorted(
        precompute_mock.call_args_list, key=lambda call: call[0][0].owner_id,
    )
    assert call_args_list[0][0][0].owner_id == 11
    assert call_args_list[0][0][0].name == RepositorySet.ALL
    assert call_args_list[1][0][0].owner_id == 12
    assert call_args_list[1][0][0].name == RepositorySet.ALL
