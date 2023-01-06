import pytest
from sqlalchemy import delete, insert, select, update

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.github.branches import BranchMinerMetrics
from athenian.api.models.metadata.github import Branch


async def test_extract_branches_zero(mdb, branch_miner, prefixer):
    metrics = BranchMinerMetrics.empty()
    branches, defaults = await branch_miner.load_branches(
        ["src-d/gitbase"], prefixer, (6366825,), mdb, None, metrics=metrics,
    )
    assert branches.empty
    assert metrics.empty_count == 1
    assert defaults == {"src-d/gitbase": "master"}
    branches, defaults = await branch_miner.load_branches(
        ["src-d/gitbase"], prefixer, (6366825,), mdb, None,
    )
    assert branches.empty
    assert defaults == {"src-d/gitbase": "master"}


async def test_extract_branches_trash(mdb, branch_miner, prefixer):
    branches, defaults = await branch_miner.load_branches(
        ["src-d/whatever"], prefixer, (6366825,), mdb, None,
    )
    assert branches.empty
    assert defaults == {"src-d/whatever": "master"}


@with_defer
async def test_extract_branches_cache(mdb, cache, branch_miner, prefixer):
    branches, defaults = await branch_miner.load_branches(
        ["src-d/go-git"], prefixer, (6366825,), mdb, cache,
    )
    await wait_deferred()
    assert not branches.empty
    assert defaults == {"src-d/go-git": "master", "src-d/юникод": "вадим"}
    branches, defaults = await branch_miner.load_branches(
        ["src-d/go-git"], prefixer, (6366825,), None, cache,
    )
    await wait_deferred()
    assert not branches.empty
    assert defaults == {"src-d/go-git": "master", "src-d/юникод": "вадим"}
    with pytest.raises(AttributeError):
        await branch_miner.load_branches(
            ["src-d/go-git", "src-d/gitbase"], prefixer, (6366825,), None, cache,
        )


async def test_extract_branches_main(mdb_rw, branch_miner, prefixer):
    mdb = mdb_rw
    metrics = BranchMinerMetrics.empty()
    await mdb.execute(
        update(Branch)
        .where(Branch.branch_name == "master")
        .values(
            {
                Branch.is_default: False,
                Branch.branch_name: "main",
            },
        ),
    )
    try:
        _, defaults = await branch_miner.load_branches(
            ["src-d/go-git"], prefixer, (6366825,), mdb, None, metrics=metrics,
        )
        assert defaults == {"src-d/go-git": "main", "src-d/юникод": "вадим"}
    finally:
        await mdb.execute(
            update(Branch)
            .where(Branch.branch_name == "main")
            .values(
                {
                    Branch.is_default: True,
                    Branch.branch_name: "master",
                },
            ),
        )
    assert metrics.count == 5
    assert metrics.no_default == 1


async def test_extract_branches_max_date(mdb_rw, branch_miner, prefixer):
    mdb = mdb_rw
    await mdb.execute(
        update(Branch)
        .where(Branch.branch_name == "master")
        .values(
            {
                Branch.is_default: False,
                Branch.branch_name: "whatever_it_takes",
            },
        ),
    )
    try:
        _, defaults = await branch_miner.load_branches(
            ["src-d/go-git"], prefixer, (6366825,), mdb, None,
        )
        assert defaults == {"src-d/go-git": "whatever_it_takes", "src-d/юникод": "вадим"}
    finally:
        await mdb.execute(
            update(Branch)
            .where(Branch.branch_name == "whatever_it_takes")
            .values(
                {
                    Branch.is_default: True,
                    Branch.branch_name: "master",
                },
            ),
        )


async def test_extract_branches_only_one(mdb_rw, branch_miner, prefixer):
    mdb = mdb_rw
    branches = await mdb.fetch_all(select([Branch]).where(Branch.branch_name != "master"))
    await mdb.execute(
        update(Branch)
        .where(Branch.branch_name == "master")
        .values(
            {
                Branch.is_default: False,
                Branch.branch_name: "whatever_it_takes",
            },
        ),
    )
    try:
        await mdb.execute(delete(Branch).where(Branch.branch_name != "whatever_it_takes"))
        try:
            _, defaults = await branch_miner.load_branches(
                ["src-d/go-git"], prefixer, (6366825,), mdb, None,
            )
            assert defaults == {"src-d/go-git": "whatever_it_takes"}
        finally:
            for branch in branches:
                await mdb.execute(insert(Branch).values(branch))
    finally:
        await mdb.execute(
            update(Branch)
            .where(Branch.branch_name == "whatever_it_takes")
            .values(
                {
                    Branch.is_default: True,
                    Branch.branch_name: "master",
                },
            ),
        )


@with_defer
async def test_extract_branches_none_repos(mdb, cache, branch_miner, prefixer):
    branches, defaults = await branch_miner.load_branches([], prefixer, (6366825,), mdb, cache)
    assert len(branches) == 0
    await wait_deferred()
    branches, defaults = await branch_miner.load_branches(None, prefixer, (6366825,), mdb, cache)
    assert len(branches) == 5


@with_defer
async def test_extract_branches_logical(mdb, cache, branch_miner, prefixer):
    branches, defaults = await branch_miner.load_branches(
        ["src-d/go-git/alpha", "src-d/go-git/beta"], prefixer, (6366825,), mdb, cache,
    )
    assert len(branches) == 5
