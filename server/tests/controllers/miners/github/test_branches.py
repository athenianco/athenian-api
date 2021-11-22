import pytest
from sqlalchemy import delete, insert, select, update

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.metadata.github import Branch


async def test_extract_branches_zero(mdb, branch_miner):
    branches, defaults = await branch_miner.extract_branches(
        ["src-d/gitbase"], (6366825,), mdb, None)
    assert branches.empty
    assert defaults == {"src-d/gitbase": "master"}
    branches, defaults = await branch_miner.extract_branches(
        ["src-d/gitbase"], (6366825,), mdb, None)
    assert branches.empty
    assert defaults == {"src-d/gitbase": "master"}


async def test_extract_branches_trash(mdb, branch_miner):
    branches, defaults = await branch_miner.extract_branches(
        ["src-d/whatever"], (6366825,), mdb, None)
    assert branches.empty
    assert defaults == {"src-d/whatever": "master"}


@with_defer
async def test_extract_branches_cache(mdb, cache, branch_miner):
    branches, defaults = await branch_miner.extract_branches(
        ["src-d/go-git"], (6366825,), mdb, cache)
    await wait_deferred()
    assert not branches.empty
    assert defaults == {"src-d/go-git": "master"}
    branches, defaults = await branch_miner.extract_branches(
        ["src-d/go-git"], (6366825,), None, cache)
    await wait_deferred()
    assert not branches.empty
    assert defaults == {"src-d/go-git": "master"}
    with pytest.raises(AttributeError):
        await branch_miner.extract_branches(
            ["src-d/go-git", "src-d/gitbase"], (6366825,), None, cache)


async def test_extract_branches_main(mdb_rw, branch_miner, with_preloading_enabled):
    mdb = mdb_rw
    await mdb.execute(update(Branch).where(Branch.branch_name == "master").values({
        Branch.is_default: False,
        Branch.branch_name: "main",
    }))
    if with_preloading_enabled:
        await mdb.cache.refresh()
    try:
        _, defaults = await branch_miner.extract_branches(["src-d/go-git"], (6366825,), mdb, None)
        assert defaults == {"src-d/go-git": "main"}
    finally:
        await mdb.execute(update(Branch).where(Branch.branch_name == "main").values({
            Branch.is_default: True,
            Branch.branch_name: "master",
        }))


async def test_extract_branches_max_date(mdb_rw, branch_miner, with_preloading_enabled):
    mdb = mdb_rw
    await mdb.execute(update(Branch).where(Branch.branch_name == "master").values({
        Branch.is_default: False,
        Branch.branch_name: "whatever_it_takes",
    }))
    if with_preloading_enabled:
        await mdb.cache.refresh()
    try:
        _, defaults = await branch_miner.extract_branches(["src-d/go-git"], (6366825,), mdb, None)
        assert defaults == {"src-d/go-git": "whatever_it_takes"}
    finally:
        await mdb.execute(update(Branch).where(Branch.branch_name == "whatever_it_takes").values({
            Branch.is_default: True,
            Branch.branch_name: "master",
        }))


async def test_extract_branches_only_one(mdb_rw, branch_miner, with_preloading_enabled):
    mdb = mdb_rw
    branches = await mdb.fetch_all(select([Branch]).where(Branch.branch_name != "master"))
    await mdb.execute(update(Branch).where(Branch.branch_name == "master").values({
        Branch.is_default: False,
        Branch.branch_name: "whatever_it_takes",
    }))
    if with_preloading_enabled:
        await mdb.cache.refresh()
    try:
        await mdb.execute(delete(Branch).where(Branch.branch_name != "whatever_it_takes"))
        try:
            _, defaults = await branch_miner.extract_branches(
                ["src-d/go-git"], (6366825,), mdb, None)
            assert defaults == {"src-d/go-git": "whatever_it_takes"}
        finally:
            for branch in branches:
                await mdb.execute(insert(Branch).values(branch))
    finally:
        await mdb.execute(update(Branch).where(Branch.branch_name == "whatever_it_takes").values({
            Branch.is_default: True,
            Branch.branch_name: "master",
        }))


@with_defer
async def test_extract_branches_none_repos(mdb, cache, branch_miner):
    branches, defaults = await branch_miner.extract_branches(
        None, (6366825,), mdb, cache)
    assert len(branches) == 4


@with_defer
async def test_extract_branches_logical(mdb, cache, branch_miner):
    branches, defaults = await branch_miner.extract_branches(
        ["src-d/go-git/alpha", "src-d/go-git/beta"], (6366825,), mdb, cache)
    assert len(branches) == 4
