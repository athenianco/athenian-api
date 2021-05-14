from random import random

import pytest
from sqlalchemy import delete, insert, select, update

from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.experiments.preloading.entries import PreloadedBranchMiner
from athenian.api.models.metadata.github import Branch


async def test_extract_branches_zero(mdb, with_preloading):
    branch_miner = PreloadedBranchMiner if with_preloading else BranchMiner
    branches, defaults = await branch_miner.extract_branches(
        ["src-d/gitbase"], (6366825,), mdb, None)
    assert branches.empty
    assert defaults == {"src-d/gitbase": "master"}
    async with mdb.connection() as conn:
        branches, defaults = await branch_miner.extract_branches(
            ["src-d/gitbase"], (6366825,), conn, None)
        assert branches.empty
        assert defaults == {"src-d/gitbase": "master"}


async def test_extract_branches_trash(mdb, with_preloading):
    branch_miner = PreloadedBranchMiner if with_preloading else BranchMiner
    branches, defaults = await branch_miner.extract_branches(
        ["src-d/whatever"], (6366825,), mdb, None)
    assert branches.empty
    assert defaults == {"src-d/whatever": "master"}


@with_defer
async def test_extract_branches_cache(mdb, cache, with_preloading):
    branch_miner = PreloadedBranchMiner if with_preloading else BranchMiner
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


@pytest.mark.flaky(reruns=5, reruns_delay=0.1 + random())
async def test_extract_branches_main(mdb, with_preloading):
    branch_miner = PreloadedBranchMiner if with_preloading else BranchMiner
    await mdb.execute(update(Branch).where(Branch.branch_name == "master").values({
        Branch.is_default: False,
        Branch.branch_name: "main",
    }))
    try:
        _, defaults = await branch_miner.extract_branches(["src-d/go-git"], (6366825,), mdb, None)
        assert defaults == {"src-d/go-git": "main"}
    finally:
        await mdb.execute(update(Branch).where(Branch.branch_name == "main").values({
            Branch.is_default: True,
            Branch.branch_name: "master",
        }))


@pytest.mark.flaky(reruns=5, reruns_delay=0.1 + random())
async def test_extract_branches_max_date(mdb, with_preloading):
    branch_miner = PreloadedBranchMiner if with_preloading else BranchMiner
    await mdb.execute(update(Branch).where(Branch.branch_name == "master").values({
        Branch.is_default: False,
        Branch.branch_name: "whatever_it_takes",
    }))
    try:
        _, defaults = await branch_miner.extract_branches(["src-d/go-git"], (6366825,), mdb, None)
        assert defaults == {"src-d/go-git": "whatever_it_takes"}
    finally:
        await mdb.execute(update(Branch).where(Branch.branch_name == "whatever_it_takes").values({
            Branch.is_default: True,
            Branch.branch_name: "master",
        }))


@pytest.mark.flaky(reruns=5, reruns_delay=0.1 + random())
async def test_extract_branches_only_one(mdb, with_preloading):
    branch_miner = PreloadedBranchMiner if with_preloading else BranchMiner
    branches = await mdb.fetch_all(select([Branch]).where(Branch.branch_name != "master"))
    await mdb.execute(update(Branch).where(Branch.branch_name == "master").values({
        Branch.is_default: False,
        Branch.branch_name: "whatever_it_takes",
    }))
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
