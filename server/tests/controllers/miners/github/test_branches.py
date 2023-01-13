from freezegun import freeze_time
import numpy as np
from pandas.testing import assert_frame_equal
import pytest
from sqlalchemy import delete, insert, select, update

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.github.branches import BranchMinerMetrics
from athenian.api.models.metadata.github import Branch


async def test_load_branches_zero(mdb, branch_miner, prefixer):
    metrics = BranchMinerMetrics.empty()
    branches, defaults = await branch_miner.load_branches(
        ["src-d/gitbase"], prefixer, 1, (6366825,), mdb, None, None, metrics=metrics, fresh=True,
    )
    assert branches.empty
    assert metrics.empty_count == 1
    assert defaults == {"src-d/gitbase": "master"}
    branches, defaults = await branch_miner.load_branches(
        ["src-d/gitbase"], prefixer, 1, (6366825,), mdb, None, None, fresh=True,
    )
    assert branches.empty
    assert defaults == {"src-d/gitbase": "master"}


async def test_load_branches_trash(mdb, branch_miner, prefixer):
    branches, defaults = await branch_miner.load_branches(
        ["src-d/whatever"], prefixer, 1, (6366825,), mdb, None, None, fresh=True,
    )
    assert branches.empty
    assert defaults == {"src-d/whatever": "master"}


@with_defer
async def test_load_branches_cache(mdb, cache, branch_miner, prefixer):
    branches, defaults = await branch_miner.load_branches(
        ["src-d/go-git"], prefixer, 1, (6366825,), mdb, None, cache, fresh=True,
    )
    await wait_deferred()
    assert "src-d/go-git" in branches[Branch.repository_full_name.name].values
    assert defaults == {"src-d/go-git": "master", "src-d/юникод": "вадим"}
    branches1, defaults1 = await branch_miner.load_branches(
        ["src-d/go-git"], prefixer, 1, (6366825,), None, None, cache, fresh=True,
    )
    await wait_deferred()
    assert "src-d/go-git" in branches1[Branch.repository_full_name.name].values
    assert defaults1 == {"src-d/go-git": "master"}
    branches2, defaults2 = await branch_miner.load_branches(
        ["github.com/src-d/go-git"],
        prefixer,
        1,
        (6366825,),
        None,
        None,
        cache,
        fresh=True,
        strip=True,
    )
    await wait_deferred()
    assert_frame_equal(branches1, branches2)
    assert defaults1 == defaults2
    with pytest.raises(AttributeError):
        await branch_miner.load_branches(
            ["src-d/go-git", "src-d/gitbase"],
            prefixer,
            1,
            (6366825,),
            None,
            None,
            cache,
            fresh=True,
        )
    with pytest.raises(AttributeError):
        await branch_miner.load_branches(
            ["src-d/go-git"],
            prefixer,
            2,
            (6366825,),
            None,
            None,
            cache,
            fresh=True,
        )


async def test_load_branches_main(mdb_rw, branch_miner, prefixer):
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
            ["src-d/go-git"],
            prefixer,
            1,
            (6366825,),
            mdb,
            None,
            None,
            metrics=metrics,
            fresh=True,
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


async def test_load_branches_max_date(mdb_rw, branch_miner, prefixer):
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
            ["src-d/go-git"], prefixer, 1, (6366825,), mdb, None, None, fresh=True,
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


async def test_load_branches_only_one(mdb_rw, branch_miner, prefixer):
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
                ["src-d/go-git"], prefixer, 1, (6366825,), mdb, None, None, fresh=True,
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
async def test_load_branches_none_repos(mdb, cache, branch_miner, prefixer):
    branches, defaults = await branch_miner.load_branches(
        [], prefixer, 1, (6366825,), mdb, None, cache, fresh=True,
    )
    assert len(branches) == 0
    await wait_deferred()
    branches, defaults = await branch_miner.load_branches(
        None, prefixer, 1, (6366825,), mdb, None, cache, fresh=True,
    )
    assert len(branches) == 5


@with_defer
async def test_load_branches_logical(mdb, cache, branch_miner, prefixer):
    branches, defaults = await branch_miner.load_branches(
        ["src-d/go-git/alpha", "src-d/go-git/beta"],
        prefixer,
        1,
        (6366825,),
        mdb,
        None,
        cache,
        fresh=True,
    )
    assert len(branches) == 5


@with_defer
async def test_load_branches_precomputed_from_scratch(mdb, pdb, branch_miner, prefixer):
    await _test_load_branches_precomputed(mdb, pdb, branch_miner, prefixer)


@freeze_time("2018-01-01")
@with_defer
async def test_load_branches_precomputed_append(mdb, pdb, branch_miner, prefixer):
    await _test_load_branches_precomputed(mdb, pdb, branch_miner, prefixer)


async def _test_load_branches_precomputed(mdb, pdb, branch_miner, prefixer):
    branches, defaults = await branch_miner.load_branches(
        ["src-d/go-git"],
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        None,
        fresh=True,
    )
    # unicode check, must exclude this repo with the same ID
    branches = branches.take(
        np.flatnonzero(branches[Branch.repository_full_name.name].values != "src-d/юникод"),
    )
    defaults.pop("src-d/юникод", None)
    assert len(branches) == 4
    await wait_deferred()
    branches_new, defaults_new = await branch_miner.load_branches(
        ["src-d/go-git"],
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        None,
        fresh=False,
    )
    # the same repo node ID but different repo name "src-d/юникод" - loaded as "src-d/go-git"
    branches_new = branches_new.take(
        np.flatnonzero(branches_new[Branch.branch_name.name].values != "вадим"),
    )
    branches = (
        branches.reindex(sorted(branches.columns), axis=1)
        .sort_values(Branch.branch_name.name, ignore_index=True)
        .reset_index(drop=True)
    )
    branches_new = (
        branches_new.reindex(sorted(branches_new.columns), axis=1)
        .sort_values(Branch.branch_name.name, ignore_index=True)
        .reset_index(drop=True)
    )
    assert_frame_equal(branches, branches_new)
    defaults_new.pop("src-d/юникод", None)
    assert defaults == defaults_new
