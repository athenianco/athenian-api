from datetime import datetime, timezone

from freezegun import freeze_time
import numpy as np
from pandas.testing import assert_frame_equal
import pytest
from sqlalchemy import delete, insert, select, update

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.github.branches import BranchMinerMetrics
from athenian.api.models.metadata.github import Branch


async def test_load_branches_zero(mdb, branch_miner, prefixer, meta_ids):
    metrics = BranchMinerMetrics.empty()
    branches, defaults = await branch_miner.load_branches(
        ["src-d/gitbase"], prefixer, 1, meta_ids, mdb, None, None, metrics=metrics,
    )
    assert branches.empty
    assert metrics.empty_count == 1
    assert defaults == {"src-d/gitbase": "master"}
    branches, defaults = await branch_miner.load_branches(
        ["src-d/gitbase"], prefixer, 1, meta_ids, mdb, None, None,
    )
    assert branches.empty
    assert defaults == {"src-d/gitbase": "master"}


async def test_load_branches_trash(mdb, branch_miner, prefixer, meta_ids):
    branches, defaults = await branch_miner.load_branches(
        ["src-d/whatever"], prefixer, 1, meta_ids, mdb, None, None,
    )
    assert branches.empty
    assert defaults == {"src-d/whatever": "master"}


@with_defer
async def test_load_branches_cache(mdb, cache, branch_miner, prefixer, meta_ids):
    branches, defaults = await branch_miner.load_branches(
        ["src-d/go-git"], prefixer, 1, meta_ids, mdb, None, cache,
    )
    await wait_deferred()
    assert "src-d/go-git" in branches[Branch.repository_full_name.name].values
    assert defaults == {"src-d/go-git": "master", "src-d/юникод": "вадим"}
    branches1, defaults1 = await branch_miner.load_branches(
        ["src-d/go-git"], prefixer, 1, meta_ids, None, None, cache,
    )
    await wait_deferred()
    assert "src-d/go-git" in branches1[Branch.repository_full_name.name].values
    assert defaults1 == {"src-d/go-git": "master"}
    branches2, defaults2 = await branch_miner.load_branches(
        ["github.com/src-d/go-git"],
        prefixer,
        1,
        meta_ids,
        None,
        None,
        cache,
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
            meta_ids,
            None,
            None,
            cache,
        )
    with pytest.raises(AttributeError):
        await branch_miner.load_branches(
            ["src-d/go-git"],
            prefixer,
            2,
            meta_ids,
            None,
            None,
            cache,
        )


async def test_load_branches_main(mdb_rw, branch_miner, prefixer, meta_ids):
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
            meta_ids,
            mdb,
            None,
            None,
            metrics=metrics,
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


async def test_load_branches_max_date(mdb_rw, branch_miner, prefixer, meta_ids):
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
            ["src-d/go-git"], prefixer, 1, meta_ids, mdb, None, None,
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


async def test_load_branches_only_one(mdb_rw, branch_miner, prefixer, meta_ids):
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
                ["src-d/go-git"], prefixer, 1, meta_ids, mdb, None, None,
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
async def test_load_branches_none_repos(mdb, cache, branch_miner, prefixer, meta_ids):
    branches, defaults = await branch_miner.load_branches(
        [], prefixer, 1, meta_ids, mdb, None, cache,
    )
    assert len(branches) == 0
    await wait_deferred()
    branches, defaults = await branch_miner.load_branches(
        None, prefixer, 1, meta_ids, mdb, None, cache,
    )
    assert len(branches) == 5


@with_defer
async def test_load_branches_logical(mdb, cache, branch_miner, prefixer, meta_ids):
    branches, defaults = await branch_miner.load_branches(
        ["src-d/go-git/alpha", "src-d/go-git/beta"],
        prefixer,
        1,
        meta_ids,
        mdb,
        None,
        cache,
    )
    assert len(branches) == 5


@with_defer
async def test_load_branches_precomputed_from_scratch(mdb, pdb, branch_miner, prefixer, meta_ids):
    await _test_load_branches_precomputed(mdb, pdb, branch_miner, prefixer, meta_ids)


@freeze_time("2018-01-01")
@with_defer
async def test_load_branches_precomputed_append(mdb, pdb, branch_miner, prefixer, meta_ids):
    await _test_load_branches_precomputed(mdb, pdb, branch_miner, prefixer, meta_ids)


async def _test_load_branches_precomputed(mdb, pdb, branch_miner, prefixer, meta_ids):
    branches, defaults = await branch_miner.load_branches(
        ["src-d/go-git"],
        prefixer,
        1,
        meta_ids,
        mdb,
        pdb,
        None,
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
        meta_ids,
        mdb,
        pdb,
        None,
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


@with_defer
async def test_load_branches_extend_precomputed(mdb_rw, pdb, branch_miner, prefixer, meta_ids):
    for i, repo in enumerate([39652699, 39652769], start=1):
        await mdb_rw.execute(
            insert(Branch).values(
                acc_id=meta_ids[0],
                branch_id=170 + i,
                branch_name="first-things-first",
                is_default=True,
                commit_id=2023,
                commit_sha="0" * 40,
                commit_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
                commit_repository_node_id=repo,
                repository_node_id=repo,
                repository_full_name="src-d/gitbase",
            ),
        )
    try:
        branches, defaults = await branch_miner.load_branches(
            ["src-d/gitbase"],
            prefixer,
            1,
            meta_ids,
            mdb_rw,
            pdb,
            None,
        )
    finally:
        await mdb_rw.execute(delete(Branch).where(Branch.branch_id.in_([171, 172])))
    assert len(branches) == 1
    assert defaults == {"src-d/gitbase": "first-things-first"}
    await wait_deferred()
    branches, defaults = await branch_miner.load_branches(
        ["src-d/gitbase", "src-d/go-git"],
        prefixer,
        1,
        meta_ids,
        mdb_rw,
        pdb,
        None,
    )
    assert len(branches) == 1 + 5
    assert defaults["src-d/gitbase"] == "first-things-first"
    assert defaults["src-d/go-git"] == "master"
    assert (branches[Branch.repository_full_name.name].values == "src-d/gitbase").any()
    assert (branches[Branch.repository_full_name.name].values == "src-d/go-git").sum() == 4
