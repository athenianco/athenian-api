from datetime import datetime, timezone

import numpy as np
import pytest

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.commit import extract_commits, FilterCommitsProperty
from athenian.api.internal.miners.github.dag_accelerated import find_orphans, \
    validate_edges_integrity


@pytest.mark.parametrize("property, count", [
    (FilterCommitsProperty.NO_PR_MERGES, 596),
    (FilterCommitsProperty.BYPASSING_PRS, 289),
])
@with_defer
async def test_extract_commits_users(mdb, pdb, property, count, cache, prefixer):
    """
    date_from: datetime,
                          date_to: datetime,
                          repos: Collection[str],
                          with_author: Optional[Collection[str]],
                          with_committer: Optional[Collection[str]],
                          only_default_branch: bool,
                          branch_miner: Optional[BranchMiner],
                          account: int,
                          meta_ids: Tuple[int, ...],
                          mdb: DatabaseLike,
                          pdb: DatabaseLike,
                          cache: Optional[aiomcache.Client],
    """
    args = dict(
        prop=property,
        date_from=datetime(2015, 1, 1, tzinfo=timezone.utc),
        date_to=datetime(2020, 6, 1, tzinfo=timezone.utc),
        repos=["src-d/go-git"],
        with_author=["mcuadros"],
        with_committer=["mcuadros"],
        only_default_branch=False,
        branch_miner=BranchMiner(),
        prefixer=prefixer,
        account=1,
        meta_ids=(6366825,),
        mdb=mdb,
        pdb=pdb,
        cache=cache,
    )
    commits = await extract_commits(**args)
    assert len(commits) == count
    await wait_deferred()
    args["only_default_branch"] = True
    commits = await extract_commits(**args)
    assert len(commits) < count


def test_validate_edges_integrity_indexes():
    assert validate_edges_integrity([])
    edges = [
        ("1" * 40, "2" * 40, 0),
        ("2" * 40, "3" * 40, 0),
    ]
    assert validate_edges_integrity(edges)
    edges = [
        ("1" * 40, "2" * 40, 0),
        ("2" * 40, "3" * 40, 1),
    ]
    assert not validate_edges_integrity(edges)
    edges = [
        ("1" * 40, "2" * 40, 0),
        ("2" * 40, "3" * 40, 1),
        ("2" * 40, "4" * 40, 0),
    ]
    assert validate_edges_integrity(edges)
    edges = [
        ("1" * 40, "2" * 40, 0),
        ("2" * 40, None, 0),
    ]
    assert not validate_edges_integrity(edges)
    edges = [
        ("1" * 40, "2" * 40, 0),
        (None, "3" * 40, 0),
    ]
    assert not validate_edges_integrity(edges)
    edges = [
        ("1" * 40, "2" * 40, 0),
        ("2" * 40, "", 0),
    ]
    assert not validate_edges_integrity(edges)
    edges = [
        ("1" * 40, "2" * 40, 0),
        ("", "3" * 40, 0),
    ]
    assert not validate_edges_integrity(edges)
    edges = [
        ("1" * 40, "2" * 40, 0),
        ("2" * 40, "0" * 40, 0),
    ]
    assert validate_edges_integrity(edges)
    edges = [
        ("1" * 40, "2" * 40, 0),
        ("2" * 40, "3" * 40, 0),
        ("2" * 40, "4" * 40, 1),
        ("2" * 40, "4" * 40, 1),
    ]
    assert validate_edges_integrity(edges)
    edges = [
        ("1" * 40, "2" * 40, 0),
        ("2" * 40, "3" * 40, 0),
        ("2" * 40, "4" * 40, 1),
        ("2" * 40, "5" * 40, 1),
    ]
    assert not validate_edges_integrity(edges)


def test_find_orphans():
    hashes = np.array([b"3" * 40], dtype="S40")
    assert len(find_orphans([], hashes)) == 0
    edges = [
        ("1" * 40, "2" * 40, 0),
        ("2" * 40, "3" * 40, 0),
    ]
    assert len(find_orphans(edges, hashes)) == 0
    edges = [
        ("1" * 40, "2" * 40, 0),
        ("2" * 40, "4" * 40, 0),
    ]
    assert find_orphans(edges, hashes) == np.array(["4" * 40], dtype="S40")
    edges = [
        ("1" * 40, "2" * 40, 0),
        ("2" * 40, "4" * 40, 0),
        ("2" * 40, "4" * 40, 0),
    ]
    assert find_orphans(edges, hashes) == np.array(["4" * 40], dtype="S40")
