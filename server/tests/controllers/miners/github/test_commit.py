from datetime import datetime, timezone

import numpy as np
import pytest
from sqlalchemy import delete, insert

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.commit import (
    FilterCommitsProperty,
    _fetch_commit_history_dag,
    extract_commits,
)
from athenian.api.internal.miners.github.dag_accelerated import (
    extract_subdag,
    find_orphans,
    verify_edges_integrity,
)
from athenian.api.models.metadata.github import NodeCommitParent


@pytest.mark.parametrize(
    "property, count",
    [
        (FilterCommitsProperty.NO_PR_MERGES, 596),
        (FilterCommitsProperty.BYPASSING_PRS, 289),
    ],
)
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


@pytest.mark.parametrize(
    "edges, result",
    [
        ([], []),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, "3" * 40, 0),
            ],
            [],
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, "3" * 40, 1),
            ],
            [0, 1],
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, "3" * 40, 1),
                ("2" * 40, "4" * 40, 0),
            ],
            [],
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, None, 0),
            ],
            [0, 1],
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                (None, "3" * 40, 0),
            ],
            [1],
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, "", 0),
            ],
            [0, 1],
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("", "3" * 40, 0),
            ],
            [1],
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, "0" * 40, 0),
            ],
            [],
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, "3" * 40, 0),
                ("2" * 40, "4" * 40, 1),
                ("2" * 40, "4" * 40, 1),
            ],
            [],
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, "3" * 40, 0),
                ("2" * 40, "4" * 40, 1),
                ("2" * 40, "5" * 40, 1),
            ],
            [0, 1, 2, 3],
        ),
        (
            [
                ("2" * 40, "3" * 40, 0),
                ("2" * 40, "4" * 40, 1),
                ("1" * 40, "2" * 40, 1),
                ("2" * 40, "4" * 40, 1),
            ],
            [2],
        ),
        (
            [
                ("2" * 40, "3" * 40, 0),
                ("2" * 40, "4" * 40, 1),
                ("1" * 40, "2" * 40, 1),
                ("2" * 40, "4" * 40, 1),
                ("5" * 40, "1" * 40, 0),
                ("6" * 40, "1" * 40, 1),
            ],
            [2, 4, 5],
        ),
    ],
)
def test_verify_edges_integrity_indexes(edges, result):
    assert verify_edges_integrity(edges) == result


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
    assert find_orphans(edges, hashes) == {"4" * 40: [1]}
    edges = [
        ("1" * 40, "2" * 40, 0),
        ("2" * 40, "4" * 40, 0),
        ("2" * 40, "4" * 40, 0),
    ]
    assert find_orphans(edges, hashes) == {"4" * 40: [0, 1, 2]}


async def test__fetch_commit_history_dag_inconsistent(mdb_rw, dag):
    hashes, vertexes, edges = dag["src-d/go-git"][1]
    subhashes, subvertexes, subedges = extract_subdag(
        hashes,
        vertexes,
        edges,
        np.array([b"364866fc77fac656e103c1048dd7da4764c6d9d9"], dtype="S40"),
    )
    assert len(subhashes) == 1414
    await mdb_rw.execute(
        insert(NodeCommitParent).values(
            {
                NodeCommitParent.acc_id: 6366825,
                NodeCommitParent.index: 0,
                NodeCommitParent.parent_id: 100500,
                NodeCommitParent.child_id: 2755363,
            },
        ),
    )
    try:
        consistent, _, newhashes, newvertexes, newedges = await _fetch_commit_history_dag(
            subhashes,
            subvertexes,
            subedges,
            ["17dbd886616f82be2a59c0d02fd93d3d69f2392c", "1" * 40],
            [2755363, 100500],
            "src-d/go-git",
            (6366825,),
            mdb_rw,
        )
    finally:
        await mdb_rw.execute(delete(NodeCommitParent).where(NodeCommitParent.parent_id == 100500))
    assert not consistent
    assert len(newhashes) == len(hashes)
    assert b"1" * 40 not in newhashes
