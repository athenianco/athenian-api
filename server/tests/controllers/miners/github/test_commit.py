from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import delete, insert

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.commit import (
    CommitDAGMetrics,
    FilterCommitsProperty,
    _fetch_commit_history_dag,
    extract_commits,
)
from athenian.api.internal.miners.github.dag_accelerated import (
    extract_subdag,
    find_orphans,
    verify_edges_integrity,
)
from athenian.api.internal.miners.types import DeployedComponent, Deployment, DeploymentConclusion
from athenian.api.models.metadata.github import NodeCommitParent, PushCommit


@pytest.mark.parametrize(
    "property, count",
    [
        (FilterCommitsProperty.EVERYTHING, 596),
        (FilterCommitsProperty.NO_PR_MERGES, 596),
        (FilterCommitsProperty.BYPASSING_PRS, 289),
    ],
)
@with_defer
async def test_extract_commits_users(
    mdb,
    pdb,
    rdb,
    property,
    count,
    cache,
    prefixer,
    precomputed_deployments,
):
    args = dict(  # noqa: C408
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
        rdb=rdb,
        cache=cache,
    )
    commits, deps = await extract_commits(**args)
    assert deps == {
        "Dummy deployment": Deployment(
            name="Dummy deployment",
            conclusion=DeploymentConclusion.SUCCESS,
            environment="production",
            url=None,
            started_at=pd.Timestamp("2019-11-01 12:00:00+0000", tz="UTC"),
            finished_at=pd.Timestamp("2019-11-01 12:15:00+0000", tz="UTC"),
            components=[
                DeployedComponent(
                    repository_full_name="src-d/go-git",
                    reference="v4.13.1",
                    sha="0d1a009cbb604db18be960db5f1525b99a55d727",
                ),
            ],
            labels=None,
        ),
    }
    assert len(commits) == count
    if property == FilterCommitsProperty.EVERYTHING:
        sum_children = none_children = 0
        checked_flag = False
        for oid, hashes in zip(commits[PushCommit.sha.name].values, commits["children"].values):
            assert hashes is None or isinstance(hashes, list)
            if hashes is None:
                none_children += 1
            else:
                sum_children += len(hashes)
            if oid == b"ec6f456c0e8c7058a29611429965aa05c190b54b":
                assert set(hashes) == {
                    b"d82f291cde9987322c8a0c81a325e1ba6159684c",
                    b"3048d280d2d5b258d9e582a226ff4bbed34fd5c9",
                }
                checked_flag = True
        assert checked_flag
        assert none_children == 1
        assert sum_children == 675
    with_deps = 0
    for dep in commits["deployments"].values:
        with_deps += dep == ["Dummy deployment"]
    if property != FilterCommitsProperty.BYPASSING_PRS:
        assert with_deps == 395
    else:
        assert with_deps == 258
    await wait_deferred()
    args["only_default_branch"] = True
    commits = await extract_commits(**args)
    assert len(commits) < count


@pytest.mark.parametrize(
    "edges, result",
    [
        ([], ([], [])),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, "3" * 40, 0),
            ],
            ([], []),
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, "3" * 40, 1),
            ],
            ([0, 1], [1]),
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, "3" * 40, 1),
                ("2" * 40, "4" * 40, 0),
            ],
            ([], []),
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, None, 0),
            ],
            ([0, 1], [1]),
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                (None, "3" * 40, 0),
            ],
            ([1], [1]),
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, "", 0),
            ],
            ([0, 1], [1]),
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("", "3" * 40, 0),
            ],
            ([1], [1]),
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, "0" * 40, 0),
            ],
            ([], []),
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, "3" * 40, 0),
                ("2" * 40, "4" * 40, 1),
                ("2" * 40, "4" * 40, 1),
            ],
            ([], []),
        ),
        (
            [
                ("1" * 40, "2" * 40, 0),
                ("2" * 40, "3" * 40, 0),
                ("2" * 40, "4" * 40, 1),
                ("2" * 40, "5" * 40, 1),
            ],
            ([0, 1, 2, 3], [1, 2, 3]),
        ),
        (
            [
                ("2" * 40, "3" * 40, 0),
                ("2" * 40, "4" * 40, 1),
                ("1" * 40, "2" * 40, 1),
                ("2" * 40, "4" * 40, 1),
            ],
            ([2], [2]),
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
            ([2, 4, 5], [2, 5]),
        ),
    ],
)
def test_verify_edges_integrity_indexes(edges, result):
    assert verify_edges_integrity(edges)[:2] == result


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
    metrics = CommitDAGMetrics.empty()
    try:
        consistent, _, newhashes, newvertexes, newedges = await _fetch_commit_history_dag(
            True,
            subhashes,
            subvertexes,
            subedges,
            ["17dbd886616f82be2a59c0d02fd93d3d69f2392c", "1" * 40],
            [2755363, 100500],
            "src-d/go-git",
            (6366825,),
            mdb_rw,
            metrics=metrics,
        )
    finally:
        await mdb_rw.execute(delete(NodeCommitParent).where(NodeCommitParent.parent_id == 100500))
    assert not consistent
    assert len(newhashes) == len(hashes)
    assert b"1" * 40 not in newhashes
    assert metrics.corrupted == {"src-d/go-git"}
    assert metrics.orphaned == {"src-d/go-git"}
