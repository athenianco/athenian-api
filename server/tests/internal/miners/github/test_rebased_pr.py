import numpy as np
from pandas.testing import assert_frame_equal
from sqlalchemy import select

from athenian.api.async_utils import read_sql_query
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.github.pull_request import PullRequestMiner
from athenian.api.internal.miners.github.rebased_pr import match_rebased_prs
from athenian.api.models.metadata.github import NodeCommit, NodePullRequest, PullRequest
from athenian.api.models.precomputed.models import (
    GitHubRebaseCheckedCommit,
    GitHubRebasedPullRequest,
)


@with_defer
async def test_match_rebased_prs_ids_smoke(mdb, pdb, dag):
    rows = await mdb.fetch_all(
        select(NodeCommit.node_id).where(
            NodeCommit.acc_id == 6366825, NodeCommit.sha.in_(dag["src-d/go-git"][1][0]),
        ),
    )
    ids = np.array([r[0] for r in rows])
    await _test_match_rebased_smoke(mdb, pdb, {"commit_ids": ids})


@with_defer
async def test_match_rebased_prs_shas_smoke(mdb, pdb, dag):
    ids = dag["src-d/go-git"][1][0]
    await _test_match_rebased_smoke(mdb, pdb, {"commit_shas": ids})


async def _test_match_rebased_smoke(mdb, pdb, kwarg) -> None:
    prs1 = await match_rebased_prs([40550], 1, (6366825,), mdb, pdb, **kwarg)
    await wait_deferred()
    assert len(prs1) == 129
    prs2 = await match_rebased_prs([40550], 1, (6366825,), mdb, pdb, **kwarg)
    for df in (prs1, prs2):
        df.sort_index(axis=1, inplace=True)
        df.sort_values(GitHubRebasedPullRequest.pr_node_id.name, inplace=True, ignore_index=True)
    assert_frame_equal(prs1, prs2)
    pr_merges = dict(
        await mdb.fetch_all(
            select(NodePullRequest.node_id, NodePullRequest.merge_commit_id).where(
                NodePullRequest.acc_id == 6366825,
                NodePullRequest.node_id.in_(prs1[GitHubRebasedPullRequest.pr_node_id.name].values),
            ),
        ),
    )
    for pr_node_id, rebased_commit_id in zip(
        prs1[GitHubRebasedPullRequest.pr_node_id.name].values,
        prs1[GitHubRebasedPullRequest.matched_merge_commit_id.name].values,
    ):
        assert pr_merges[pr_node_id] != rebased_commit_id
    rows = await pdb.fetch_all(select(GitHubRebasedPullRequest))
    assert len(rows) == 129
    rows = await pdb.fetch_all(select(GitHubRebaseCheckedCommit))
    assert len(rows) == 1410


@with_defer
async def test_mark_dead_prs_smoke(mdb, pdb, branches, dag):
    forward = await read_sql_query(
        select(PullRequest), mdb, PullRequest, index=PullRequest.node_id,
    )
    dag_shas = dag["src-d/go-git"][1][0]
    merge_ids = forward[PullRequest.merge_commit_id.name].values.copy()
    alive = np.in1d(forward[PullRequest.merge_commit_sha.name].values, dag_shas)
    await PullRequestMiner.mark_dead_prs(forward, branches, dag, 1, (6366825,), mdb, pdb)
    matched = np.in1d(forward[PullRequest.merge_commit_sha.name].values, dag_shas)
    prs_forward = set(forward.index.values[matched & ~alive])
    backward = await match_rebased_prs(
        [40550], 1, (6366825,), mdb, pdb, commit_shas=dag["src-d/go-git"][1][0],
    )
    prs_backward = set(backward[GitHubRebasedPullRequest.pr_node_id.name].values)
    assert not (prs_backward - prs_forward)
    for pr_node_id in prs_forward - prs_backward:
        pr_merge = merge_ids[forward.index.values == pr_node_id]
        assert len(pr_merge) == 1
        assert pr_merge[0] == 0
