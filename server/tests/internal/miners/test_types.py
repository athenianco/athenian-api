import numpy as np
import pandas as pd

from athenian.api.defer import with_defer
from athenian.api.internal.features.github.pull_request_filter import _fetch_pull_requests
from athenian.api.internal.miners.participation import PRParticipants, PRParticipationKind
from athenian.api.internal.miners.types import MinedPullRequest
from athenian.api.internal.miners.types_accelerated import extract_participant_nodes
from athenian.api.internal.settings import LogicalRepositorySettings
from athenian.api.models.metadata.github import (
    PullRequest,
    PullRequestComment,
    PullRequestCommit,
    PullRequestReview,
    Release,
)
from athenian.api.native.mi_heap_destroy_stl_allocator import make_mi_heap_allocator_capsule


def pure_participant_nodes(mpr: MinedPullRequest) -> PRParticipants:
    """Collect unique developer node IDs that are mentioned in this pull request."""
    author = mpr.pr[PullRequest.user_node_id.name]
    merger = mpr.pr[PullRequest.merged_by_id.name]
    releaser = mpr.release[Release.author_node_id.name]
    participants = {
        PRParticipationKind.AUTHOR: {author} if author else set(),
        PRParticipationKind.REVIEWER: _extract_people(
            mpr.reviews, PullRequestReview.user_node_id.name,
        ),
        PRParticipationKind.COMMENTER: _extract_people(
            mpr.comments, PullRequestComment.user_node_id.name,
        ),
        PRParticipationKind.COMMIT_COMMITTER: _extract_people(
            mpr.commits, PullRequestCommit.committer_user_id.name,
        ),
        PRParticipationKind.COMMIT_AUTHOR: _extract_people(
            mpr.commits, PullRequestCommit.author_user_id.name,
        ),
        PRParticipationKind.MERGER: {merger} if merger else set(),
        PRParticipationKind.RELEASER: {releaser} if releaser else set(),
    }
    reviewers = participants[PRParticipationKind.REVIEWER]
    if author in reviewers:
        reviewers.remove(author)
    return participants


def _extract_people(df: pd.DataFrame, col: str) -> set[str]:
    values = df[col].values
    return set(np.unique(values[np.flatnonzero(values)]).tolist())


@with_defer
async def test_participant_nodes(
    bots,
    release_match_setting_tag,
    prefixer,
    mdb,
    pdb,
    rdb,
    meta_ids,
):
    prs, *_ = await _fetch_pull_requests(
        {"src-d/go-git": set(range(1000))},
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        meta_ids,
        mdb,
        pdb,
        rdb,
        None,
    )
    alloc = make_mi_heap_allocator_capsule()
    for pr in prs:
        accelerated = extract_participant_nodes(pr, alloc)
        assert pure_participant_nodes(pr) == accelerated, pr
