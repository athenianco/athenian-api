from datetime import datetime, timedelta
from enum import auto, IntEnum
from typing import Any, Dict, List, Mapping, Optional, Set, Union

import numpy as np
import pandas as pd
import sqlalchemy as sa

from athenian.api.models.metadata.github import PullRequest, PullRequestComment, \
    PullRequestCommit, PullRequestReview, Release
from athenian.api.typing_utils import dataclass, numpy_struct


class PRParticipationKind(IntEnum):
    """Developer relationship with a pull request.

    These values are written to the precomputed DB, so be careful with changing them.
    """

    AUTHOR = auto()
    REVIEWER = auto()
    COMMENTER = auto()
    COMMIT_AUTHOR = auto()
    COMMIT_COMMITTER = auto()
    MERGER = auto()
    RELEASER = auto()


PRParticipants = Mapping[PRParticipationKind, Set[str]]


class ReleaseParticipationKind(IntEnum):
    """Developer relationship with a release."""

    PR_AUTHOR = auto()
    COMMIT_AUTHOR = auto()
    RELEASER = auto()


ReleaseParticipants = Mapping[ReleaseParticipationKind, List[str]]


class Property(IntEnum):
    """PR's modelled lifecycle stage or corresponding events between `time_from` and `time_to`."""

    # stages begin
    WIP = auto()
    REVIEWING = auto()
    MERGING = auto()
    RELEASING = auto()
    FORCE_PUSH_DROPPED = auto()
    DONE = auto()
    # stages end
    # events begin
    CREATED = auto()
    COMMIT_HAPPENED = auto()
    REVIEW_HAPPENED = auto()
    APPROVE_HAPPENED = auto()
    REVIEW_REQUEST_HAPPENED = auto()
    CHANGES_REQUEST_HAPPENED = auto()
    MERGE_HAPPENED = auto()
    REJECTION_HAPPENED = auto()
    RELEASE_HAPPENED = auto()
    # events end


class PullRequestEvent(IntEnum):
    """PR's modelled lifecycle event."""

    CREATED = auto()
    COMMITTED = auto()
    REVIEWED = auto()
    APPROVED = auto()
    REVIEW_REQUESTED = auto()
    CHANGES_REQUESTED = auto()
    MERGED = auto()
    REJECTED = auto()
    RELEASED = auto()


class PullRequestStage(IntEnum):
    """PR's modelled lifecycle stage."""

    WIP = auto()
    REVIEWING = auto()
    MERGING = auto()
    RELEASING = auto()
    FORCE_PUSH_DROPPED = auto()
    DONE = auto()


@dataclass(slots=True, frozen=True)
class Label:
    """Pull request label."""

    name: str
    description: Optional[str]
    color: str


@dataclass(slots=True, frozen=True)
class PullRequestJIRAIssueItem:
    """JIRA PR properties."""

    id: str
    title: str
    labels: Optional[Set[str]]
    epic: Optional[str]
    type: str


@dataclass(slots=True, frozen=True, first_mutable="merged_with_failed_check_runs")
class PullRequestListItem:
    """
    General PR properties used to list PRs on the frontend.

    We have to declare `merged_with_failed_check_runs` mutable because it has to be set async.
    """

    node_id: int
    repository: str
    number: int
    title: str
    size_added: int
    size_removed: int
    files_changed: int
    created: datetime
    updated: datetime
    closed: Optional[datetime]
    comments: int
    commits: int
    review_requested: Optional[datetime]
    first_review: Optional[datetime]
    approved: Optional[datetime]
    review_comments: int
    reviews: int
    merged: Optional[datetime]
    released: Optional[datetime]
    release_url: str
    stage_timings: Dict[str, timedelta]
    events_time_machine: Optional[Set[PullRequestEvent]]
    stages_time_machine: Optional[Set[PullRequestStage]]
    events_now: Set[PullRequestEvent]
    stages_now: Set[PullRequestStage]
    participants: PRParticipants
    labels: List[Label]
    jira: Optional[List[PullRequestJIRAIssueItem]]
    merged_with_failed_check_runs: Optional[List[str]]


@dataclass(slots=True, frozen=True)
class MinedPullRequest:
    """All the relevant information we are able to load from the metadata DB about a PR.

    All the DataFrame-s have a two-layered index:
    1. pull request id
    2. own id
    The artificial first index layer makes it is faster to select data belonging to a certain PR.
    """

    pr: Dict[str, Any]
    commits: pd.DataFrame
    reviews: pd.DataFrame
    review_comments: pd.DataFrame
    review_requests: pd.DataFrame
    comments: pd.DataFrame
    release: Dict[str, Any]
    labels: pd.DataFrame
    jiras: pd.DataFrame

    def participant_logins(self) -> PRParticipants:
        """Collect unique developer logins that are mentioned in this pull request."""
        author = self.pr[PullRequest.user_login.name]
        merger = self.pr[PullRequest.merged_by_login.name]
        releaser = self.release[Release.author.name]
        participants = {
            PRParticipationKind.AUTHOR: {author} if author else set(),
            PRParticipationKind.REVIEWER: self._extract_people(
                self.reviews, PullRequestReview.user_login.name),
            PRParticipationKind.COMMENTER: self._extract_people(
                self.comments, PullRequestComment.user_login.name),
            PRParticipationKind.COMMIT_COMMITTER: self._extract_people(
                self.commits, PullRequestCommit.committer_login.name),
            PRParticipationKind.COMMIT_AUTHOR: self._extract_people(
                self.commits, PullRequestCommit.author_login.name),
            PRParticipationKind.MERGER: {merger} if merger else set(),
            PRParticipationKind.RELEASER: {releaser} if releaser else set(),
        }
        reviewers = participants[PRParticipationKind.REVIEWER]
        if author in reviewers:
            reviewers.remove(author)
        return participants

    def participant_nodes(self) -> PRParticipants:
        """Collect unique developer node IDs that are mentioned in this pull request."""
        author = self.pr[PullRequest.user_node_id.name]
        merger = self.pr[PullRequest.merged_by_id.name]
        releaser = self.release[Release.author_node_id.name]
        participants = {
            PRParticipationKind.AUTHOR: {author} if author else set(),
            PRParticipationKind.REVIEWER: self._extract_people(
                self.reviews, PullRequestReview.user_node_id.name),
            PRParticipationKind.COMMENTER: self._extract_people(
                self.comments, PullRequestComment.user_node_id.name),
            PRParticipationKind.COMMIT_COMMITTER: self._extract_people(
                self.commits, PullRequestCommit.committer_user_id.name),
            PRParticipationKind.COMMIT_AUTHOR: self._extract_people(
                self.commits, PullRequestCommit.author_user_id.name),
            PRParticipationKind.MERGER: {merger} if merger else set(),
            PRParticipationKind.RELEASER: {releaser} if releaser else set(),
        }
        reviewers = participants[PRParticipationKind.REVIEWER]
        if author in reviewers:
            reviewers.remove(author)
        return participants

    @staticmethod
    def _extract_people(df: pd.DataFrame, col: str) -> Set[str]:
        values = df[col].values
        return set(np.unique(values[np.flatnonzero(values)]).tolist())


# avoid F821 in the annotations
datetime64 = timedelta64 = List
s = None


@numpy_struct
class PullRequestFacts:
    """Various PR event timestamps and other properties."""

    class Immutable:
        """
        Immutable fields, we store them in `_data` and mirror in `_arr`.

        We generate `dtype` from this spec.
        """

        created: "datetime64[s]"
        first_commit: "datetime64[s]"
        work_began: "datetime64[s]"
        last_commit_before_first_review: "datetime64[s]"
        last_commit: "datetime64[s]"
        merged: "datetime64[s]"
        closed: "datetime64[s]"
        first_comment_on_first_review: "datetime64[s]"
        first_review_request: "datetime64[s]"
        first_review_request_exact: "datetime64[s]"
        approved: "datetime64[s]"
        last_review: "datetime64[s]"
        released: "datetime64[s]"
        done: np.bool_
        reviews: ["datetime64[s]"]
        activity_days: ["datetime64[s]"]
        size: np.int64
        force_push_dropped: np.bool_
        review_comments: np.uint16
        participants: np.uint16

    class Optional:
        """Mutable fields that are None by default. We do not serialize them."""

        jira_id: str
        repository_full_name: str
        author: str
        merger: str
        releaser: str

    def max_timestamp(self) -> pd.Timestamp:
        """Find the maximum timestamp contained in the struct."""
        if self.released is not None:
            return self.released
        if self.closed is not None:
            return self.closed
        return max(t for t in (self.created, self.first_commit, self.last_commit,
                               self.first_review_request, self.last_review)
                   if t is not None)

    def truncate(self, after_dt: Union[pd.Timestamp, datetime]) -> "PullRequestFacts":
        """Create a copy of the facts without timestamps bigger than or equal to `dt`."""
        changed = []
        if after_dt.tzinfo is not None:
            after_dt = after_dt.replace(tzinfo=None)
        for field_name, (field_dtype, _) in self.dtype.fields.items():
            if np.issubdtype(field_dtype, np.datetime64):
                if (dt := getattr(self, field_name)) is not None and dt >= after_dt:
                    changed.append(field_name)
        if not changed:
            return self
        arr = self._arr.copy()
        for k in changed:
            arr[k] = None
        arr["done"] = (
            (released := arr["released"]) == released or
            arr["force_push_dropped"] or
            ((closed := arr["closed"]) == closed and (merged := arr["merged"]) != merged)
        )
        data = b"".join([arr.view(np.byte).data, self.data[self.dtype.itemsize:]])
        return PullRequestFacts(data)

    def __lt__(self, other: "PullRequestFacts") -> bool:
        """Order by `work_began`."""
        if self.work_began != other.work_began:
            return self.work_began < other.work_began
        return self.created < other.created


def nonemin(*args: Union[pd.Timestamp, type(None)]) -> Optional[pd.Timestamp]:
    """Find the minimum of several dates handling NaNs gracefully."""
    if all(arg is None for arg in args):
        return None
    return min(arg for arg in args if arg)


def nonemax(*args: Union[pd.Timestamp, type(None)]) -> Optional[pd.Timestamp]:
    """Find the maximum of several dates handling NaNs gracefully."""
    if all(arg is None for arg in args):
        return None
    return max(arg for arg in args if arg)


released_prs_columns = [
    PullRequest.node_id,
    PullRequest.number,
    PullRequest.additions,
    PullRequest.deletions,
    PullRequest.user_login,
]


def _add_prs_annotations(cls):
    col_types_map = {
        sa.Text: ascii,
        sa.BigInteger: int,
    }
    for col in released_prs_columns:
        cls.__annotations__["prs_" + col.name] = [col_types_map[type(col.type)]]
    return cls


@numpy_struct
class ReleaseFacts:
    """Various release properties and statistics."""

    @_add_prs_annotations
    class Immutable:
        """
        Immutable fields, we store them in `_data` and mirror in `_arr`.

        We generate `dtype` from this spec.
        """

        published: "datetime64[s]"
        publisher: "ascii[39]"
        matched_by: np.int8
        age: "timedelta64[s]"
        additions: int
        deletions: int
        commits_count: int
        commit_authors: [ascii]

    class Optional:
        """Mutable fields that are None by default. We do not serialize them."""

        repository_full_name: str
        prs_title: List[str]
        prs_jira: np.ndarray

    def max_timestamp(self) -> datetime:
        """Find the maximum timestamp contained in the struct."""
        return self.published


@numpy_struct
class DAG:
    """Commit history DAG."""

    class Immutable:
        """
        Immutable fields, we store them in `_data` and mirror in `_arr`.

        We generate `dtype` from this spec.
        """

        hashes: ["S40"]  # noqa
        vertexes: [np.uint32]
        edges: [np.uint32]


@dataclass(slots=True, frozen=True)
class CodeCheckRunListStats:
    """Mined statistics of a check run type identified by the name (title) in a repository in \
    a time window."""

    count: int
    successes: int
    skips: int
    flaky_count: int
    mean_execution_time: Optional[timedelta]
    median_execution_time: Optional[timedelta]
    count_timeline: List[int]
    successes_timeline: List[int]
    mean_execution_time_timeline: List[Optional[timedelta]]
    median_execution_time_timeline: List[Optional[timedelta]]


@dataclass(slots=True, frozen=True)
class CodeCheckRunListItem:
    """Overview of a check run type identified by the name (title) in a repository in a time \
    window."""

    title: str
    repository: str
    last_execution_time: datetime
    last_execution_url: str
    size_groups: List[int]
    total_stats: CodeCheckRunListStats
    prs_stats: CodeCheckRunListStats


class DeploymentConclusion(IntEnum):
    """Possible deployment outcomes."""

    SUCCESS = auto()
    FAILURE = auto()
    CANCELLED = auto()


@numpy_struct
class DeploymentFacts:
    """Various precomputed data about a deployment."""

    class Immutable:
        """
        Immutable fields, we store them in `_data` and mirror in `_arr`.

        We generate `dtype` from this spec.
        """

        pr_authors: [ascii]
        commit_authors: [ascii]
        prs: int
        lines_prs: int
        lines_overall: int
        commits_prs: int
        commits_overall: int
