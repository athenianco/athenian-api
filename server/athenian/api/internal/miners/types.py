from datetime import datetime, timedelta
from enum import IntEnum, auto
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union

import numpy as np
import pandas as pd

from athenian.api.models.metadata.github import (
    NodePullRequest,
    PullRequest,
    PullRequestComment,
    PullRequestCommit,
    PullRequestReview,
    Release,
)
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


PRParticipants = Mapping[PRParticipationKind, Set[int]]


class ReleaseParticipationKind(IntEnum):
    """Developer relationship with a release."""

    PR_AUTHOR = auto()
    COMMIT_AUTHOR = auto()
    RELEASER = auto()


ReleaseParticipants = Mapping[ReleaseParticipationKind, Sequence[int]]


class JIRAParticipationKind(IntEnum):
    """User relationship with a JIRA issue."""

    ASSIGNEE = auto()
    REPORTER = auto()
    COMMENTER = auto()


JIRAParticipants = Mapping[JIRAParticipationKind, Sequence[str]]


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
    DEPLOYED = auto()


class PullRequestStage(IntEnum):
    """PR's modelled lifecycle stage."""

    WIP = auto()
    REVIEWING = auto()
    MERGING = auto()
    RELEASING = auto()
    FORCE_PUSH_DROPPED = auto()
    DONE = auto()
    DEPLOYED = auto()
    RELEASE_IGNORED = auto()


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
    stage_timings: Dict[str, Union[timedelta, Dict[str, timedelta]]]
    events_time_machine: Optional[Set[PullRequestEvent]]
    stages_time_machine: Optional[Set[PullRequestStage]]
    events_now: Set[PullRequestEvent]
    stages_now: Set[PullRequestStage]
    participants: PRParticipants
    labels: List[Label]
    jira: Optional[List[PullRequestJIRAIssueItem]]
    merged_with_failed_check_runs: Optional[List[str]]
    deployments: Optional[np.ndarray]


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
    deployments: pd.DataFrame
    check_run: Dict[str, Any]

    def participant_nodes(self) -> PRParticipants:
        """Collect unique developer node IDs that are mentioned in this pull request."""
        author = self.pr[PullRequest.user_node_id.name]
        merger = self.pr[PullRequest.merged_by_id.name]
        releaser = self.release[Release.author_node_id.name]
        participants = {
            PRParticipationKind.AUTHOR: {author} if author else set(),
            PRParticipationKind.REVIEWER: self._extract_people(
                self.reviews, PullRequestReview.user_node_id.name,
            ),
            PRParticipationKind.COMMENTER: self._extract_people(
                self.comments, PullRequestComment.user_node_id.name,
            ),
            PRParticipationKind.COMMIT_COMMITTER: self._extract_people(
                self.commits, PullRequestCommit.committer_user_id.name,
            ),
            PRParticipationKind.COMMIT_AUTHOR: self._extract_people(
                self.commits, PullRequestCommit.author_user_id.name,
            ),
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


class DeploymentConclusion(IntEnum):
    """Possible deployment outcomes."""

    SUCCESS = auto()
    FAILURE = auto()
    CANCELLED = auto()


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
        release_ignored: np.bool_
        reviews: ["datetime64[s]"]
        activity_days: ["datetime64[s]"]
        size: np.int64
        force_push_dropped: np.bool_
        review_comments: np.uint16
        regular_comments: np.uint16
        participants: np.uint16
        merged_with_failed_check_runs: [str]

    class Optional:
        """Mutable fields that are None by default. We do not serialize them."""

        node_id: int
        jira_ids: List[str]
        repository_full_name: str
        author: str
        merger: str
        releaser: str
        deployments: Sequence[str]
        environments: Sequence[str]
        deployment_conclusions: Sequence[DeploymentConclusion]
        deployed: Sequence[datetime]

    def max_timestamp(self) -> pd.Timestamp:
        """Find the maximum timestamp contained in the struct."""
        if self.released is not None:
            return self.released
        if self.closed is not None:
            return self.closed
        return max(
            t
            for t in (
                self.created,
                self.first_commit,
                self.last_commit,
                self.first_review_request,
                self.last_review,
            )
            if t is not None
        )

    def truncate(self, after_dt: Union[pd.Timestamp, datetime]) -> "PullRequestFacts":
        """Create a copy of the facts without timestamps bigger than or equal to `dt`."""
        changed = []
        assert self.created <= after_dt
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
            (released := arr["released"]) == released
            or arr["force_push_dropped"]
            or ((closed := arr["closed"]) == closed and (merged := arr["merged"]) != merged)
        )
        data = b"".join([arr.view(np.byte).data, self.data[self.dtype.itemsize :]])
        if len(self.deployed):
            deployed = np.asarray(self.deployed)
            np_after_dt = np.array(after_dt, dtype=deployed.dtype)
            deps_passed = np.flatnonzero(deployed < np_after_dt)
            deployed = deployed[deps_passed]
            deployments = [self.deployments[i] for i in deps_passed]
            environments = [self.environments[i] for i in deps_passed]
            return PullRequestFacts(
                data,
                node_id=self.node_id,
                deployed=deployed,
                deployments=deployments,
                environments=environments,
            )
        return PullRequestFacts(data, node_id=self.node_id)

    def __lt__(self, other: "PullRequestFacts") -> bool:
        """Order by `work_began`."""
        if self.work_began != other.work_began:
            return self.work_began < other.work_began
        return self.created < other.created


# a PullRequest is identified by the couple (pull_request_node_id, repository_full_name)
PullRequestID = Tuple[int, str]
PullRequestFactsMap = Dict[PullRequestID, PullRequestFacts]


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


def released_prs_columns(model: Union[Type[PullRequest], Type[NodePullRequest]]):
    """Return the columns that must exist in the released PR DataFrame."""
    return [
        model.node_id.label(PullRequest.node_id.name),
        model.number,
        model.additions,
        model.deletions,
        model.user_node_id.label(PullRequest.user_node_id.name),
    ]


def _add_prs_annotations(cls):
    for col in released_prs_columns(PullRequest):
        cls.__annotations__["prs_" + col.name] = [int]
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
        publisher: int
        matched_by: np.int8
        age: "timedelta64[s]"
        additions: int
        deletions: int
        commits_count: int
        commit_authors: [int]

    class Optional:
        """Mutable fields that are None by default. We do not serialize them."""

        node_id: int
        repository_full_name: str
        prs_title: List[str]
        prs_jira: np.ndarray
        deployments: Optional[np.ndarray]

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
    critical: bool
    skips: int
    flaky_count: int
    mean_execution_time: Optional[timedelta]
    stddev_execution_time: Optional[timedelta]
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


@dataclass(slots=True, frozen=True)
class DeployedComponent:
    """Deployed repository reference."""

    repository_full_name: str
    reference: str
    sha: str


@dataclass(slots=True, frozen=True)
class Deployment:
    """Lightweight deployment information used in includes."""

    name: str
    conclusion: DeploymentConclusion
    environment: str
    url: Optional[str]
    started_at: datetime
    finished_at: datetime
    components: List[DeployedComponent]
    labels: Optional[Dict[str, Any]]


@numpy_struct
class DeploymentFacts:
    """Various precomputed data about a deployment."""

    class Immutable:
        """
        Immutable fields, we store them in `_data` and mirror in `_arr`.

        We generate `dtype` from this spec.
        """

        pr_authors: [int]
        commit_authors: [int]
        release_authors: [int]
        repositories: [str]
        prs: [int]
        prs_offsets: [np.int32]
        lines_prs: [int]
        lines_overall: [int]
        commits_prs: [np.int32]
        commits_overall: [np.int32]

    class Optional:
        """Mutable fields that are None by default. We do not serialize them."""

        name: str


@numpy_struct
class PullRequestCheckRun:
    """Check runs belonging to a pull request."""

    class Immutable:
        """
        Immutable fields, we store them in `_data` and mirror in `_arr`.

        We generate `dtype` from this spec.
        """

        started_at: ["datetime64[s]"]
        completed_at: ["datetime64[s]"]
        check_suite_started_at: ["datetime64[s]"]
        check_suite_completed_at: ["datetime64[s]"]
        check_suite_node_id: [int]
        conclusion: [ascii]
        status: [ascii]
        check_suite_conclusion: [ascii]
        check_suite_status: [ascii]
        name: [str]
        commit_ids: [int]

    class Optional:
        """Mutable fields that are None by default. We do not serialize them."""

        node_id: int  # PullRequest node ID


@dataclass(slots=True, frozen=True)
class Environment:
    """Mined deployment environment facts."""

    name: str
    deployments_count: int
    last_conclusion: str
