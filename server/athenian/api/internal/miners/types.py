from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum, auto
from typing import Any, Optional, Sequence, Type, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from sqlalchemy.orm import InstrumentedAttribute

from athenian.api.internal.miners.participation import PRParticipants
from athenian.api.internal.miners.types_accelerated import (
    extract_participant_nodes,
    find_truncated_datetime,
)
from athenian.api.models.metadata.github import NodePullRequest, PullRequest
from athenian.api.models.metadata.jira import Issue
from athenian.api.typing_utils import numpy_struct


class JIRAEntityToFetch(IntEnum):
    """JIRA details to load for each PR."""

    NOTHING = 0
    ISSUES = 1
    PROJECTS = 1 << 1
    PRIORITIES = 1 << 2
    TYPES = 1 << 3

    @classmethod
    def EVERYTHING(cls) -> int:
        """Return all the supported JIRA entities."""
        mask = 0
        for v in cls:
            mask |= v
        return mask

    @classmethod
    def to_columns(cls, value: int) -> list[InstrumentedAttribute]:
        """Return the `model`'s columns corresponding to the `value`."""
        result = []
        if value & cls.ISSUES:
            result.append(Issue.key)
        if value & cls.PROJECTS:
            result.append(Issue.project_id)
        if value & cls.PRIORITIES:
            result.append(Issue.priority_id)
        if value & cls.TYPES:
            result.append(Issue.type_id)
        return result


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
    labels: Optional[set[str]]
    epic: Optional[str]
    type: str


@dataclass(slots=True, frozen=True)
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
    stage_timings: dict[str, Union[timedelta, dict[str, timedelta]]]
    events_time_machine: Optional[set[PullRequestEvent]]
    stages_time_machine: Optional[set[PullRequestStage]]
    events_now: set[PullRequestEvent]
    stages_now: set[PullRequestStage]
    participants: PRParticipants
    labels: list[Label]
    jira: Optional[list[PullRequestJIRAIssueItem]]
    merged_with_failed_check_runs: Optional[list[str]]
    deployments: Optional[np.ndarray]


@dataclass(slots=True, frozen=True)
class MinedPullRequest:
    """All the relevant information we are able to load from the metadata DB about a PR.

    All the DataFrame-s have a two-layered index:
    1. pull request id
    2. own id
    The artificial first index layer makes it is faster to select data belonging to a certain PR.

    Do not reorder the fields! The native extension depends on it.
    """

    pr: dict[str, Any]
    commits: pd.DataFrame
    reviews: pd.DataFrame
    review_comments: pd.DataFrame
    review_requests: pd.DataFrame
    comments: pd.DataFrame
    release: dict[str, Any]
    labels: pd.DataFrame
    jiras: pd.DataFrame
    deployments: pd.DataFrame
    check_run: dict[str, Any]

    def participant_nodes(self, alloc=None) -> PRParticipants:
        """Collect unique developer node IDs that are mentioned in this pull request."""
        return extract_participant_nodes(self, alloc)


class DeploymentConclusion(IntEnum):
    """Possible deployment outcomes."""

    SUCCESS = auto()
    FAILURE = auto()
    CANCELLED = auto()


# avoid F821 in the annotations
datetime64 = timedelta64 = list
s = None


LJD = TypeVar("LJD")


@dataclass(frozen=True, slots=True)
class LoadedJIRADetails:
    """Extra JIRA information loaded for pull requests, releases, etc."""

    ids: npt.NDArray[object]
    projects: npt.NDArray[bytes]
    priorities: npt.NDArray[bytes]
    types: npt.NDArray[bytes]

    @classmethod
    def _fields(cls) -> dict[str, Any]:
        return {
            name: np.array([], dtype=dtype) for name, dtype in PR_JIRA_DETAILS_COLUMN_MAP.values()
        }

    @classmethod
    def empty(cls: Type[LJD]) -> LJD:
        """Return an empty instance."""
        attr = f"_empty_{cls.__qualname__}"
        try:
            return getattr(cls, attr)
        except AttributeError:
            obj = cls(**cls._fields())
            setattr(cls, attr, obj)
            return obj


PR_JIRA_DETAILS_COLUMN_MAP = {
    Issue.key: (LoadedJIRADetails.ids.__name__, object),
    Issue.project_id: (LoadedJIRADetails.projects.__name__, Issue.project_id.info["dtype"]),
    Issue.priority_id: (LoadedJIRADetails.priorities.__name__, Issue.priority_id.info["dtype"]),
    Issue.type_id: (LoadedJIRADetails.types.__name__, Issue.type_id.info["dtype"]),
}


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
        jira: LoadedJIRADetails
        repository_full_name: str
        author: str
        merger: str
        releaser: str
        deployments: Sequence[str]
        environments: Sequence[str]
        deployment_conclusions: Sequence[DeploymentConclusion]
        deployed: Sequence[datetime]

    class INDIRECT_FIELDS:
        """Indirect fields found in dataframe conversion done by df_from_structs."""

        # added by df_from_structs() conversion when exploding `jira` field
        JIRA_IDS = "jira_ids"
        JIRA_PROJECTS = "jira_projects"
        JIRA_PRIORITIES = "jira_priorities"
        JIRA_TYPES = "jira_types"

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

    def truncate(self, after_dt: np.datetime64) -> "PullRequestFacts":
        """Create a copy of the facts without timestamps bigger than or equal to `dt`."""
        assert self.created <= after_dt
        changed = find_truncated_datetime(self, self.dt64_offsets, after_dt)
        if not changed:
            return self
        arr = self._arr.copy()
        dt64_names = self.dt64_names
        for k in changed:
            arr[dt64_names[k]] = None
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


PullRequestFacts.dt64_names = [
    name
    for name, (dtype, _) in PullRequestFacts.dtype.fields.items()
    if np.issubdtype(dtype, np.datetime64)
]
PullRequestFacts.dt64_offsets = np.array(
    [
        offset
        for dtype, offset in PullRequestFacts.dtype.fields.values()
        if np.issubdtype(dtype, np.datetime64)
    ],
)


# a PullRequest is identified by the couple (pull_request_node_id, repository_full_name)
PullRequestID = tuple[int, str]
PullRequestFactsMap = dict[PullRequestID, PullRequestFacts]


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


@dataclass(frozen=True, slots=True)
class LoadedJIRAReleaseDetails(LoadedJIRADetails):
    """Extra JIRA information loaded for releases."""

    pr_offsets: npt.NDArray[np.uint32]

    @classmethod
    def _fields(cls) -> dict[str, Any]:
        dikt = super(LoadedJIRAReleaseDetails, cls)._fields()
        dikt["pr_offsets"] = np.array([], dtype=np.uint32)
        return dikt


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
        sha: np.dtype("S40")  # noqa: F821
        name: str
        url: str
        repository_full_name: str
        prs_title: npt.NDArray[str]
        prs_created_at: npt.NDArray["datetime64[s]"]
        jira: LoadedJIRAReleaseDetails
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

        hashes: ["S40"]  # noqa: F821
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
    count_timeline: list[int]
    successes_timeline: list[int]
    mean_execution_time_timeline: list[Optional[timedelta]]
    median_execution_time_timeline: list[Optional[timedelta]]


@dataclass(slots=True, frozen=True)
class CodeCheckRunListItem:
    """Overview of a check run type identified by the name (title) in a repository in a time \
    window."""

    title: str
    repository: str
    last_execution_time: datetime
    last_execution_url: str
    size_groups: list[int]
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
    components: list[DeployedComponent]
    labels: Optional[dict[str, Any]]


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
        # this is a feature! deployment facts depend on the release settings
        # so, when they change, we re-compute the deployments
        release_authors: [int]  # do not remove
        repositories: [str]
        prs: [int]
        prs_offsets: [np.uint32]
        lines_prs: [int]
        lines_overall: [int]
        commits_prs: [np.int32]
        commits_overall: [np.int32]

    class Optional:
        """Mutable fields that are None by default. We do not serialize them."""

        name: str

    class Virtual:
        """Related fields that do not actually exist anywhere except the constructed dataframe."""

        prs_number: npt.NDArray[int]
        prs_title: npt.NDArray[object]
        prs_created_at: npt.NDArray["datetime64[s]"]
        prs_additions: npt.NDArray[int]
        prs_deletions: npt.NDArray[int]
        prs_user_node_id: npt.NDArray[int]
        prs_jira_ids: npt.NDArray[object]
        prs_jira_offsets: npt.NDArray[np.uint32]
        jira_ids: npt.NDArray[object]
        jira_offsets: npt.NDArray[np.uint32]

    def with_nothing_deployed(self):
        """Return a copy of this DeploymentFacts with no deployed entities.

        The returned DeploymentFacts will have no commits, prs, releases.
        """
        return DeploymentFacts.from_fields(
            pr_authors=[],
            commit_authors=[],
            release_authors=[],
            repositories=self.repositories,
            prs=[],
            prs_offsets=[],
            lines_prs=[],
            lines_overall=self.lines_overall,
            commits_prs=self.commits_prs,
            commits_overall=self.commits_overall,
        )


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
    repositories: Sequence[str]
