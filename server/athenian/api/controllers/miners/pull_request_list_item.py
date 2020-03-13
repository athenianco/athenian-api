from dataclasses import dataclass
from enum import IntEnum
from typing import Collection, Mapping, Optional

import pandas as pd


class ParticipationKind(IntEnum):
    """The way the developer relates to a pull request."""

    AUTHOR = 1
    REVIEWER = 2
    COMMENTER = 3
    COMMIT_AUTHOR = 4
    COMMIT_COMMITTER = 5
    MERGER = 6


class Property(IntEnum):
    """PR's modelled lifecycle stage or corresponding events between `time_from` and `time_to`."""

    WIP = 1
    REVIEWING = 2
    MERGING = 3
    RELEASING = 4
    DONE = 5
    CREATED = 6
    COMMIT_HAPPENED = 7
    REVIEW_HAPPENED = 8
    APPROVE_HAPPENED = 9
    MERGE_HAPPENED = 10
    RELEASE_HAPPENED = 11


@dataclass(frozen=True)
class PullRequestListItem:
    """General PR properties used to list PRs on the frontend."""

    repository: str
    number: int
    title: str
    size_added: int
    size_removed: int
    files_changed: int
    created: pd.Timestamp
    updated: pd.Timestamp
    closed: Optional[pd.Timestamp]
    comments: int
    commits: int
    review_requested: bool
    review_comments: int
    merged: bool
    properties: Collection[Property]
    participants: Mapping[ParticipationKind, Collection[str]]
