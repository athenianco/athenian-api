from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Mapping, Sequence, Union

import pandas as pd


class ParticipationKind(Enum):
    """The way the developer relates to a pull request."""

    AUTHOR = 1
    REVIEWER = 2
    COMMENTER = 3
    COMMIT_AUTHOR = 4
    COMMIT_COMMITTER = 5


class Stage(Enum):
    """Modelled evolution pipeline stage."""

    WIP = 1
    REVIEW = 2
    MERGE = 3
    RELEASE = 4


@dataclass(frozen=True)
class PullRequestListItem:
    """General PR properties used to list PRs on the frontend."""

    repository: str
    title: str
    size_added: int
    size_removed: int
    files_changed: int
    created: Union[pd.Timestamp, datetime]
    updated: Union[pd.Timestamp, datetime]
    stage: Stage
    participants: Mapping[ParticipationKind, Sequence[str]]
