from datetime import date
from enum import Enum
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.team_tree import TeamTree
from athenian.api.typing_utils import VerbatimOptional


class AlignGoalsRequest(Model):
    """Definition of the goal-metric."""

    account: int
    team: int
    only_with_targets: bool = False
    include_series: bool = False


MetricValue = float | int | str


class GoalSeriesGranularity(Enum):
    """The granularity of a timeseries for goal metrics."""

    WEEK = "week"
    MONTH = "month"


class GoalMetricSeriesPoint(Model):
    """A point in the goal metric values timeseries."""

    date: date
    value: VerbatimOptional["MetricValue"]


class GoalValue(Model):
    """The current metric values for the goal on a team."""

    # do NOT change the wrapped type "MetricValue" to a direct reference,
    # base Model doesn't support wrapping a union with an optional form
    current: VerbatimOptional["MetricValue"]
    initial: VerbatimOptional["MetricValue"]
    target: Optional["MetricValue"]
    series: Optional[list[GoalMetricSeriesPoint]]
    series_granularity: Optional[str]  # valid values are from GoalSeriesGranularity


class TeamGoalTree(Model):
    """The team goal tree relative to a team and its descendants."""

    team: TeamTree
    value: GoalValue
    children: list["TeamGoalTree"]


class GoalTree(Model):
    """A goal attached to a tree of teams."""

    id: int
    name: str
    metric: str
    valid_from: date
    expires_at: date
    team_goal: TeamGoalTree
    repositories: Optional[list[str]]
    jira_projects: Optional[list[str]]
    jira_priorities: Optional[list[str]]
    jira_issue_types: Optional[list[str]]
