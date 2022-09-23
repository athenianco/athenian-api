from datetime import date
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.team_tree import TeamTree


class AlignGoalsRequest(Model):
    """Definition of the goal-metric."""

    account: int
    team: int
    only_with_targets: bool = False


MetricValue = float | int | str


class GoalValue(Model):
    """The current metric values for the goal on a team."""

    current: MetricValue
    initial: MetricValue
    target: Optional[MetricValue]


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
