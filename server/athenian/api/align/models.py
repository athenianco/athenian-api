from __future__ import annotations

from datetime import date
from typing import Any, Optional, Union

import numpy as np

from athenian.api.models.web import Contributor, TeamTree
from athenian.api.models.web.base_model_ import Enum, Model


class GoalRemoveStatus(Model):
    """The status of the GraphQL removeGoal mutation."""

    attribute_types = {
        "success": bool,
    }

    def __init__(self, success: bool):
        """Init the GoalRemoveStatus."""
        self._success = success

    @property
    def success(self) -> bool:
        """Whether the removal operation was successful or not."""
        return self._success


class MutateGoalResultGoal(Model):
    """The goal nested inside MutateGoalResult."""

    attribute_types = {
        "id": int,
    }

    def __init__(self, id: int):
        """Init the MutateGoalResultGoal."""
        self._id = id

    @property
    def id(self) -> int:
        """Get the id of the goal."""
        return self._id


class MutateGoalResult(Model):
    """The result of a GraphQL mutation about a goal."""

    attribute_types = {
        "goal": MutateGoalResultGoal,
    }

    def __init__(self, goal: MutateGoalResultGoal):
        """Init the MutateGoalResultGoal."""
        self._goal = goal

    @property
    def goal(self) -> MutateGoalResultGoal:
        """Get the id of the goal."""
        return self._goal


class GraphQLTeamTree(TeamTree):
    """Wraps the generic TeamTree model to add camel case attribute map for GraphQL."""

    attribute_types: dict = {}
    attribute_map = {
        "members_count": "membersCount",
        "total_teams_count": "totalTeamsCount",
        "total_members_count": "totalMembersCount",
    }

    @classmethod
    def from_team_tree(cls, team_tree: TeamTree) -> GraphQLTeamTree:
        """Build a GraphQLTeamTree from a base TeamTree."""
        kwargs = {
            name: getattr(team_tree, name)
            for name in team_tree.attribute_types
            if name != "children"
        }
        kwargs["children"] = [cls.from_team_tree(child) for child in team_tree.children]
        return cls(**kwargs)


class MetricValue(Model):
    """The value for a given team and metric."""

    attribute_types = {
        "int_": Optional[int],
        "str_": Optional[object],
        "float_": Optional[float],
    }

    attribute_map = {
        "int_": "int",
        "str_": "str",
        "float_": "float",
    }

    def __init__(self, value: Any):
        """Initialize a new instance of MetricValue."""
        self._int_ = self._str_ = self._float_ = None
        if isinstance(value, (int, np.integer)):
            self._int_ = value
        elif isinstance(value, float):
            self._float_ = value
        else:
            self._str_ = value

    @property
    def int_(self) -> Optional[Union[int, np.integer]]:
        """Return the metric value as an integer."""
        return self._int_

    @property
    def str_(self) -> Optional[object]:
        """Return the metric value as a string."""
        return self._str_

    @property
    def float_(self) -> float:
        """Return the metric value as a floating point number."""
        return self._float_


class TeamMetricValue(Model):
    """Team metric value tree node."""

    attribute_types = {
        "team": TeamTree,
        "value": MetricValue,
        "children": list[Model],  # list[TeamMetricValue],
    }

    attribute_map = {
        "team_id": "teamId",
    }

    def __init__(self, team: TeamTree, value: MetricValue, children: list[TeamMetricValue]):
        """Initialize a new instance of TeamMetricValue."""
        self._team = team
        self._value = value
        self._children = children

    @property
    def team(self) -> TeamTree:
        """Return the team relative to this metric value."""
        return self._team

    @property
    def value(self) -> MetricValue:
        """Return the metric value."""
        return self._value

    @property
    def children(self) -> list[TeamMetricValue]:
        """Return the list of child team metrics."""
        return self._children


TeamMetricValue.attribute_types["children"] = list[TeamMetricValue]


class MetricValues(Model):
    """Response from metricsCurrentValues(), a specific metric team tree."""

    attribute_types = {
        "metric": str,
        "value": TeamMetricValue,
    }

    def __init__(self, metric: str, value: TeamMetricValue):
        """Init the MetricValues."""
        self._metric = metric
        self._value = value

    @property
    def metric(self) -> str:
        """Return the metric ID."""
        return self._metric

    @property
    def value(self) -> TeamMetricValue:
        """Return the team tree of metric values."""
        return self._value


class _GoalMetricFilters(metaclass=Enum):  # noqa: PIE795
    repositories = "repositories"
    jiraProjects = "jiraProjects"
    jiraPriorities = "jiraPriorities"
    jiraIssueTypes = "jiraIssueTypes"


class _BaseGoalInputFields(_GoalMetricFilters):
    name = "name"
    metric = "metric"


class CreateGoalInputFields(_BaseGoalInputFields):
    """Fields definitions for GraphQL CreateGoalInput type."""

    teamGoals = "teamGoals"
    validFrom = "validFrom"
    expiresAt = "expiresAt"


class TeamGoalInputFields(metaclass=Enum):
    """Fields definitions for GraphQL TeamGoalInput type."""

    teamId = "teamId"
    target = "target"


class UpdateRepositoriesInputFields(metaclass=Enum):
    """Fields of the input to update Goal's repositories."""

    value = "value"


class UpdateGoalInputFields(_BaseGoalInputFields):
    """Fields definitions for GraphQL UpdateGoalInput type."""

    goalId = "goalId"
    archived = "archived"
    teamGoalChanges = "teamGoalChanges"


class TeamGoalChangeFields(metaclass=Enum):
    """Fields definitions for GraphQL TeamGoalChange type."""

    teamId = "teamId"
    target = "target"
    remove = "remove"


class MetricParamsFields(_GoalMetricFilters):
    """Fields definitions for GraphQL MetricParams type."""

    teamId = "teamId"
    metrics = "metrics"
    validFrom = "validFrom"
    expiresAt = "expiresAt"


class GoalValue(Model):
    """The current metric values for the goal on a team."""

    attribute_types = {
        "current": MetricValue,
        "initial": MetricValue,
        "target": Optional[MetricValue],
    }

    def __init__(self, current: MetricValue, initial: MetricValue, target: Optional[MetricValue]):
        """Init the GoalValue."""
        self._current = current
        self._initial = initial
        self._target = target

    @property
    def current(self) -> MetricValue:
        """Get current metric value."""
        return self._current

    @property
    def initial(self) -> MetricValue:
        """Get initial metric value."""
        return self._initial

    @property
    def target(self) -> Optional[MetricValue]:
        """Get target metric value."""
        return self._target


class TeamGoalTree(Model):
    """The team goal tree relative to a team and its descendants."""

    attribute_types = {
        "team": TeamTree,
        "value": GoalValue,
        "children": list[Model],  # list[TeamGoalTree]
    }

    def __init__(self, team: TeamTree, value: GoalValue, children: list[TeamGoalTree]):
        """Init the TeamGoalTree."""
        self._team = team
        self._value = value
        self._children = children

    @property
    def team(self) -> TeamTree:
        """Get the team this node is applied to."""
        return self._team

    @property
    def value(self) -> GoalValue:
        """Get the GoalValue attached to the team, if any."""
        return self._value

    @property
    def children(self) -> list[TeamGoalTree]:
        """Get the list of TeamGoalTree for the Team children."""
        return self._children


TeamGoalTree.attribute_types["children"] = list[TeamGoalTree]


class GoalTree(Model):
    """A goal attached to a tree of teams."""

    attribute_types = {
        "id": int,
        "name": str,
        "metric": str,
        "valid_from": date,
        "expires_at": date,
        "team_goal": TeamGoalTree,
        "repositories": Optional[list[str]],
        "jira_projects": Optional[list[str]],
        "jira_priorities": Optional[list[str]],
        "jira_issue_types": Optional[list[str]],
    }

    attribute_map = {
        "valid_from": "validFrom",
        "expires_at": "expiresAt",
        "team_goal": "teamGoal",
        "jira_projects": "jiraProjects",
        "jira_priorities": "jiraPriorities",
        "jira_issue_types": "jiraIssueTypes",
    }

    def __init__(
        self,
        id: int,
        name: str,
        metric: str,
        valid_from: date,
        expires_at: date,
        team_goal: TeamGoalTree,
        repositories: Optional[list[str]],
        jira_projects: Optional[list[str]],
        jira_priorities: Optional[list[str]],
        jira_issue_types: Optional[list[str]],
    ):
        """Init the GoalTree."""
        self._id = id
        self._name = name
        self._metric = metric
        self._valid_from = valid_from
        self._expires_at = expires_at
        self._team_goal = team_goal
        self._repositories = repositories
        self._jira_projects = jira_projects
        self._jira_priorities = jira_priorities
        self._jira_issue_types = jira_issue_types

    @property
    def id(self) -> int:
        """Get the identifier of the goal."""
        return self._id

    @property
    def name(self) -> str:
        """Get the name of the goal."""
        return self._name

    @property
    def metric(self) -> str:
        """Get the metric of the goal."""
        return self._metric

    @property
    def valid_from(self) -> date:
        """Get the valid from date of the goal."""
        return self._valid_from

    @property
    def expires_at(self) -> date:
        """Get the epxire date of the goal."""
        return self._expires_at

    @property
    def team_goal(self) -> TeamGoalTree:
        """Get the root of the `TeamGoalTree` attached to the goal."""
        return self._team_goal

    @property
    def repositories(self) -> Optional[list[str]]:
        """Get the repositories defining the goal scope."""
        return self._repositories

    @property
    def jira_projects(self) -> Optional[list[str]]:
        """Get the JIRA projects defining the goal scope."""
        return self._jira_projects

    @property
    def jira_priorities(self) -> Optional[list[str]]:
        """Get the JIRA priorities defining the goal scope."""
        return self._jira_priorities

    @property
    def jira_issue_types(self) -> Optional[list[str]]:
        """Get the JIRA issue types defining the goal scope."""
        return self._jira_issue_types


class Member(Contributor):
    """A member of a team.

    This is equal to OpenAPI Contributor model, except that jira_user is aliased to jiraUser.
    """

    @property
    def jiraUser(self) -> Optional[str]:
        """Alias of self.jira_user."""
        return self.jira_user
