from __future__ import annotations

from datetime import date
from itertools import chain
from typing import Any, List, Optional, Union

import numpy as np

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


class TeamTree(Model):
    """A team with the tree of child teams."""

    attribute_types = {
        "id": int,
        "name": str,
        "members_count": int,
        "total_teams_count": int,
        "total_members_count": int,
        "children": List[Model],  # List[TeamTree],
        "members": List[int],
        "total_members": List[int],
    }

    attribute_map = {
        "members_count": "membersCount",
        "total_teams_count": "totalTeamsCount",
        "total_members_count": "totalMembersCount",
    }

    def __init__(
        self,
        id: int,
        name: str,
        children: List[TeamTree],
        members: List[int],
    ):
        """Init the TeamTree."""
        self._id = id
        self._name = name
        self._members = members
        self._children = children

        self._members_count = len(members)
        self._total_members = sorted(
            set(chain(members, *(child.total_members for child in children))),
        )
        self._total_teams_count = sum(
            child.total_teams_count for child in children
        ) + len(children)

    @property
    def id(self) -> int:
        """Get the identifier of the team."""
        return self._id

    @property
    def name(self) -> str:
        """Get the name of the team."""
        return self._name

    @property
    def members_count(self) -> int:
        """Get the number of members directly included in the team."""
        return self._members_count

    @property
    def total_teams_count(self) -> int:
        """Get the number of teams included in the team tree."""
        return self._total_teams_count

    @property
    def total_members_count(self) -> int:
        """Get the number of team members included in the team tree."""
        return len(self._total_members)

    @property
    def children(self) -> List[TeamTree]:
        """Get the direct child teams of the team."""
        return self._children

    @property
    def total_members(self) -> List[int]:
        """Get the team members recursively included in the team tree."""
        return self._total_members

    @property
    def members(self) -> List[int]:
        """Get the directly contained members of the team."""
        return self._members


TeamTree.attribute_types["children"] = List[TeamTree]


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
        "children": List[Model],  # List[TeamMetricValue],
    }

    attribute_map = {
        "team_id": "teamId",
    }

    def __init__(self, team: TeamTree, value: MetricValue, children: List[TeamMetricValue]):
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
    def children(self) -> List[TeamMetricValue]:
        """Return the list of child team metrics."""
        return self._children


TeamMetricValue.attribute_types["children"] = List[TeamMetricValue]


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


class CreateGoalInputFields(metaclass=Enum):
    """Fields definitions for GraphQL CreateGoalInput type."""

    templateId = "templateId"
    teamGoals = "teamGoals"
    validFrom = "validFrom"
    expiresAt = "expiresAt"


class TeamGoalInputFields(metaclass=Enum):
    """Fields definitions for GraphQL TeamGoalInput type."""

    teamId = "teamId"
    target = "target"


class UpdateGoalInputFields(metaclass=Enum):
    """Fields definitions for GraphQL UpdateGoalInput type."""

    goalId = "goalId"
    teamGoalChanges = "teamGoalChanges"


class TeamGoalChangeFields(metaclass=Enum):
    """Fields definitions for GraphQL TeamGoalChange type."""

    teamId = "teamId"
    target = "target"
    remove = "remove"


class MetricParamsFields(metaclass=Enum):
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
        "target": MetricValue,
    }

    def __init__(self, current: MetricValue, initial: MetricValue, target: MetricValue):
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
    def target(self) -> MetricValue:
        """Get target metric value."""
        return self._target


class TeamGoalTree(Model):
    """The team goal tree relative to a team and its descendants."""

    attribute_types = {
        "team": TeamTree,
        "value": Optional[GoalValue],
        "children": List[Model],  # List[TeamGoalTree]
    }

    def __init__(self, team: TeamTree, value: Optional[GoalValue], children: List[TeamGoalTree]):
        """Init the TeamGoalTree."""
        self._team = team
        self._value = value
        self._children = children

    @property
    def team(self) -> TeamTree:
        """Get the team this node is applied to."""
        return self._team

    @property
    def value(self) -> Optional[GoalValue]:
        """Get the GoalValue attached to the team, if any."""
        return self._value

    @property
    def children(self) -> List[TeamGoalTree]:
        """Get the list of TeamGoalTree for the Team children."""
        return self._children


TeamGoalTree.attribute_types["children"] = List[TeamGoalTree]


class GoalTree(Model):
    """A goal attached to a tree of teams."""

    attribute_types = {
        "id": int,
        "template_id": int,
        "valid_from": date,
        "expires_at": date,
        "team_goal": TeamGoalTree,
    }

    attribute_map = {
        "template_id": "templateId",
        "valid_from": "validFrom",
        "expires_at": "expiresAt",
        "team_goal": "teamGoal",
    }

    def __init__(
        self,
        id: int,
        template_id: int,
        valid_from: date,
        expires_at: date,
        team_goal: TeamGoalTree,
    ):
        """Init the GoalTree."""
        self._id = id
        self._template_id = template_id
        self._valid_from = valid_from
        self._expires_at = expires_at
        self._team_goal = team_goal

    @property
    def id(self) -> int:
        """Get the identifier of the goal."""
        return self._id

    @property
    def template_id(self) -> int:
        """Get the template identifier of the goal."""
        return self._template_id

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
