from __future__ import annotations

from itertools import chain
from typing import List

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


class TeamMetricValue(Model):
    """Team metric value tree node."""

    attribute_types = {
        "team_id": int,
        "value": object,
        "children": List[Model],  # List[TeamMetricValue],
    }

    attribute_map = {
        "team_id": "teamId",
    }

    def __init__(self, team_id: int, value: object, children: List[TeamMetricValue]):
        """Initialize a new instance of TeamMetricValue."""
        self._team_id = team_id
        self._value = value
        self._children = children

    @property
    def team_id(self) -> int:
        """Return the team identifier."""
        return self._team_id

    @property
    def value(self) -> object:
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
    def matric(self) -> str:
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


class TeamTree(Model):
    """A team with the tree of child teams."""

    attribute_types = {
        "id": int,
        "name": str,
        "total_teams_count": int,
        "total_members_count": int,
        "children": List[Model],  # List[Team],
        "members": List[int],
        "total_members": List[int],
    }

    attribute_map = {
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


class MetricParamsFields(metaclass=Enum):
    """Fields definitions for GraphQL MetricParams type."""

    teamId = "teamId"
    metrics = "metrics"
    validFrom = "validFrom"
    expiresAt = "expiresAt"
