from typing import Any, Optional

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


class GraphQLMetricValue(Model):
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
        """Initialize a new instance of GraphQLMetricValue."""
        self._int_ = self._str_ = self._float_ = None
        if isinstance(value, (int, np.integer)):
            self._int_ = value
        elif isinstance(value, float):
            self._float_ = value
        else:
            self._str_ = value


class GraphQLTeamTree(TeamTree):
    """Wraps the generic TeamTree model to add camel case attribute map for GraphQL."""

    attribute_types: dict = {}
    attribute_map = {
        "members_count": "membersCount",
        "total_teams_count": "totalTeamsCount",
        "total_members_count": "totalMembersCount",
    }

    @classmethod
    def from_team_tree(cls, team_tree: TeamTree) -> "GraphQLTeamTree":
        """Build a GraphQLTeamTree from a base TeamTree."""
        kwargs = {
            name: getattr(team_tree, name)
            for name in team_tree.attribute_types
            if name != "children"
        }
        kwargs["children"] = [cls.from_team_tree(child) for child in team_tree.children]
        return cls(**kwargs)


class GraphQLTeamMetricValue(Model):
    """Team metric value tree node."""

    attribute_types = {
        "team": TeamTree,
        "value": GraphQLMetricValue,
        "children": list[Model],  # list[GraphQLTeamMetricValue],
    }

    attribute_map = {
        "team_id": "teamId",
    }


GraphQLTeamMetricValue.attribute_types["children"] = list[GraphQLTeamMetricValue]


class GraphQLMetricValues(Model):
    """Response from metricsCurrentValues(), a specific metric team tree."""

    metric: str
    value: GraphQLTeamMetricValue


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


class Member(Contributor):
    """A member of a team.

    This is equal to OpenAPI Contributor model, except that jira_user is aliased to jiraUser.
    """

    @property
    def jiraUser(self) -> Optional[str]:
        """Alias of self.jira_user."""
        return self.jira_user
