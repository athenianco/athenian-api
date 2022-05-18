from athenian.api.models.web.base_model_ import Model


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
