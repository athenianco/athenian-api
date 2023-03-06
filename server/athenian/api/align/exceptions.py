from http import HTTPStatus

from athenian.api.models.web import GenericError
from athenian.api.response import ResponseError


class GoalMutationError(ResponseError):
    """An error during a goal mutation handling."""

    def __init__(self, text, status_code=HTTPStatus.BAD_REQUEST):
        """Init the GoalMutationError."""
        wrapped_error = GenericError(
            type="/errors/align/GoalMutationError",
            status=status_code,
            detail=text,
            title="Goal mutation error",
        )
        super().__init__(wrapped_error)


class GoalNotFoundError(ResponseError):
    """A Goal was not found."""

    def __init__(self, goal_id: int):
        """Init the GoalNotFoundError."""
        wrapped_error = GenericError(
            type="/errors/align/GoalNotFoundError",
            status=HTTPStatus.NOT_FOUND,
            detail=f"Goal {goal_id} not found or access denied.",
            title="Goal not found",
        )
        super().__init__(wrapped_error)


class GoalTemplateNotFoundError(ResponseError):
    """A goal template was not found."""

    def __init__(self, template_id: int):
        """Init the GoalTemplateNotFoundError."""
        wrapped_error = GenericError(
            type="/errors/align/GoalTemplateNotFound",
            status=HTTPStatus.NOT_FOUND,
            detail=f"Goal template {template_id} not found",
            title="Goal template not found",
        )
        super().__init__(wrapped_error)


class TeamGoalNotFoundError(ResponseError):
    """A team - goal assignment was not found."""

    def __init__(self, goal_id, team_id: int):
        """Init the TeamGoalNotFound."""
        wrapped_error = GenericError(
            type="/errors/align/TeamGoalNotFound",
            status=HTTPStatus.NOT_FOUND,
            detail=f"Team {team_id} not assigned to goal {goal_id}",
            title="Team not assigned to Goal",
        )
        super().__init__(wrapped_error)
