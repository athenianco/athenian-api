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
