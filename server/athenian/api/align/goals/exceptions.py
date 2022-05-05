from http import HTTPStatus

from athenian.api.models.web import GenericError
from athenian.api.response import ResponseError


class GoalMutationError(ResponseError):
    """An error during a goal mutation handling."""

    def __init__(self, text):
        """Init the GoalMutationError."""
        wrapped_error = GenericError(
            type="/errors/align/GoalMutationError",
            status=HTTPStatus.BAD_REQUEST,
            detail=text,
            title="Goal mutation error",
        )
        super().__init__(wrapped_error)
