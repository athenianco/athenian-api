from http import HTTPStatus
from typing import Optional

from athenian.api.models.web.generic_error import _GenericError


class InvalidRequestError(_GenericError):
    """HTTP 400 with a pointer to the request body where an error exists."""

    pointer: str

    def __init__(
        self,
        pointer: str,
        detail: Optional[str] = None,
        instance: Optional[str] = None,
    ):
        """InvalidRequestError - a model defined in OpenAPI

        :param title: The title of this InvalidRequestError.
        :param status: The status of this InvalidRequestError.
        :param detail: The detail of this InvalidRequestError.
        :param instance: The instance of this InvalidRequestError.
        :param pointer: The pointer of this InvalidRequestError.
        """
        super().__init__(
            type="/errors/InvalidRequestError",
            title=HTTPStatus.BAD_REQUEST.phrase,
            status=HTTPStatus.BAD_REQUEST,
            detail=detail,
            instance=instance,
            pointer=pointer,
        )

    @classmethod
    def from_validation_error(cls, e: ValueError) -> "InvalidRequestError":
        """Convert the validation exception in Model.from_dict() to InvalidRequestError."""
        return cls(getattr(e, "path", "?"), detail=str(e))
