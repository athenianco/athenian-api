from http import HTTPStatus
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class _GenericError(Model, sealed=False):
    type: str
    title: str
    status: Optional[int]
    detail: Optional[str]
    instance: Optional[str]


class GenericError(_GenericError):
    """Base HTTP error."""


class BadRequestError(GenericError):
    """HTTP 400."""

    def __init__(self, detail: Optional[str] = None):
        """Initialize a new instance of BadRequestError.

        :param detail: The details about this error.
        """
        super().__init__(
            type="/errors/BadRequest",
            title=HTTPStatus.BAD_REQUEST.phrase,
            status=HTTPStatus.BAD_REQUEST,
            detail=detail,
        )


class NotFoundError(GenericError):
    """HTTP 404."""

    def __init__(self, detail: Optional[str] = None, type_: str = "/errors/NotFoundError"):
        """Initialize a new instance of NotFoundError.

        :param detail: The details about this error.
        """
        super().__init__(
            type=type_,
            title=HTTPStatus.NOT_FOUND.phrase,
            status=HTTPStatus.NOT_FOUND,
            detail=detail,
        )


class ForbiddenError(GenericError):
    """HTTP 403."""

    def __init__(self, detail: Optional[str] = None):
        """Initialize a new instance of ForbiddenError.

        :param detail: The details about this error.
        """
        super().__init__(
            type="/errors/ForbiddenError",
            title=HTTPStatus.FORBIDDEN.phrase,
            status=HTTPStatus.FORBIDDEN,
            detail=detail,
        )


class UnauthorizedError(GenericError):
    """HTTP 401."""

    def __init__(self, detail: Optional[str] = None):
        """Initialize a new instance of UnauthorizedError.

        :param detail: The details about this error.
        """
        super().__init__(
            type="/errors/Unauthorized",
            title=HTTPStatus.UNAUTHORIZED.phrase,
            status=HTTPStatus.UNAUTHORIZED,
            detail=detail,
        )


class DatabaseConflict(GenericError):
    """HTTP 409."""

    def __init__(self, detail: Optional[str] = None):
        """Initialize a new instance of DatabaseConflict.

        :param detail: The details about this error.
        """
        super().__init__(
            type="/errors/DatabaseConflict",
            title=HTTPStatus.CONFLICT.phrase,
            status=HTTPStatus.CONFLICT,
            detail=detail,
        )


class TooManyRequestsError(GenericError):
    """HTTP 429."""

    def __init__(self, detail: Optional[str] = None, type="/errors/TooManyRequestsError"):
        """Initialize a new instance of TooManyRequestsError.

        :param detail: The details about this error.
        :param type: The type identifier of this error.
        """
        super().__init__(
            type=type,
            title=HTTPStatus.TOO_MANY_REQUESTS.phrase,
            status=HTTPStatus.TOO_MANY_REQUESTS,
            detail=detail,
        )


class ServerNotImplementedError(GenericError):
    """HTTP 501."""

    def __init__(self, detail="This API endpoint is not implemented yet."):
        """Initialize a new instance of ServerNotImplementedError.

        :param detail: The details about this error.
        """
        super().__init__(
            type="/errors/NotImplemented",
            title=HTTPStatus.NOT_IMPLEMENTED.phrase,
            status=HTTPStatus.NOT_IMPLEMENTED,
            detail=detail,
        )


class ServiceUnavailableError(GenericError):
    """HTTP 503."""

    def __init__(self, type: str, detail: Optional[str], instance: Optional[str] = None):
        """Initialize a new instance of ServiceUnavailableError.

        :param detail: The details about this error.
        :param type: The type identifier of this error.
        :param instance: Sentry event ID of this error.
        """
        super().__init__(
            type=type,
            title=HTTPStatus.SERVICE_UNAVAILABLE.phrase,
            status=HTTPStatus.SERVICE_UNAVAILABLE,
            detail=detail,
            instance=instance,
        )
