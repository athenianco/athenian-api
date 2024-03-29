from http import HTTPStatus

from athenian.api.models.web.generic_error import GenericError


class MissingSettingsError(GenericError):
    """This class is auto generated by OpenAPI Generator (https://openapi-generator.tech)."""

    def __init__(
        self,
        title: str = HTTPStatus.FAILED_DEPENDENCY.phrase,
        status: int = HTTPStatus.FAILED_DEPENDENCY,
        detail: str = None,
    ):
        """MissingSettingsError - a model defined in OpenAPI

        :param title: The title of this MissingSettingsError.
        :param status: The status of this MissingSettingsError.
        :param detail: The detail of this MissingSettingsError.
        """
        super().__init__(
            type="/errors/MissingSettingsError", title=title, status=status, detail=detail,
        )
