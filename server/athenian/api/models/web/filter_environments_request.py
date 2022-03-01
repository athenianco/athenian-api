from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties


class _FilterEnvironmentsRequest(Model, sealed=False):
    """Request body of `/filter/environments`. Filters for deployment environments."""

    openapi_types = {
        "repositories": List[str],
    }

    attribute_map = {
        "repositories": "repositories",
    }

    def __init__(
        self,
        repositories: Optional[List[str]] = None,
    ):
        """FilterEnvironmentsRequest - a model defined in OpenAPI

        :param repositories: The repositories of this FilterEnvironmentsRequest.
        """
        self._repositories = repositories

    @property
    def repositories(self) -> Optional[List[str]]:
        """Gets the repositories of this FilterEnvironmentsRequest.

        At least one repository in the list must be deployed in `[date_from, date_to)`.

        :return: The repositories of this FilterEnvironmentsRequest.
        """
        return self._repositories

    @repositories.setter
    def repositories(self, repositories: Optional[List[str]]):
        """Sets the repositories of this FilterEnvironmentsRequest.

        At least one repository in the list must be deployed in `[date_from, date_to)`.

        :param repositories: The repositories of this FilterEnvironmentsRequest.
        """
        self._repositories = repositories


FilterEnvironmentsRequest = AllOf(_FilterEnvironmentsRequest, CommonFilterProperties,
                                  name="FilterEnvironmentsRequest", module=__name__)
