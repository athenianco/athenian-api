from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.for_set_common import CommonPullRequestFilters
from athenian.api.models.web.release_with import ReleaseWith


class _FilterReleasesRequest(Model, sealed=False):
    """Structure to specify the filter traits of releases."""

    attribute_types = {
        "in_": List[str],
        "with_": Optional[ReleaseWith],
    }

    attribute_map = {
        "in_": "in",
        "with_": "with",
    }

    def __init__(
        self,
        in_: Optional[List[str]] = None,
        with_: Optional[ReleaseWith] = None,
    ):
        """FilterReleasesRequest - a model defined in OpenAPI

        :param in_: The in of this FilterReleasesRequest.
        :param with_: The with of this FilterReleasesRequest.
        """
        self._in_ = in_
        self._with_ = with_

    @property
    def in_(self) -> List[str]:
        """Gets the in_ of this FilterReleasesRequest.

        :return: The in_ of this FilterReleasesRequest.
        """
        return self._in_

    @in_.setter
    def in_(self, in_: List[str]):
        """Sets the in_ of this FilterReleasesRequest.

        :param in_: The in_ of this FilterReleasesRequest.
        """
        self._in_ = in_

    @property
    def with_(self) -> Optional[ReleaseWith]:
        """Gets the with_ of this FilterReleasesRequest.

        Release contribution roles.

        :return: The with_ of this FilterReleasesRequest.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[ReleaseWith]):
        """Sets the with_ of this FilterReleasesRequest.

        Release contribution roles.

        :param with_: The with_ of this FilterReleasesRequest.
        """
        self._with_ = with_


FilterReleasesRequest = AllOf(_FilterReleasesRequest,
                              CommonFilterProperties,
                              CommonPullRequestFilters,
                              name="FilterReleasesRequest", module=__name__)
