from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.commit_filter import _CommitFilter
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.granularity import Granularity


class _CodeFilter(Model, sealed=False):
    """Filter for revealing code bypassing PRs."""

    attribute_types = {
        "granularity": str,
    }

    attribute_map = {
        "granularity": "granularity",
    }

    def __init__(
        self,
        granularity: Optional[str] = None,
    ):
        """CodeFilter - a model defined in OpenAPI

        :param granularity: The granularity of this CodeFilter.
        """
        self._granularity = granularity

    @property
    def granularity(self) -> str:
        """Gets the granularity of this CodeFilter.

        How often the metrics are reported. The value must satisfy the following regular
        expression: (^([1-9]\\d* )?(day|week|month|year)$

        :return: The granularity of this CodeFilter.
        """
        return self._granularity

    @granularity.setter
    def granularity(self, granularity: str):
        """Sets the granularity of this CodeFilter.

        How often the metrics are reported. The value must satisfy the following regular
        expression: (^([1-9]\\d* )?(day|week|month|year)$

        :param granularity: The granularity of this CodeFilter.
        """
        if granularity is None:
            raise ValueError("Invalid value for `granularity`, must not be `None`")
        if not Granularity.format.match(granularity):
            raise ValueError("Invalid value for `granularity`, does not match /%s/" %
                             Granularity.format.pattern)

        self._granularity = granularity


CodeFilter = AllOf(_CodeFilter, _CommitFilter, CommonFilterProperties,
                   name="CodeFilter", module=__name__)
