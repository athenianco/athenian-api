from datetime import date
from typing import List

from athenian.api.models.web.commit_filter import CommitFilter
from athenian.api.models.web.granularity import Granularity


class CodeFilter(CommitFilter):
    """Filter for revealing code bypassing PRs."""

    def __init__(
        self,
        account: int = None,
        date_from: date = None,
        date_to: date = None,
        in_: List[str] = None,
        with_author: List[str] = None,
        with_committer: List[str] = None,
        granularity: str = None,
    ):
        """CodeFilter - a model defined in OpenAPI

        :param account: The account of this CodeFilter.
        :param date_from: The date_from of this CodeFilter.
        :param date_to: The date_to of this CodeFilter.
        :param in_: The in of this CodeFilter.
        :param with_author: The with_author of this CodeFilter.
        :param with_committer: The with_committer of this CodeFilter.
        :param granularity: The granularity of this CodeFilter.
        """
        super().__init__(account=account,
                         date_from=date_from,
                         date_to=date_to,
                         in_=in_,
                         with_author=with_author,
                         with_committer=with_committer)
        self.openapi_types["granularity"] = str
        self.attribute_map["granularity"] = "granularity"
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
