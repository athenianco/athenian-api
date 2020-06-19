from datetime import date
from typing import List, Optional

from athenian.api.controllers.miners.github.commit import FilterCommitsProperty
from athenian.api.models.web.commit_filter import CommitFilter


class FilterCommitsRequest(CommitFilter):
    """Filter for listing commits."""

    openapi_types = CommitFilter.openapi_types.copy()
    openapi_types["property"] = str
    attribute_map = CommitFilter.attribute_map.copy()
    attribute_map["property"] = "property"

    def __init__(
        self,
        account: Optional[int] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        in_: Optional[List[str]] = None,
        with_author: Optional[List[str]] = None,
        with_committer: Optional[List[str]] = None,
        property: Optional[str] = None,
    ):
        """CodeFilter - a model defined in OpenAPI

        :param account: The account of this CodeFilter.
        :param date_from: The date_from of this CodeFilter.
        :param date_to: The date_to of this CodeFilter.
        :param timezone: The timezone of this CodeFilter.
        :param in_: The in of this CodeFilter.
        :param with_author: The with_author of this CodeFilter.
        :param with_committer: The with_committer of this CodeFilter.
        :param property: The property of this CodeFilter.
        """
        super().__init__(account=account,
                         date_from=date_from,
                         date_to=date_to,
                         timezone=timezone,
                         in_=in_,
                         with_author=with_author,
                         with_committer=with_committer)
        self._property = property

    @property
    def property(self) -> str:
        """Gets the property of this CodeFilter.

        Main trait of the commits - the core of the filter.

        :return: The property of this CodeFilter.
        """
        return self._property

    @property.setter
    def property(self, property: str):
        """Sets the property of this CodeFilter.

        Main trait of the commits - the core of the filter.

        :param property: The property of this CodeFilter.
        """
        if property is None:
            raise ValueError("Invalid value for `property`, must not be `None`")
        try:
            FilterCommitsProperty(property)
        except ValueError:
            raise ValueError("Invalid value for `property` - is not one of [%s]" %
                             ",".join('"%s"' % f.value for f in FilterCommitsProperty)) from None

        self._property = property
