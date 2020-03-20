from datetime import date
from enum import Enum
from typing import List

from athenian.api.models.web.commit_filter import CommitFilter


class FilterCommitsProperty(Enum):
    """Possible values of `FilterCommitsRequest.property`."""

    BYPASSING_PRS = "bypassing_prs"


class FilterCommitsRequest(CommitFilter):
    """Filter for listing commits."""

    def __init__(
        self,
        account: int = None,
        date_from: date = None,
        date_to: date = None,
        in_: List[str] = None,
        with_author: List[str] = None,
        with_committer: List[str] = None,
        property: str = None,
    ):
        """CodeFilter - a model defined in OpenAPI

        :param account: The account of this CodeFilter.
        :param date_from: The date_from of this CodeFilter.
        :param date_to: The date_to of this CodeFilter.
        :param in_: The in of this CodeFilter.
        :param with_author: The with_author of this CodeFilter.
        :param with_committer: The with_committer of this CodeFilter.
        :param property: The property of this CodeFilter.
        """
        super().__init__(account=account,
                         date_from=date_from,
                         date_to=date_to,
                         in_=in_,
                         with_author=with_author,
                         with_committer=with_committer)
        self.openapi_types["property"] = str
        self.attribute_map["property"] = "property"
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
        if property not in {f.value for f in FilterCommitsProperty}:
            raise ValueError("Invalid value for `property` - is not one of [%s]" %
                             ",".join('"%s"' % f.value for f in FilterCommitsProperty))

        self._property = property
