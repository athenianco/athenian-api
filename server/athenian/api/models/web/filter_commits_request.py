from enum import Enum
from typing import Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.commit_filter import _CommitFilter
from athenian.api.models.web.common_filter_properties import CommonFilterProperties


class _FilterCommitsProperty(Enum):
    """Primary commit filter modes."""

    NO_PR_MERGES = "no_pr_merges"
    BYPASSING_PRS = "bypassing_prs"


class _FilterCommitsRequest(Model):
    """Filter for listing commits."""

    openapi_types = {
        "property": str,
    }

    attribute_map = {
        "property": "property",
    }

    __enable_slots__ = False

    def __init__(
        self,
        property: Optional[str] = None,
    ):
        """CodeFilter - a model defined in OpenAPI

        :param property: The property of this CodeFilter.
        """
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
            _FilterCommitsProperty(property)
        except ValueError:
            raise ValueError("Invalid value for `property` - is not one of [%s]" %
                             ",".join('"%s"' % f.value for f in _FilterCommitsProperty)) from None

        self._property = property


FilterCommitsRequest = AllOf(_FilterCommitsRequest, _CommitFilter, CommonFilterProperties,
                             name="FilterCommitsRequest", module=__name__)
