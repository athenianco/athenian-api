from enum import Enum

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.commit_filter import _CommitFilter
from athenian.api.models.web.common_filter_properties import CommonFilterProperties


class _FilterCommitsProperty(Enum):
    """Primary commit filter modes."""

    NO_PR_MERGES = "no_pr_merges"
    BYPASSING_PRS = "bypassing_prs"
    EVERYTHING = "everything"


class _FilterCommitsRequest(Model, sealed=False):
    """Filter for listing commits."""

    property: str

    def validate_property(self, property: str) -> str:
        """Sets the property of this CodeFilter.

        Main trait of the commits - the core of the filter.

        :param property: The property of this CodeFilter.
        """
        if property is None:
            raise ValueError("Invalid value for `property`, must not be `None`")
        try:
            _FilterCommitsProperty(property)
        except ValueError:
            raise ValueError(
                "Invalid value for `property` - is not one of [%s]"
                % ",".join('"%s"' % f.value for f in _FilterCommitsProperty),
            ) from None

        return property


FilterCommitsRequest = AllOf(
    _FilterCommitsRequest,
    _CommitFilter,
    CommonFilterProperties,
    name="FilterCommitsRequest",
    module=__name__,
)
