from typing import Optional

from athenian.api.models.web import CommonFilterProperties
from athenian.api.models.web.base_model_ import AllOf, Model


class _FilterContributorsRequest(Model, sealed=False):
    """Filters for `/filter/contributors`."""

    in_: (Optional[list[str]], "in")
    as_: (Optional[list[str]], "as")

    def validate_as_(self, as_: Optional[list[str]]) -> Optional[list[str]]:
        """Sets the as_ of this FilterContributorsRequest.

        :param as_: The as_ of this FilterContributorsRequest.
        """
        if as_ is None:
            return None
        allowed_values = {
            "author",
            "reviewer",
            "commit_author",
            "commit_committer",
            "commenter",
            "merger",
            "releaser",
        }
        if not set(as_).issubset(allowed_values):
            raise ValueError(
                "Invalid values for `as_` [%s], must be a subset of [%s]"
                % (", ".join(set(as_) - allowed_values), ", ".join(allowed_values)),
            )

        return as_


FilterContributorsRequest = AllOf(
    _FilterContributorsRequest,
    CommonFilterProperties,
    name="FilterContributorsRequest",
    module=__name__,
)
