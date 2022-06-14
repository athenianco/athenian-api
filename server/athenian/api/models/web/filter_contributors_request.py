from typing import List, Optional

from athenian.api.models.web import CommonFilterProperties
from athenian.api.models.web.base_model_ import AllOf, Model


class _FilterContributorsRequest(Model, sealed=False):
    """Filters for `/filter/contributors`."""

    attribute_types = {
        "in_": List[str],
        "as_": List[str],
    }

    attribute_map = {
        "in_": "in",
        "as_": "as",
    }

    def __init__(
        self,
        in_: Optional[List[str]] = None,
        as_: List[str] = None,
    ):
        """FilterContributorsRequest - a model defined in OpenAPI

        :param in_: The in_ of this FilterContributorsRequest.
        :param as_: The as_ of this FilterContributorsRequest.
        """
        self._in_ = in_
        self._as_ = as_

    @property
    def in_(self) -> List[str]:
        """Gets the in_ of this FilterContributorsRequest.

        A set of repositories. An empty list results an empty response in contrary to DeveloperSet.
        Duplicates are automatically ignored.

        :return: The in_ of this FilterContributorsRequest.
        """
        return self._in_

    @in_.setter
    def in_(self, in_: List[str]):
        """Sets the in_ of this FilterContributorsRequest.

        A set of repositories. An empty list results an empty response in contrary to DeveloperSet.
        Duplicates are automatically ignored.

        :param in_: The in_ of this FilterContributorsRequest.
        """
        self._in_ = in_

    @property
    def as_(self) -> List[str]:
        """Gets the as_ of this FilterContributorsRequest.

        :return: The as_ of this FilterContributorsRequest.
        """
        return self._as_

    @as_.setter
    def as_(self, as_: List[str]):
        """Sets the as_ of this FilterContributorsRequest.

        :param as_: The as_ of this FilterContributorsRequest.
        """
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
                % (", ".join(set(as_) - allowed_values), ", ".join(allowed_values),),
            )

        self._as_ = as_


FilterContributorsRequest = AllOf(
    _FilterContributorsRequest,
    CommonFilterProperties,
    name="FilterContributorsRequest",
    module=__name__,
)
