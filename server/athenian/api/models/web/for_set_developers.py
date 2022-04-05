from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.for_set_common import CommonPullRequestFilters, RepositoryGroupsMixin


class _ForSetDevelopers(Model, RepositoryGroupsMixin, sealed=False):
    """Filter for `/metrics/developers`."""

    openapi_types = {
        "repositories": List[str],
        "repogroups": Optional[List[List[int]]],
        "developers": List[str],
        "aggregate_devgroups": Optional[List[List[int]]],
    }

    attribute_map = {
        "repositories": "repositories",
        "repogroups": "repogroups",
        "developers": "developers",
        "aggregate_devgroups": "aggregate_devgroups",
    }

    def __init__(
        self,
        repositories: Optional[List[str]] = None,
        repogroups: Optional[List[List[int]]] = None,
        developers: Optional[List[str]] = None,
        aggregate_devgroups: Optional[List[List[int]]] = None,
    ):
        """_ForSetDevelopers - a model defined in OpenAPI

        :param repositories: The repositories of this ForSetDevelopers.
        :param repogroups: The repogroups of this ForSetDevelopers.
        :param developers: The developers of this ForSetDevelopers.
        :param aggregate_devgroups: The aggregate_devgroups of this ForSetDevelopers.
        """
        self._repositories = repositories
        self._repogroups = repogroups
        self._developers = developers
        self._aggregate_devgroups = aggregate_devgroups

    @property
    def developers(self) -> List[str]:
        """Gets the developers of this ForSetDevelopers.

        :return: The developers of this ForSetDevelopers.
        """
        return self._developers

    @developers.setter
    def developers(self, developers: List[str]):
        """Sets the developers of this ForSetDevelopers.

        :param developers: The developers of this ForSetDevelopers.
        """
        if developers is None:
            raise ValueError("Invalid value for `developers`, must not be `None`")
        if len(developers) == 0:
            raise ValueError("Invalid value for `developers`, must not be an empty list")

        self._developers = developers

    @property
    def aggregate_devgroups(self) -> Optional[List[List[int]]]:
        """Gets the aggregate_devgroups of this ForSetDevelopers.

        :return: The aggregate_devgroups of this ForSetDevelopers.
        """
        return self._aggregate_devgroups

    @aggregate_devgroups.setter
    def aggregate_devgroups(self, aggregate_devgroups: Optional[List[List[int]]]):
        """Sets the aggregate_devgroups of this ForSetDevelopers.

        :param aggregate_devgroups: The aggregate_devgroups of this ForSetDevelopers.
        """
        if aggregate_devgroups is not None:
            if len(aggregate_devgroups) == 0:
                raise ValueError("`aggregate_devgroups` must contain at least one list")
            for i, group in enumerate(aggregate_devgroups):
                if len(group) == 0:
                    raise ValueError(
                        "`aggregate_devgroups[%d]` must contain at least one element" % i)
                for j, v in enumerate(group):
                    if v < 0:
                        raise ValueError(
                            "`aggregate_devgroups[%d][%d]` = %s must not be negative" % (i, j, v))
                    if self._developers is not None and v >= len(self._developers):
                        raise ValueError(
                            "`aggregate_devgroups[%d][%d]` = %s must be less than the number of "
                            "developers (%d)" % (i, j, v, len(self._developers)))
                if len(set(group)) < len(group):
                    raise ValueError("`aggregate_devgroups[%d]` has duplicate items" % i)

        self._aggregate_devgroups = aggregate_devgroups


ForSetDevelopers = AllOf(_ForSetDevelopers, CommonPullRequestFilters,
                         name="ForSetDevelopers", module=__name__)
