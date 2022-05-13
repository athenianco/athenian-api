from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model


class PullRequestPaginationPlan(Model):
    """Response of `/paginate/pull_requests`. Computed split of the PR updated timestamp range."""

    attribute_types = {"updated": List[date]}
    attribute_map = {"updated": "updated"}

    def __init__(self, updated: Optional[List[date]] = None):
        """PullRequestPaginationPlan - a model defined in OpenAPI

        :param updated: The updated of this PullRequestPaginationPlan.
        """
        self._updated = updated

    @property
    def updated(self) -> List[date]:
        """Gets the updated of this PullRequestPaginationPlan.

        :return: The updated of this PullRequestPaginationPlan.
        """
        return self._updated

    @updated.setter
    def updated(self, updated: List[date]):
        """Sets the updated of this PullRequestPaginationPlan.

        :param updated: The updated of this PullRequestPaginationPlan.
        """
        self._updated = updated
