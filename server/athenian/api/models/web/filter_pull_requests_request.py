from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import AllOf, Model
from athenian.api.models.web.common_filter_properties import CommonFilterProperties
from athenian.api.models.web.for_set_common import CommonPullRequestFilters
from athenian.api.models.web.pull_request_event import PullRequestEvent
from athenian.api.models.web.pull_request_stage import PullRequestStage
from athenian.api.models.web.pull_request_with import PullRequestWith


class _FilterPullRequestsRequest(Model, sealed=False):
    """PR filters for /filter/pull_requests."""

    in_: (List[str], "in")
    events: Optional[List[str]]
    stages: Optional[List[str]]
    with_: (Optional[PullRequestWith], "with")
    exclude_inactive: bool
    updated_from: Optional[date]
    updated_to: Optional[date]
    limit: Optional[int]
    environments: Optional[list[str]]

    def validate_events(self, events: Optional[list[str]]) -> Optional[list[str]]:
        """Sets the events of this FilterPullRequestsRequest.

        :param events: The events of this FilterPullRequestsRequest.
        """
        if events is None:
            return None

        for stage in events:
            if stage not in PullRequestEvent:
                raise ValueError("Invalid stage: %s" % stage)

        return events

    def validate_stages(self, stages: Optional[list[str]]) -> Optional[list[str]]:
        """Sets the stages of this FilterPullRequestsRequest.

        :param stages: The stages of this FilterPullRequestsRequest.
        """
        if stages is None:
            return None

        for stage in stages:
            if stage not in PullRequestStage:
                raise ValueError("Invalid stage: %s" % stage)

        return stages

    def validate_limit(self, limit: Optional[int]) -> Optional[int]:
        """Sets the limit of this FilterPullRequestsRequest.

        Value indicating whether PRs without events in the given time frame shall be ignored.

        :param limit: The limit of this FilterPullRequestsRequest.
        """
        if limit is not None and limit < 1:
            raise ValueError("`limit` must be greater than 0: %s" % limit)

        return limit


FilterPullRequestsRequest = AllOf(
    _FilterPullRequestsRequest,
    CommonFilterProperties,
    CommonPullRequestFilters,
    name="FilterPullRequestsRequest",
    module=__name__,
)
