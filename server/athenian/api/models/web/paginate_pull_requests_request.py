from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.filter_pull_requests_request import FilterPullRequestsRequest


class PaginatePullRequestsRequest(Model):
    """
    Request of `/paginate/pull_requests`.

    According to the target batch size, compute the optimal PR updated timestamp ranges.
    """

    attribute_types = {"batch": int, "request": FilterPullRequestsRequest}
    attribute_map = {"batch": "batch", "request": "request"}

    def __init__(self,
                 batch: Optional[int] = None,
                 request: Optional[FilterPullRequestsRequest] = None):
        """PaginatePullRequestsRequest - a model defined in OpenAPI

        :param batch: The batch of this PaginatePullRequestsRequest.
        :param request: The request of this PaginatePullRequestsRequest.
        """
        self._batch = batch
        self._request = request

    @property
    def batch(self) -> int:
        """Gets the batch of this PaginatePullRequestsRequest.

        Target batch size. The returned ranges do not guarantee the exact match.

        :return: The batch of this PaginatePullRequestsRequest.
        """
        return self._batch

    @batch.setter
    def batch(self, batch: int):
        """Sets the batch of this PaginatePullRequestsRequest.

        Target batch size. The returned ranges do not guarantee the exact match.

        :param batch: The batch of this PaginatePullRequestsRequest.
        """
        if batch is None:
            raise ValueError("Invalid value for `batch`, must not be `None`")
        if batch < 1:
            raise ValueError("`batch` must be greater than zero")

        self._batch = batch

    @property
    def request(self) -> FilterPullRequestsRequest:
        """Gets the request of this PaginatePullRequestsRequest.

        :return: The request of this PaginatePullRequestsRequest.
        """
        return self._request

    @request.setter
    def request(self, request: FilterPullRequestsRequest):
        """Sets the request of this PaginatePullRequestsRequest.

        :param request: The request of this PaginatePullRequestsRequest.
        """
        if request is None:
            raise ValueError("Invalid value for `request`, must not be `None`")

        self._request = request
