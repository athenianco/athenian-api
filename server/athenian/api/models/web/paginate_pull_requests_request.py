from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.filter_pull_requests_request import FilterPullRequestsRequest


class PaginatePullRequestsRequest(Model):
    """
    Request of `/paginate/pull_requests`.

    According to the target batch size, compute the optimal PR updated timestamp ranges.
    """

    batch: int
    request: FilterPullRequestsRequest

    def validate_batch(self, batch: int) -> int:
        """Sets the batch of this PaginatePullRequestsRequest.

        Target batch size. The returned ranges do not guarantee the exact match.

        :param batch: The batch of this PaginatePullRequestsRequest.
        """
        if batch is None:
            raise ValueError("Invalid value for `batch`, must not be `None`")
        if batch < 1:
            raise ValueError("`batch` must be greater than zero")

        return batch
