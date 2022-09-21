from datetime import date

from athenian.api.models.web.base_model_ import Model


class PullRequestPaginationPlan(Model):
    """Response of `/paginate/pull_requests`. Computed split of the PR updated timestamp range."""

    updated: list[date]
