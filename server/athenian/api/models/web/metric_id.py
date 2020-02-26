from athenian.api.models.web.base_model_ import Model


class MetricID(Model):
    """Linear metric identifier."""

    """
    allowed enum values
    """
    PR_WIP_TIME = "pr-wip-time"
    PR_WIP_COUNT = "pr-wip-count"
    PR_REVIEW_TIME = "pr-review-time"
    PR_REVIEW_COUNT = "pr-review-count"
    PR_MERGING_TIME = "pr-merging-time"
    PR_MERGING_COUNT = "pr-merging-count"
    PR_RELEASE_TIME = "pr-release-time"
    PR_RELEASE_COUNT = "pr-release-count"
    PR_LEAD_TIME = "pr-lead-time"
    PR_LEAD_COUNT = "pr-lead-count"
    PR_FLOW_RATIO = "pr-flow-ratio"
    PR_OPENED = "pr-opened"
    PR_MERGED = "pr-merged"
    PR_CLOSED = "pr-closed"
    PR_WAIT_FIRST_REVIEW_TIME = "pr-wait-first-review"
    ALL = {
        PR_WIP_TIME,
        PR_WIP_COUNT,
        PR_REVIEW_TIME,
        PR_REVIEW_COUNT,
        PR_MERGING_TIME,
        PR_MERGING_COUNT,
        PR_RELEASE_TIME,
        PR_RELEASE_COUNT,
        PR_LEAD_TIME,
        PR_LEAD_COUNT,
        PR_FLOW_RATIO,
        PR_OPENED,
        PR_MERGED,
        PR_CLOSED,
        PR_WAIT_FIRST_REVIEW_TIME,
    }

    def __init__(self):
        """MetricID - a model defined in OpenAPI"""
        self.openapi_types = {}

        self.attribute_map = {}
