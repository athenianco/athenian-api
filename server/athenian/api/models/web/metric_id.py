from athenian.api.models.web.base_model_ import Model


class MetricID(Model):
    """Linear metric identifier."""

    """
    allowed enum values
    """
    PR_WIP_TIME = "pr-wip-time"
    PR_REVIEW_TIME = "pr-review-time"
    PR_MERGING_TIME = "pr-merging-time"
    PR_RELEASE_TIME = "pr-release-time"
    PR_LEAD_TIME = "pr-lead-time"
    PR_FLOW_RATIO = "pr-flow-ratio"
    PR_OPENED = "pr-opened"
    PR_MERGED = "pr-merged"
    PR_CLOSED = "pr-closed"
    PR_WAIT_FIRST_REVIEW = "pr-wait-first-review"
    ALL = {
        PR_WIP_TIME,
        PR_REVIEW_TIME,
        PR_MERGING_TIME,
        PR_RELEASE_TIME,
        PR_LEAD_TIME,
        PR_FLOW_RATIO,
        PR_OPENED,
        PR_MERGED,
        PR_CLOSED,
        PR_WAIT_FIRST_REVIEW,
    }

    def __init__(self):
        """MetricID - a model defined in OpenAPI"""
        self.openapi_types = {}

        self.attribute_map = {}
