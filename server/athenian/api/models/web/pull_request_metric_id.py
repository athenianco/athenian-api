from athenian.api.models.web.base_model_ import Enum, Model


class PullRequestMetricID(Model, metaclass=Enum):
    """Linear metric identifier."""

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
    PR_CYCLE_TIME = "pr-cycle-time"
    PR_CYCLE_COUNT = "pr-cycle-count"
    PR_FLOW_RATIO = "pr-flow-ratio"
    PR_OPENED = "pr-opened"
    PR_MERGED = "pr-merged"
    PR_CLOSED = "pr-closed"
    PR_RELEASED = "pr-released"
    PR_WAIT_FIRST_REVIEW_TIME = "pr-wait-first-review"
    PR_WAIT_FIRST_REVIEW_COUNT = "pr-wait-first-review-count"
