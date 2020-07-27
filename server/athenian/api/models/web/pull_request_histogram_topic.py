from athenian.api.models.web.base_model_ import Enum, Model


class PullRequestHistogramTopic(Model, metaclass=Enum):
    """Supported histogram measurement kinds."""

    CYCLE_TIME = "pr-cycle-time"
    LEAD_TIME = "pr-lead-time"
    WIP_TIME = "pr-wip-time"
    REVIEW_TIME = "pr-review-time"
    MERGE_TIME = "pr-merge-time"
    RELEASE_TIME = "pr-release-time"
    RELEASE_TIME = "pr-release-time"
