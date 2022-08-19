from __future__ import annotations

from typing import TypedDict

from athenian.api.models.web import JIRAMetricID, PullRequestMetricID, ReleaseMetricID

TEMPLATES_COLLECTION: dict[int, TemplateDefinition] = {
    1: {
        "metric": PullRequestMetricID.PR_REVIEW_TIME,
        "name": "Reduce code review time",
    },
    2: {
        "metric": PullRequestMetricID.PR_REVIEW_COMMENTS_PER,
        "name": "Improve code review quality",
    },
    3: {
        "metric": PullRequestMetricID.PR_MEDIAN_SIZE,
        "name": "Decrease PR Size",
    },
    4: {
        "metric": PullRequestMetricID.PR_LEAD_TIME,
        "name": "Accelerate software delivery",
    },
    5: {
        "metric": ReleaseMetricID.RELEASE_COUNT,
        "name": "Increase release frequency",
    },
    6: {
        "metric": JIRAMetricID.JIRA_RESOLVED,
        "name": "Increase Jira throughput",
    },
    7: {
        "metric": PullRequestMetricID.PR_ALL_MAPPED_TO_JIRA,
        "name": "Improve PR mapping to Jira",
    },
    8: {
        "metric": PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME,
        "name": "Reduce the time PRs are waiting for review",
    },
    9: {
        "metric": PullRequestMetricID.PR_OPEN_TIME,
        "name": "Reduce the time PRs remain open",
    },
    10: {
        "metric": JIRAMetricID.JIRA_LEAD_TIME,
        "name": "Accelerate Jira resolution time",
    },
    11: {
        "metric": PullRequestMetricID.PR_REVIEWED_RATIO,
        "name": "Increase the proportion of PRs reviewed",
    },
}


class TemplateDefinition(TypedDict):
    """The definition of a Goal Template."""

    metric: str
    name: str
