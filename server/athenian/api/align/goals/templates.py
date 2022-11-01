from __future__ import annotations

from typing import Sequence, TypedDict

from athenian.api.models.web import JIRAMetricID, PullRequestMetricID, ReleaseMetricID

TEMPLATES_COLLECTION: Sequence[TemplateDefinition] = [
    {
        "metric": PullRequestMetricID.PR_REVIEW_TIME,
        "name": "Reduce code review time",
    },
    {
        "metric": PullRequestMetricID.PR_REVIEW_COMMENTS_PER,
        "name": "Improve code review quality",
    },
    {
        "metric": PullRequestMetricID.PR_MEDIAN_SIZE,
        "name": "Decrease PR Size",
    },
    {
        "metric": PullRequestMetricID.PR_LEAD_TIME,
        "name": "Accelerate software delivery",
    },
    {
        "metric": ReleaseMetricID.RELEASE_COUNT,
        "name": "Increase release frequency",
    },
    {
        "metric": JIRAMetricID.JIRA_RESOLVED,
        "name": "Increase Jira throughput",
    },
    {
        "metric": PullRequestMetricID.PR_ALL_MAPPED_TO_JIRA,
        "name": "Improve PR mapping to Jira",
    },
    {
        "metric": PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME,
        "name": "Reduce the time PRs are waiting for review",
    },
    {
        "metric": PullRequestMetricID.PR_OPEN_TIME,
        "name": "Reduce the time PRs remain open",
    },
    {
        "metric": JIRAMetricID.JIRA_LEAD_TIME,
        "name": "Accelerate Jira resolution time",
    },
    {
        "metric": PullRequestMetricID.PR_REVIEWED_RATIO,
        "name": "Increase the proportion of PRs reviewed",
    },
]


class TemplateDefinition(TypedDict):
    """The definition of a Goal Template."""

    metric: str
    name: str
