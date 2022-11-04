from __future__ import annotations

from typing import Optional, Sequence, TypedDict

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
    {
        "metric": PullRequestMetricID.PR_REVIEW_TIME_BELOW_THRESHOLD_RATIO,
        "name": "Review Time",
        "metric_params": {"threshold": "172800s"},
    },
    {
        "metric": PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME_BELOW_THRESHOLD_RATIO,
        "name": "Wait Time for 1st Review",
        "metric_params": {"threshold": "21600s"},
    },
    {
        "metric": PullRequestMetricID.PR_SIZE_BELOW_THRESHOLD_RATIO,
        "name": "Median PR Size",
        "metric_params": {"threshold": 100},
    },
    {
        "metric": PullRequestMetricID.PR_REVIEW_COMMENTS_PER_ABOVE_THRESHOLD_RATIO,
        "name": "Review Comments / PR",
        "metric_params": {"threshold": 3},
    },
    {
        "metric": PullRequestMetricID.PR_CYCLE_DEPLOYMENT_TIME_BELOW_THRESHOLD_RATIO,
        "name": "PR Cycle Time",
        "metric_params": {"threshold": "432000s"},
    },
    {
        "metric": PullRequestMetricID.PR_OPEN_TIME_BELOW_THRESHOLD_RATIO,
        "name": "PR Open Time",
        "metric_params": {"threshold": "259200s"},
    },
    {
        "metric": JIRAMetricID.JIRA_LEAD_TIME_BELOW_THRESHOLD_RATIO,
        "name": "Jira Lead Time",
        "metric_params": {"threshold": "432000s"},
    },
]


class _TemplateDefinitionRequired(TypedDict):
    metric: str
    name: str


class TemplateDefinition(_TemplateDefinitionRequired, total=False):
    """The definition of a Goal Template."""

    metric_params: Optional[dict]
