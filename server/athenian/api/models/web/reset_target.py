from athenian.api.models.web import Enum


class ResetTarget(Enum):
    """What to clear in `/reset`."""

    COMMITS = "commits"
    DEPLOYMENTS = "deployments"
    JIRA_ACCOUNT = "jira_account"
    METADATA_ACCOUNT = "metadata_account"
    PRS = "prs"
    RELEASES = "releases"
    REPOSET = "reposet"
    TEAMS = "teams"
