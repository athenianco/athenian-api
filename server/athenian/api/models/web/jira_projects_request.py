from athenian.api.models.web.base_model_ import Model


class JIRAProjectsRequest(Model):
    """Enable or disable a JIRA project."""

    account: int
    projects: dict[str, bool]
