from athenian.api.models.web.base_model_ import Model


class JIRAInstallation(Model):
    """Information about a link with JIRA."""

    url: str
    projects: list[str]
