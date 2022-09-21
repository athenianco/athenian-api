from datetime import datetime

from athenian.api.models.web.base_model_ import Model


class JIRALabel(Model):
    """Details about a JIRA label."""

    title: str
    last_used: datetime
    issues_count: int
    kind: str

    def __lt__(self, other: "JIRALabel") -> bool:
        """Support sorting."""
        return self.title < other.title
