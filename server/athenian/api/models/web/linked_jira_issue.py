from typing import Optional

from athenian.api.models.web.base_model_ import Model


class LinkedJIRAIssue(Model):
    """Brief details about a JIRA issue."""

    id: str
    title: str
    epic: Optional[str]
    labels: Optional[list[str]]
    type: str
