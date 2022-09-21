from datetime import datetime, timedelta
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRAEpicIssueCommon(Model, sealed=False):
    """Common JIRA issue fields."""

    id: str
    title: str
    created: datetime
    updated: datetime
    work_began: Optional[datetime]
    resolved: Optional[datetime]
    lead_time: Optional[timedelta]
    life_time: timedelta
    reporter: str
    assignee: Optional[str]
    comments: int
    priority: str
    status: str
    type: str
    url: str
