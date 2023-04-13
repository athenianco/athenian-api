from datetime import datetime, timedelta
from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.jira_comment import JIRAComment


class JIRAEpicIssueCommon(Model, sealed=False):
    """Common JIRA issue fields."""

    id: str
    title: str
    created: datetime
    updated: datetime
    acknowledge_time: timedelta
    work_began: Optional[datetime]
    resolved: Optional[datetime]
    lead_time: Optional[timedelta]
    life_time: timedelta
    reporter: str
    assignee: Optional[str]
    comments: int
    comment_list: Optional[list[JIRAComment]]
    priority: Optional[str]
    rendered_description: Optional[str]
    status: str
    type: str
    url: str
    story_points: Optional[float]
