from datetime import datetime

from athenian.api.models.web.base_model_ import Model


class JIRAComment(Model):
    """A comment linked to a Jira Issue."""

    author: str
    created: datetime
    rendered_body: str
