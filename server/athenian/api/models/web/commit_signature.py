from datetime import datetime
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class CommitSignature(Model):
    """Git commit signature."""

    login: Optional[str]
    name: str
    email: str
    timestamp: datetime
    timezone: float
