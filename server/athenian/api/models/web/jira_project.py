from datetime import datetime

from athenian.api.models.web.base_model_ import Model
from athenian.api.typing_utils import VerbatimOptional


class JIRAProject(Model):
    """JIRA project setting."""

    name: str
    key: str
    id: str
    avatar_url: str
    enabled: bool
    issues_count: int
    last_update: VerbatimOptional[datetime]
