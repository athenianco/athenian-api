from typing import Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.developer_updates import DeveloperUpdates
from athenian.api.typing_utils import VerbatimOptional


class DeveloperSummary(Model):
    """Developer activity statistics and profile details."""

    login: str
    name: VerbatimOptional[str]
    avatar: str
    updates: DeveloperUpdates
    jira_user: Optional[str]
