from athenian.api.models.web.base_model_ import Model
from athenian.api.typing_utils import VerbatimOptional


class MappedJIRAIdentityChange(Model):
    """Individual GitHub<>JIRA user mapping change."""

    developer_id: str
    jira_name: VerbatimOptional[str]
