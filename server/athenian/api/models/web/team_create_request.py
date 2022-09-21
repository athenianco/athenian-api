from athenian.api.models.web.base_model_ import Model
from athenian.api.typing_utils import VerbatimOptional


class TeamCreateRequest(Model):
    """Team creation request."""

    account: int
    name: str
    members: list[str]
    parent: VerbatimOptional[int]
