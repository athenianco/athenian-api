from athenian.api.models.web.base_model_ import Model
from athenian.api.typing_utils import VerbatimOptional


class TeamUpdateRequest(Model):
    """Team update request."""

    name: str
    members: list[str]
    parent: VerbatimOptional[int]
