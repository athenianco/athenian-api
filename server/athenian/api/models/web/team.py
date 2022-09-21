from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.contributor import Contributor
from athenian.api.typing_utils import VerbatimOptional


class Team(Model):
    """Definition of a team of several developers."""

    id: int
    name: str
    members: list[Contributor]
    parent: VerbatimOptional[int]
