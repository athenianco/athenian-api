from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.contributor_identity import ContributorIdentity
from athenian.api.typing_utils import VerbatimOptional


class MatchedIdentity(Model):
    """Identity mapping of a specific contributor."""

    from_: (ContributorIdentity, "from")
    to: VerbatimOptional[str]
    confidence: float
