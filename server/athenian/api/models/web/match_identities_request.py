from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.contributor_identity import ContributorIdentity


class MatchIdentitiesRequest(Model):
    """Request body of `/match/identities`."""

    account: int
    identities: list[ContributorIdentity]
