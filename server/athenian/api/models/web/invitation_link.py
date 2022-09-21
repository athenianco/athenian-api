from athenian.api.models.web.base_model_ import Model


class _InvitationLink(Model, sealed=False):
    url: str


class InvitationLink(_InvitationLink, sealed=True):
    """Product invitation URL."""
