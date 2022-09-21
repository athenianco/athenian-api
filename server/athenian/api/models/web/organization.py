from athenian.api.models.web.base_model_ import Model


class Organization(Model):
    """GitHub organization details."""

    name: str
    avatar_url: str
    login: str
