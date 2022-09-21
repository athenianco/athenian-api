from athenian.api.models.web.base_model_ import Model


class CreatedToken(Model):
    """Value and ID of the generated Personal Access Token."""

    id: int
    token: str
