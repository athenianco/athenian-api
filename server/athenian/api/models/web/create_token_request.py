from athenian.api.models.web.base_model_ import Model


class CreateTokenRequest(Model):
    """Request body of `/token/create` - creating a new Personal Access Token."""

    account: int
    name: str
